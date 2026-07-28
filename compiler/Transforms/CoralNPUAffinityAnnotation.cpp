// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "compiler/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/Analysis/DeviceAnalysis.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace mlir::coralnpu_compiler {

#define GEN_PASS_DEF_CORALNPUAFFINITYANNOTATION
#include "compiler/Transforms/Passes.h.inc"

namespace {

bool isSupportedComputeOp(Operation *op) {
  // NB: batch_matmul (attention Q@K^T and P@V) is intentionally excluded so it
  // stays on the host alongside softmax. Otherwise softmax -- which sits between
  // the two attention batch_matmuls -- gets pulled onto CoralNPU by the affinity
  // solver (to avoid transfers), and its ~6KB kernel plus libm helpers overflow
  // the 8KB ITCM.
  if (isa<linalg::MatmulOp, linalg::Mmt4DOp>(op)) return true;

  if (isa<linalg::Conv2DNhwcHwcfOp>(op)) return true;

  return false;
}

bool isSupportedElementType(Type type) {
  return type.isInteger(8) || type.isInteger(16) || type.isInteger(32) ||
         type.isF32();
}

bool isSupportedType(Type type) {
  if (auto rankedTensorType = dyn_cast<RankedTensorType>(type)) {
    return rankedTensorType.hasStaticShape() &&
           isSupportedElementType(rankedTensorType.getElementType());
  }

  // TODO(sflur): do we support the same unshaped types as element types?
  return isSupportedElementType(type);
}

bool isSupportedOperandAndResultTypes(Operation *op) {
  for (Value operand : op->getOperands()) {
    if (!isSupportedType(operand.getType())) {
      return false;
    }
  }

  for (Value result : op->getResults()) {
    if (!isSupportedType(result.getType())) {
      return false;
    }
  }

  return true;
}

int64_t estimateBytesForType(Type type) {
  if (auto rankedTensorType = dyn_cast<RankedTensorType>(type)) {
    Type elementType = rankedTensorType.getElementType();

    unsigned elementBits = elementType.getIntOrFloatBitWidth();
    // Checked in isSupportedElementType
    assert(elementBits % 8 == 0);
    int64_t elementBytes = elementBits / 8;

    return rankedTensorType.getNumElements() * elementBytes;
  }

  unsigned bits = type.getIntOrFloatBitWidth();
  // Checked in isSupportedType
  assert(bits % 8 == 0);
  return bits / 8;
}

// op must be an operation that can execute on coralnpu, type wise
int64_t estimateIOBytes(Operation *op) {
  int64_t totalBytes = 0;

  for (Value operand : op->getOperands()) {
    totalBytes += estimateBytesForType(operand.getType());
  }

  for (Value operand : op->getResults()) {
    totalBytes += estimateBytesForType(operand.getType());
  }

  return totalBytes;
}

// Discover the NPU device alias.
IREE::HAL::DeviceAffinityAttr getCoralNPUDeviceAffinityAttr(
    MLIRContext *context, ModuleOp moduleOp) {
  IREE::HAL::DeviceAnalysis deviceAnalysis(moduleOp);
  if (failed(deviceAnalysis.run())) {
    return nullptr;
  }

  for (auto globalOp : deviceAnalysis.getDeviceGlobals()) {
    auto deviceSet = deviceAnalysis.lookupDeviceTargets(globalOp);
    if (!deviceSet) continue;
    for (auto targetAttr : deviceSet->getValues()) {
      if (targetAttr.getDeviceID().getValue() == "coralnpu") {
        return IREE::HAL::DeviceAffinityAttr::get(
            context, SymbolRefAttr::get(globalOp.getGlobalName()),
            /*queue_mask=*/-1ll);
      }
    }
  }

  return nullptr;
}

// Discover the host (non-CoralNPU) device alias, so we can explicitly pin ops
// there.
IREE::HAL::DeviceAffinityAttr getHostDeviceAffinityAttr(MLIRContext *context,
                                                        ModuleOp moduleOp) {
  IREE::HAL::DeviceAnalysis deviceAnalysis(moduleOp);
  if (failed(deviceAnalysis.run())) {
    return nullptr;
  }

  for (auto globalOp : deviceAnalysis.getDeviceGlobals()) {
    auto deviceSet = deviceAnalysis.lookupDeviceTargets(globalOp);
    if (!deviceSet) continue;
    for (auto targetAttr : deviceSet->getValues()) {
      if (targetAttr.getDeviceID().getValue() != "coralnpu") {
        return IREE::HAL::DeviceAffinityAttr::get(
            context, SymbolRefAttr::get(globalOp.getGlobalName()),
            /*queue_mask=*/-1ll);
      }
    }
  }

  return nullptr;
}
struct CoralNPUAffinityAnnotationPass
    : public impl::CoralNPUAffinityAnnotationBase<
          CoralNPUAffinityAnnotationPass> {
  using CoralNPUAffinityAnnotationBase::CoralNPUAffinityAnnotationBase;

  bool shouldExecuteOnCoralNPU(Operation *op) {
    if (!isSupportedComputeOp(op)) return false;

    if (!isSupportedOperandAndResultTypes(op)) return false;

    return ioMinThresholdBytes < estimateIOBytes(op);
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *context = &getContext();

    if (ioMinThresholdBytes < 0) {
      moduleOp.emitError("io-min-threshold-bytes must be non-negative, got ")
          << ioMinThresholdBytes;
      signalPassFailure();
      return;
    }

    iree_compiler::IREE::HAL::DeviceAffinityAttr coralnpuAffinityAttr =
        getCoralNPUDeviceAffinityAttr(context, moduleOp);
    if (!coralnpuAffinityAttr) return;

    // Host device, used to explicitly pin non-CoralNPU compute so the stream
    // affinity solver does not consolidate everything onto CoralNPU (which would
    // pull softmax/elementwise/reduction kernels into the tiny 8KB ITCM
    // executable).
    iree_compiler::IREE::HAL::DeviceAffinityAttr hostAffinityAttr =
        getHostDeviceAffinityAttr(context, moduleOp);

    moduleOp.walk([&](Operation *op) {
      // If op already has affinity, don't change it
      if (op->getAttr("stream.affinity")) return;

      if (shouldExecuteOnCoralNPU(op)) {
        op->setAttr("stream.affinity", coralnpuAffinityAttr);
      } else if (hostAffinityAttr &&
                 (isa<linalg::LinalgOp>(op) || isa<linalg::SoftmaxOp>(op)) &&
                 !isa<linalg::FillOp>(op)) {
        // Keep every other compute kernel (softmax, gelu, layernorm, residual
        // adds, transpose, ...) on the host so only matmuls land on CoralNPU.
        // linalg.softmax is a structured op that does not implement LinalgOp,
        // so it must be matched explicitly.
        op->setAttr("stream.affinity", hostAffinityAttr);
      }
    });
  }
};
}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
createCoralNPUAffinityAnnotationPass() {
  return std::make_unique<CoralNPUAffinityAnnotationPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createCoralNPUAffinityAnnotationPass(
    CoralNPUAffinityAnnotationOptions options) {
  return std::make_unique<CoralNPUAffinityAnnotationPass>(std::move(options));
}

}  // namespace mlir::coralnpu_compiler

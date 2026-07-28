#!/usr/bin/env bash
# End-to-end CoralNPU example.
#
# Compiles TinyStories-1M so its projection / MLP / LM-head matmuls run on the
# CoralNPU (RISC-V32) simulator inside the 8 KB instruction memory, with
# attention / softmax / layernorm / gelu on the host, then verifies the output
# against a JAX reference. One command runs all five steps.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
WORK="$HERE/work"

# The `coralnpu` conda env supplies jax / numpy / torch / huggingface_hub, and
# its bin/ holds the lld the host embedded-ELF link needs -> put it on PATH.
PY="${PYTHON:-/home/xinting/miniconda3/envs/coralnpu/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
export PATH="$(dirname "$PY"):$PATH"

CORALC="$REPO/bazel-bin/compiler/tools/coralnpu-compile"
IREE_RUN="$(find -L "$REPO/bazel-bin" -name iree-run-module -type f 2>/dev/null | grep -v runfiles | head -1 || true)"
[ -x "$CORALC" ] || { echo "!! build the compiler first:"; \
  echo "   bazel build //compiler/tools:coralnpu-compile @iree_core//tools:iree-run-module"; exit 1; }
[ -n "${IREE_RUN:-}" ] || { echo "!! iree-run-module not built (see command above)"; exit 1; }
mkdir -p "$WORK/inp"

echo "== 1/5  weights =="
"$PY" "$HERE/model.py" weights "$WORK/tinystories1m.npz"

echo "== 2/5  export TinyStories -> StableHLO (+ inputs + JAX reference) =="
"$PY" "$HERE/model.py" export "$WORK/tinystories1m.npz" "$WORK/model.mlir" "$WORK/inp" "$WORK/ref.npy"

echo "== 3/5  compile for host + CoralNPU =="
# --coralnpu-num-vector-registers=8: upstream's register-tiling pass defaults to
# 32 vector registers, which unrolls the matmuls to ~13 KB and overflows the
# 8 KB ITCM. Capping the register budget to 8 keeps the kernels small (~5 KB
# shipped) with no change to results. This replaces the old hand-rolled
# CoralNPUShrinkVectorTilesPass (which upstream's tiling superseded).
"$CORALC" \
  --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=llvm-cpu \
  --iree-llvmcpu-target-cpu-features=host \
  --iree-hal-target-device=coralnpu \
  --coralnpu-target-abi=ilp32 \
  --coralnpu-target-cpu-features=+m,+f,+zvl128b,+zve32f \
  --coralnpu-num-vector-registers=8 \
  "$WORK/model.mlir" -o "$WORK/model.vmfb"
# Measure the SHIPPED image (the RISC-V ELF inside the vmfb), not the pre-link
# .o under exe/ — the linker garbage-collects unused libm before shipping.
"$PY" "$HERE/model.py" itcm "$WORK/model.vmfb"

echo "== 4/5  run on the CoralNPU simulator =="
INPUTS=$(for f in $(ls "$WORK"/inp/in_*.npy | sort); do printf ' --input=@%s' "$f"; done)
LD_LIBRARY_PATH="$REPO/runtime/sim" "$IREE_RUN" \
  --device=local-sync --device=coralnpu \
  --module="$WORK/model.vmfb" --function=main \
  $INPUTS --output=@"$WORK/out.npy" > "$WORK/run.log" 2>&1
echo "   CoralNPU matmul dispatches executed (Halted): $(grep -c Halted "$WORK/run.log")"

echo "== 5/5  verify against JAX =="
"$PY" "$HERE/model.py" compare "$WORK/ref.npy" "$WORK/out.npy"

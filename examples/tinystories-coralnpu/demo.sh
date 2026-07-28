#!/usr/bin/env bash
# End-to-end story demo: generate a short story with TinyStories running on the
# CoralNPU simulator, reporting per-token functional-sim metrics.
#   usage: ./demo.sh ["prompt text"] [num_tokens_to_generate]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
WORK="$HERE/work"
PY="${PYTHON:-/home/xinting/miniconda3/envs/coralnpu/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
export PATH="$(dirname "$PY"):$PATH"
IREE_RUN="$(find -L "$REPO/bazel-bin" -name iree-run-module -type f 2>/dev/null | grep -v runfiles | head -1 || true)"

PROMPT="${1:-Once upon a time, there was a little}"
NGEN="${2:-12}"

[ -n "${IREE_RUN:-}" ] || { echo "!! build first (see run.sh)"; exit 1; }

# Reuse run.sh once to produce the compiled vmfb + the weight inputs.
if [ ! -f "$WORK/model.vmfb" ] || [ ! -f "$WORK/inp/in_108.npy" ]; then
  echo "== model not built yet -- running run.sh once to compile + stage inputs =="
  bash "$HERE/run.sh"
fi

echo "== generating a short story on the CoralNPU simulator (${NGEN} tokens) =="
echo "   (each token = one full forward on the sim; expect a few seconds each)"
"$PY" "$HERE/demo_story.py" "$WORK/model.vmfb" "$WORK/inp" "$IREE_RUN" "$REPO/runtime/sim" "$PROMPT" "$NGEN"

# TinyStories on CoralNPU — end-to-end example

Runs a real language model (TinyStories-1M, a GPT-Neo) across a **host + CoralNPU**
topology: the projection / MLP / LM-head **matmuls execute on the CoralNPU (RISC-V32)
simulator inside the 8 KB instruction memory**, while attention, softmax, and
layernorm run on the host (gelu is fused into the MLP-up matmul, so it runs on the
CoralNPU). The output is checked against a JAX reference.

## Run

```sh
./run.sh
```

It does five steps and prints the CoralNPU code size and a MATCH/MISMATCH:

1. **weights** — fetch the HuggingFace checkpoint and extract to `.npz` (cached after the first run)
2. **export** — pure-JAX forward pass → StableHLO, with weights passed as *inputs* (not baked
   constants) so identical per-layer kernels deduplicate; also writes the 109 input `.npy`s and the JAX reference
3. **compile** — `coralnpu-compile` for the two-device target; reports the *shipped* CoralNPU `.text + .rodata` byte count (carved from the vmfb) vs. the 8192 B ITCM limit
4. **run** — `iree-run-module` on `--device=local-sync --device=coralnpu`; the sim executes the matmul dispatches
5. **verify** — compare the CoralNPU logits to JAX (max\|diff\| and next-token argmax)

Everything generated lands in `work/` (git-ignored).

## Requirements

- The compiler and runtime built in this tree:
  `bazel build //compiler/tools:coralnpu-compile @iree_core//tools:iree-run-module`
- The `coralnpu` conda env (supplies `jax` / `numpy` / `torch` / `huggingface_hub`, and the
  `lld` the host link needs). Override the interpreter with `PYTHON=/path/to/python ./run.sh`.
- Network access on the first run only (to fetch the checkpoint).

## Why the split

A fully-unrolled in-model matmul is ~10 KB of RISC-V and overflows the 8 KB ITCM. This example
relies on the compiler's tile-cap pass (matmuls loop instead of unroll) and host-placement of the
non-matmul ops to keep the shipped CoralNPU code at **~4.2 KB (4320 B), bundled** (5 matmul
kernels, with gelu fused into the MLP-up kernel) — leaving ~3.8 KB (3872 B) of the 8 KB (8192 B)
ITCM free. Note the pre-link `.o`
dumped under `work/exe/` looks ~7 KB because it still carries libm functions the kernels never
call; the linker garbage-collects those, so `run.sh` measures the ELF actually inside the vmfb.

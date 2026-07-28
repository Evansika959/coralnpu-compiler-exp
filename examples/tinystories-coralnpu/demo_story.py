#!/usr/bin/env python3
"""Autoregressive story generation on the CoralNPU simulator.

Runs the compiled model one step at a time over a fixed 32-token window: each
step runs the full forward on the sim (projection/MLP/LM-head matmuls on the
CoralNPU, attention/softmax/norms on the host), greedily appends the argmax
token, and slides forward until the window is full.

Per-token metrics reported are FUNCTIONAL-sim metrics, on purpose:
  * NPU dispatches / token  -- exact (each "Halted" = one CoralNPU dispatch)
  * sim wall-clock / token  -- host time to *emulate* the RV32 core, NOT
                               hardware cycles (the MPACT sim is functional,
                               not cycle-accurate).
"""
import glob
import os
import subprocess
import sys
import time

import numpy as np

SEQ = 32


def main(vmfb, inp_dir, iree_run, sim_lib, prompt, n_gen):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_pretrained("gpt2")  # TinyStories uses the GPT-2 vocab

    ptoks = tok.encode(prompt).ids
    ids = np.zeros(SEQ, np.int32)
    cur = min(len(ptoks), SEQ - 1)            # leave room for >=1 generated token
    ids[:cur] = ptoks[:cur]

    inputs = sorted(glob.glob(os.path.join(inp_dir, "in_*.npy")))  # in_000=ids + weights
    out = "/tmp/demo_out.npy"
    env = dict(os.environ, LD_LIBRARY_PATH=sim_lib)

    print(f"\nprompt: {prompt!r}  ({cur} tokens)\n")
    print(f"{'pos':>4}  {'tok':>6}  {'piece':<18}  {'NPU disp':>8}  {'sim ms':>8}")
    print("-" * 56)

    target = min(cur + int(n_gen), SEQ)
    disp, times = [], []
    while cur < target:
        np.save(os.path.join(inp_dir, "in_000.npy"), ids)
        cmd = [iree_run, "--device=local-sync", "--device=coralnpu",
               f"--module={vmfb}", "--function=main"]
        cmd += [f"--input=@{f}" for f in inputs]
        cmd += [f"--output=@{out}"]
        if os.path.exists(out):
            os.remove(out)
        t0 = time.time()
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        ms = (time.time() - t0) * 1000.0
        halted = (r.stdout + r.stderr).count("Halted")
        if not os.path.exists(out):
            print("  !! run failed:\n", (r.stderr or r.stdout)[-400:])
            return
        logits = np.load(out).reshape(SEQ, -1)
        nxt = int(logits[cur - 1].argmax())      # position cur-1 predicts position cur
        ids[cur] = nxt
        print(f"{cur:>4}  {nxt:>6}  {tok.decode([nxt])!r:<18}  {halted:>8}  {ms:>8.0f}")
        disp.append(halted)
        times.append(ms)
        cur += 1

    story = tok.decode([int(x) for x in ids[:cur]])
    print("\n" + "=" * 56)
    print("GENERATED STORY:")
    print(f"  {story}")
    print("=" * 56)
    d = float(np.mean(disp)) if disp else 0.0
    m = float(np.mean(times)) if times else 0.0
    print("\nper-token on the CoralNPU sim (functional -- NOT hardware cycles):")
    print(f"  NPU dispatches / token : {d:.0f}   (exact; each = one CoralNPU kernel launch)")
    print(f"  sim wall-clock / token : {m:.0f} ms  (host emulation time, not silicon)")
    print(f"  tokens generated       : {len(times)}")


if __name__ == "__main__":
    main(*sys.argv[1:7])

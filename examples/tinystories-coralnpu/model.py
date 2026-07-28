#!/usr/bin/env python3
"""TinyStories-1M (GPT-Neo) for the CoralNPU end-to-end example.

Pure-JAX forward pass -> StableHLO, with weights/biases passed as *inputs* (not
baked constants) so identical per-layer kernels deduplicate and the CoralNPU
code fits the 8 KB ITCM.

Subcommands:
  weights <out.npz>                                  fetch + extract the HF checkpoint (once)
  export  <npz> <out.mlir> <inputs_dir> <ref.npy>    StableHLO + 109 inputs + JAX reference
  compare <ref.npy> <out.npy>                         compare CoralNPU output to JAX
"""
import os
import sys

import numpy as np

CFG = dict(n_layer=8, n_head=16, head_dim=4, eps=1e-5, vocab=50257, seq=32)
REPO = "roneneldan/TinyStories-1M"


def _model():
    """Returns (jax, jnp, forward) for the GPT-Neo forward pass."""
    import jax
    import jax.numpy as jnp
    C = CFG

    def gelu(x):
        return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) *
                                         (x + 0.044715 * x ** 3)))

    def ln(x, g, b):
        mu = jnp.mean(x, -1, keepdims=True)
        v = jnp.mean((x - mu) ** 2, -1, keepdims=True)
        return (x - mu) / jnp.sqrt(v + C["eps"]) * g + b

    def lin(x, W, b=None):
        y = x @ W.T  # torch nn.Linear weight is (out,in)
        return y if b is None else y + b

    def forward(ids, p):
        T = ids.shape[0]
        nh, hd = C["n_head"], C["head_dim"]
        x = p["transformer.wte.weight"][ids] + p["transformer.wpe.weight"][jnp.arange(T)]
        mask = jnp.tril(jnp.ones((T, T), bool))
        for li in range(C["n_layer"]):
            b = f"transformer.h.{li}."
            h = ln(x, p[b + "ln_1.weight"], p[b + "ln_1.bias"])
            q, k, v = [lin(h, p[b + f"attn.attention.{n}_proj.weight"]) for n in "qkv"]
            sp = lambda t: t.reshape(T, nh, hd).transpose(1, 0, 2)
            q, k, v = sp(q), sp(k), sp(v)
            # GPT-Neo: no 1/sqrt(d) scaling.
            s = jnp.where(mask, jnp.einsum("hqd,hkd->hqk", q, k), jnp.float32(-1e9))
            pr = jax.nn.softmax(s, -1)
            o = jnp.einsum("hqk,hkd->hqd", pr, v).transpose(1, 0, 2).reshape(T, nh * hd)
            x = x + lin(o, p[b + "attn.attention.out_proj.weight"],
                        p[b + "attn.attention.out_proj.bias"])
            h = ln(x, p[b + "ln_2.weight"], p[b + "ln_2.bias"])
            h = lin(h, p[b + "mlp.c_fc.weight"], p[b + "mlp.c_fc.bias"])
            h = gelu(h)
            h = lin(h, p[b + "mlp.c_proj.weight"], p[b + "mlp.c_proj.bias"])
            x = x + h
        x = ln(x, p["transformer.ln_f.weight"], p["transformer.ln_f.bias"])
        return x @ p["transformer.wte.weight"].T  # tied LM head

    return jax, jnp, forward


def cmd_weights(out_npz):
    if os.path.exists(out_npz):
        print(f"   weights cached: {out_npz}")
        return
    import torch
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(REPO, "pytorch_model.bin")
    sd = torch.load(path, map_location="cpu", weights_only=True)
    np.savez(out_npz, **{k: v.numpy() for k, v in sd.items()})
    print(f"   fetched + extracted {len(sd)} tensors -> {out_npz}")


def cmd_export(npz, out_mlir, inputs_dir, ref_npy):
    jax, jnp, forward = _model()
    w = np.load(npz)
    # Params the forward pass uses, in JAX pytree-flatten (sorted-key) order.
    used = sorted(k for k in w.files
                  if "attention.bias" not in k and "masked_bias" not in k)
    params = {k: jnp.asarray(w[k]) for k in used}
    ids = ((np.arange(CFG["seq"]) * 97 + 13) % CFG["vocab"]).astype(np.int32)

    open(out_mlir, "w").write(
        str(jax.jit(forward).lower(jnp.asarray(ids), params).compiler_ir("stablehlo")))

    os.makedirs(inputs_dir, exist_ok=True)
    np.save(os.path.join(inputs_dir, "in_000.npy"), ids)  # arg 0 = token ids
    for i, k in enumerate(used):                          # args 1..N = params, sorted
        np.save(os.path.join(inputs_dir, f"in_{i + 1:03d}.npy"), w[k])

    np.save(ref_npy, np.asarray(jax.jit(forward)(jnp.asarray(ids), params)))
    print(f"   {out_mlir}  |  {1 + len(used)} inputs  |  "
          f"JAX next-token argmax = {int(np.load(ref_npy)[-1].argmax())}")


def cmd_itcm(vmfb):
    """Report the CoralNPU instruction-memory footprint of the *shipped* image.

    The kernels are the RISC-V embedded ELF inside the vmfb (not the pre-link
    `.o` dumped by --iree-hal-dump-*, which still carries libm the linker later
    garbage-collects). Per crt/coralnpu_tcm.ld both .text and .rodata live in
    ITCM, so we sum the read-only allocatable sections.
    """
    import struct
    blob = open(vmfb, "rb").read()
    best, off = None, 0
    while True:
        i = blob.find(b"\x7fELF", off)
        if i < 0:
            break
        if blob[i + 4] == 1 and struct.unpack_from("<H", blob, i + 18)[0] == 0xF3:
            best = i  # last RISC-V ELF32 in the module = the dispatch executable
        off = i + 4
    if best is None:
        print("   (no RISC-V executable found in module)")
        return
    e_shoff = struct.unpack_from("<I", blob, best + 32)[0]
    e_shentsize = struct.unpack_from("<H", blob, best + 46)[0]
    e_shnum = struct.unpack_from("<H", blob, best + 48)[0]
    SHF_WRITE, SHF_ALLOC = 0x1, 0x2
    itcm = 0
    for s in range(e_shnum):
        sh = best + e_shoff + s * e_shentsize
        fl = struct.unpack_from("<I", blob, sh + 8)[0]
        sz = struct.unpack_from("<I", blob, sh + 20)[0]
        if (fl & SHF_ALLOC) and not (fl & SHF_WRITE):  # .text + .rodata -> ITCM
            itcm += sz
    fits = "FITS" if itcm <= 8192 else "OVER"
    print(f"   CoralNPU instruction memory (.text+.rodata): {itcm} B  "
          f"(ITCM limit 8192, {8192 - itcm} B free)  ->  {fits}")


def cmd_compare(ref_npy, out_npy):
    r = np.load(ref_npy)
    s = np.load(out_npy).reshape(r.shape)
    d = np.abs(r - s)
    a_ref, a_npu = int(r[-1].argmax()), int(s[-1].argmax())
    ok = a_ref == a_npu
    print(f"   logits shape : {r.shape}")
    print(f"   max|diff|    : {d.max():.3e}")
    print(f"   next-token   : JAX={a_ref}  CoralNPU={a_npu}")
    print(f"   RESULT       : {'MATCH' if ok else 'MISMATCH'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    {"weights": cmd_weights, "export": cmd_export, "itcm": cmd_itcm,
     "compare": cmd_compare}[sys.argv[1]](*sys.argv[2:])

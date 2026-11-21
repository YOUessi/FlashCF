"""One-shot Tang GPU check (Triton execution + FlashCF references on CUDA).

Not a CFPO model run and NOT a performance benchmark. Runs without downloads.
Usage: python experiments/gpu_probe.py [--json experiments/results/tang_gpu_probe.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference"))
from cfpo_reference import dense_cfpo, streaming_cfpo
from custom_backward_reference import flashcf_custom


def run_probe():
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required; run this on Tang.")
    import triton
    import triton.language as tl

    # JIT compilation requires a Python source file (not a stdin snippet).
    @triton.jit
    def vecadd(x, y, z, n: tl.constexpr, B: tl.constexpr):
        idx = tl.program_id(0) * B + tl.arange(0, B)
        a = tl.load(x + idx, idx < n, 0)
        b = tl.load(y + idx, idx < n, 0)
        tl.store(z + idx, a + b, idx < n)

    n = 4096
    a, b = torch.rand(n, device="cuda"), torch.rand(n, device="cuda")
    out = torch.empty_like(a)
    vecadd[(triton.cdiv(n, 256),)](a, b, out, n, 256)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, a+b)

    torch.manual_seed(20261121)
    B, H, L, D = 1, 2, 12, 8
    q, k, v = [torch.randn(B,H,L,D,device="cuda",dtype=torch.float32,requires_grad=True)
               for _ in range(3)]
    spans = [[(1,4),(7,9)]]
    kw = dict(thres_mode="medium",mean_mode="image")
    dense = dense_cfpo(q,k,v,spans,**kw)
    tiled = streaming_cfpo(q,k,v,spans,tile_k=4,**kw)
    custom = flashcf_custom(q,k,v,spans,tile_k=4,**kw)
    direction = torch.randn_like(dense)
    gd = torch.autograd.grad((dense*direction).sum(),(q,k,v),retain_graph=True)
    gt = torch.autograd.grad((tiled*direction).sum(),(q,k,v),retain_graph=True)
    gc = torch.autograd.grad((custom*direction).sum(),(q,k,v))
    oerr = max((dense-x).detach().abs().max().item() for x in (tiled,custom))
    gerr = max((a-b).detach().abs().max().item() for arr in (gt,gc) for a,b in zip(gd,arr))
    if oerr>1e-4 or gerr>2e-4:
        raise AssertionError(f"FlashCF CUDA reference mismatch: output={oerr}, gradient={gerr}")
    device = torch.cuda.get_device_properties(0)
    return {
        "status": "PASS", "device": device.name, "total_memory_gib": round(device.total_memory/2**30,2),
        "torch_version": torch.__version__, "torch_cuda": torch.version.cuda,
        "triton_version": triton.__version__, "gpu_compute": list(torch.cuda.get_device_capability()),
        "triton_vecadd": "PASS", "reference_dense_streaming_custom": "PASS",
        "reference_output_max_abs": oerr, "reference_gradient_max_abs": gerr,
        "note": "synthetic inputs only; not official CFPO fixture or GPU speedup benchmark"
    }


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=None)
    args=parser.parse_args()
    report=run_probe()
    print(json.dumps(report,indent=2,ensure_ascii=False))
    if args.json:
        args.json.parent.mkdir(parents=True,exist_ok=True)
        args.json.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")

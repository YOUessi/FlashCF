"""CUDA training-step benchmark: dense BF16 vs rowwise V2 vs BF16 tile+V2.

Qwen-like Hq=16,Hkv=2,D=128 with 4D additive causal attention masks and
CMVE image span [L//8,L//2). Synthetic inputs, frozen timing conditions.
"""
import argparse,json,torch,os,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reference.native_bf16_gqa import dense_native_cmve
from kernels.triton_native_backward_v2 import flashcf_native_v2
from kernels.triton_native_tile import flashcf_tiled
from experiments.benchmark_native_bf16 import measure


def inspect(L,reps=7,row_tile=16,key_tile=64):
    torch.manual_seed(20261121+L)
    B,Hq,Hkv,D=1,16,2,128
    xs=[torch.randn((B,h,L,D),device="cuda",dtype=torch.bfloat16,requires_grad=True)
        for h in (Hq,Hkv,Hkv)]
    mask=torch.zeros((B,1,L,L),device="cuda",dtype=torch.bfloat16)
    mask.masked_fill_(torch.ones(L,L,device="cuda",dtype=torch.bool).triu(1)[None,None],
                      torch.finfo(mask.dtype).min)
    spans=[[(L//8,L//2)]]
    opt=dict(thres_mode="high",mean_mode="image")
    fn={
      "dense":lambda q,k,v:dense_native_cmve(q,k,v,spans,mask,**opt),
      "rowwise":lambda q,k,v:flashcf_native_v2(q,k,v,spans,mask,block_k=32,**opt),
      "tiled":lambda q,k,v:flashcf_tiled(q,k,v,spans,mask,block_k=32,
                                        row_tile=row_tile,key_tile=key_tile,**opt),
    }
    weight=torch.randn_like(xs[0])
    ref=fn["dense"](*xs)
    grad_ref=torch.autograd.grad((ref*weight).sum(),xs)
    res=dict(length=L,q_heads=Hq,kv_heads=Hkv,head_dim=D,
             dtype="BF16",num_intervals=1,row_tile=row_tile,key_tile=key_tile)
    for name,method in fn.items():
        y=method(*xs)
        grad=torch.autograd.grad((y*weight).sum(),xs)
        fout=(y.float()-ref.float())
        grad_errors=[]
        for actual,expected in zip(grad,grad_ref):
            e=(actual-expected).float()
            grad_errors.append((torch.linalg.vector_norm(e)/
                                torch.linalg.vector_norm(expected.float()).clamp(min=1)).item())
        latency,memory=measure(method,xs,weight,reps)
        res[name]={"step_median_ms":latency,"extra_peak_mib":memory,
                   "output_rel_l2":(torch.linalg.vector_norm(fout)/
                                    torch.linalg.vector_norm(ref.float()).clamp(min=1)).item(),
                   "output_max_abs":fout.abs().max().item(),
                   "qkv_grad_relative":grad_errors}
    print(json.dumps(res),flush=True)
    return res


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--lengths",type=int,nargs="+",default=[97,256,512,1024])
    p.add_argument("--repeat",type=int,default=7)
    p.add_argument("--row-tile",type=int,default=16)
    p.add_argument("--key-tile",type=int,default=64)
    p.add_argument("--output",type=Path,
                   default=Path.home()/".cache"/"flashcf_results"/"p2_native_tile_bench.json")
    args=p.parse_args()
    data=[inspect(L,args.repeat,args.row_tile,args.key_tile) for L in args.lengths]
    payload={"git_commit":os.environ.get("FLASHCF_SOURCE_COMMIT","unknown"),
             "machine":torch.cuda.get_device_name(0),"torch":torch.__version__,
             "caveats":["synthetic BF16, one image span",
                        "dense PyTorch reference not optimized CFPO + FlashAttention",
                        "median of small number of repeats; not full training"],"results":data}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2)+"\n")
    print("SUMMARY",json.dumps([{ "L":a["length"],
        "rowwise_ms":a["rowwise"]["step_median_ms"],
        "tiled_ms":a["tiled"]["step_median_ms"],
        "tiled_over_rowwise":round(a["rowwise"]["step_median_ms"]/
                                     a["tiled"]["step_median_ms"],3)}
                                     for a in data]),flush=True)

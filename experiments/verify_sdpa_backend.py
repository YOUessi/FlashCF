"""Inspect PyTorch BF16 SDPA factual-attention backend on Tang RTX4090.

The original CFPO training path separately invokes FlashAttention for factual
output. Our native SDPA+CMVE comparison is only a surrogate; this script
checks whether its factual path is actually dispatched to CUDA Flash
Attention instead of PyTorch's math backend.

Outputs sanitized timing/backend summary without model activations.
"""
import json
import torch
import torch.nn.functional as F
from torch.profiler import profile,ProfilerActivity


def run(L):
    B,Hq,Hkv,D=1,16,2,128
    torch.manual_seed(118+L)
    q,k,v=[torch.randn((B,h,L,D),device="cuda",dtype=torch.bfloat16,
                       requires_grad=True) for h in (Hq,Hkv,Hkv)]
    kr=k.repeat_interleave(Hq//Hkv,dim=1)
    vr=v.repeat_interleave(Hq//Hkv,dim=1)
    mask=torch.zeros((B,1,L,L),device="cuda",dtype=torch.bfloat16)
    mask.masked_fill_(torch.ones(L,L,device="cuda",dtype=torch.bool).triu(1)[None,None],
                      torch.finfo(mask.dtype).min)
    def body():
        y=F.scaled_dot_product_attention(q,kr,vr,attn_mask=mask,dropout_p=0.,
                                         is_causal=False)
        g=torch.autograd.grad(y.float().square().mean(),(q,k,v))
        return y,g
    for _ in range(3):body()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as p:
        body()
        torch.cuda.synchronize()
    names=[(e.key,float(e.self_device_time_total)) for e in p.key_averages()
           if any(x in e.key.lower() for x in
                  ("flash_attention","scaled_dot_product","efficient_attention","attention_math"))]
    names=sorted(names,key=lambda x:-x[1])
    return dict(length=L,gpu=torch.cuda.get_device_name(0),
                torch=torch.__version__,
                attention_kernels=names,
                flash_detected=any("flash_attention" in x[0].lower() for x in names))


if __name__=="__main__":
    summary=[run(L) for L in (256,1024)]
    print(json.dumps(summary,indent=2),flush=True)

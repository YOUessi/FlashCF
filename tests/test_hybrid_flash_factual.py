"""Check diagnostic hybrid follows a split Flash factual + CMVE baseline."""
import unittest
import torch
from reference.native_sdpa_cmve import native_sdpa_cmve
from kernels.hybrid_flash_factual import hybrid_flash_factual


@unittest.skipUnless(torch.cuda.is_available(),"CUDA required")
class TestHybridFactual(unittest.TestCase):
    def test_bf16_causal(self):
        for seed,L in ((901,24),(902,97)):
            torch.manual_seed(seed)
            B,Hq,Hkv,D=1,16,2,128
            q,k,v=[torch.randn((B,h,L,D),dtype=torch.bfloat16,
                                device="cuda",requires_grad=True)
                   for h in (Hq,Hkv,Hkv)]
            mask=torch.zeros((B,1,L,L),device="cuda",dtype=torch.bfloat16)
            mask.masked_fill_(torch.ones(L,L,device="cuda",dtype=torch.bool).triu(1)[None,None],
                              torch.finfo(torch.bfloat16).min)
            spans=[[(2,L//2)]]
            opts=dict(thres_mode="medium",mean_mode="image")
            ref=native_sdpa_cmve(q,k,v,spans,mask,factual_backend="flash_causal",**opts)
            out=hybrid_flash_factual(q,k,v,spans,mask,**opts)
            dy=torch.randn_like(q)
            rg=torch.autograd.grad((ref*dy).sum(),(q,k,v))
            og=torch.autograd.grad((out*dy).sum(),(q,k,v))
            re=(torch.linalg.vector_norm((out-ref).float())/
                torch.linalg.vector_norm(ref.float()).clamp(min=1)).item()
            gr=[(torch.linalg.vector_norm((a-b).float())/
                 torch.linalg.vector_norm(b.float()).clamp(min=1)).item()
                for a,b in zip(og,rg)]
            self.assertLess(re,0.018,(L,re))
            self.assertLess(max(gr),0.05,(L,gr))
            print("HYBRID",L,"rel_output",re,"rel_grad",gr,flush=True)


if __name__=="__main__":
    unittest.main()

"""Validate BF16 SDPA+explicit-CMVE baseline vs full dense BF16 reference."""
import unittest
import torch
from reference.native_bf16_gqa import dense_native_cmve
from reference.native_sdpa_cmve import native_sdpa_cmve


class TestNativeSDPACMVE(unittest.TestCase):
    def test_bf16_gqa(self):
        if not torch.cuda.is_available():self.skipTest("CUDA required")
        for seed,L,Hq,Hkv,D in ((401,24,4,2,16),(402,97,16,2,128)):
            torch.manual_seed(seed)
            q,k,v=[torch.randn((1,h,L,D),dtype=torch.bfloat16,device="cuda",
                                requires_grad=True) for h in (Hq,Hkv,Hkv)]
            mask=torch.zeros((1,1,L,L),device="cuda",dtype=torch.bfloat16)
            mask.masked_fill_(torch.ones(L,L,device="cuda",dtype=torch.bool).triu(1)[None,None],
                              torch.finfo(torch.bfloat16).min)
            spans=[[(2,L//2)]]
            opts=dict(thres_mode="medium",mean_mode="image")
            d=dense_native_cmve(q,k,v,spans,mask,**opts)
            a=native_sdpa_cmve(q,k,v,spans,mask,**opts)
            w=torch.randn_like(d)
            gd=torch.autograd.grad((d*w).sum(),(q,k,v))
            ga=torch.autograd.grad((a*w).sum(),(q,k,v))
            rel=(torch.linalg.vector_norm((a-d).float())/
                 torch.linalg.vector_norm(d.float()).clamp(min=1)).item()
            gerr=max((torch.linalg.vector_norm((x-y).float())/
                      torch.linalg.vector_norm(y.float()).clamp(min=1)).item()
                     for x,y in zip(ga,gd))
            self.assertLess(rel,0.01,(L,"output",rel))
            self.assertLess(gerr,0.03,(L,"gradient",gerr))
            print("SDPA_CMVE_VERIFIED",L,"output_relative",rel,"grad_relative",gerr,flush=True)


if __name__=="__main__":
    unittest.main()

"""Direct correction-only CMVE tests: forward, gradients, grouped K/V masks."""
import unittest
import torch
import torch.nn.functional as F
from reference.native_sdpa_cmve import native_sdpa_cmve
from reference.native_bf16_gqa import dense_native_cmve
from kernels.triton_cmve_delta import flashcf_cmve_delta,flashcf_factual_plus_delta


@unittest.skipUnless(torch.cuda.is_available(),"Requires Tang CUDA")
class TestDirectCorrection(unittest.TestCase):
    def run_case(self,seed,L,D,Hq,Hkv,spans,mode,mean,backend):
        torch.manual_seed(seed)
        q,k,v=[torch.randn((1,h,L,D),device="cuda",dtype=torch.bfloat16,
                            requires_grad=True) for h in (Hq,Hkv,Hkv)]
        mask=torch.zeros((1,1,L,L),device="cuda",dtype=torch.bfloat16)
        mask.masked_fill_(torch.ones((L,L),device="cuda",dtype=torch.bool).triu(1)[None,None],
                          torch.finfo(torch.bfloat16).min)
        if backend=="auto":
            # Prefix-only keys are valid for masked padding case: every
            # query retains at least one allowed key.
            mask[:,:,:,-4:]=torch.finfo(torch.bfloat16).min
        kw={"thres_mode":mode,"mean_mode":mean}
        reference=native_sdpa_cmve(q,k,v,spans,mask,factual_backend=backend,**kw)
        actual=flashcf_factual_plus_delta(q,k,v,spans,mask,
                                           factual_backend=backend,**kw)
        # Also ensure correction does not depend on running a second
        # factual attention path.
        correction=flashcf_cmve_delta(q,k,v,spans,mask,**kw)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue(torch.isfinite(correction).all())
        weight=torch.randn_like(q)
        refg=torch.autograd.grad((reference*weight).sum(),(q,k,v))
        testg=torch.autograd.grad((actual*weight).sum(),(q,k,v))
        def rel(a,b):
            return (torch.linalg.vector_norm((a-b).float())/
                    torch.linalg.vector_norm(b.float()).clamp(min=1)).item()
        outputrel=rel(actual,reference)
        grads=[rel(a,b) for a,b in zip(testg,refg)]
        self.assertLess(outputrel,0.02,(L,"output",outputrel))
        self.assertLess(max(grads),0.04,(L,"grads",grads))
        print("DIRECT_CORRECTION",L,D,Hq,Hkv,backend,
              "output_rel",outputrel,"grads",grads,flush=True)

    def test_qwen_short_causal(self):
        self.run_case(101,97,128,16,2,[[(15,79)]],"high","image","flash_causal")

    def test_two_intervals(self):
        self.run_case(102,48,32,4,2,[[(2,9),(20,31)]],"medium","text","flash_causal")

    def test_padding_four_d(self):
        self.run_case(103,40,32,4,1,[[(2,15)]],"low","image","auto")

    def test_empty_intervention(self):
        self.run_case(104,32,32,4,2,[[]],"high","image","flash_causal")


if __name__=="__main__":
    unittest.main()

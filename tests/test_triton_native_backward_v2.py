"""Compare native BF16 backward v2 with eager dense autograd and v1."""
import unittest
import torch

from reference.native_bf16_gqa import dense_native_cmve
try:
    from kernels.triton_native_bf16 import flashcf_native
    from kernels.triton_native_backward_v2 import flashcf_native_v2
    CUDA=torch.cuda.is_available()
except ImportError:
    CUDA=False


@unittest.skipUnless(CUDA,"CUDA required")
class BetterBF16GradTests(unittest.TestCase):
    def _check(self,seed,shape,kv_heads,mask_heads,mode,mean_mode):
        B,Hq,L,D=shape
        torch.manual_seed(seed)
        q=torch.randn(shape,device="cuda",dtype=torch.bfloat16,requires_grad=True)
        k=torch.randn((B,kv_heads,L,D),device="cuda",dtype=torch.bfloat16,requires_grad=True)
        v=torch.randn((B,kv_heads,L,D),device="cuda",dtype=torch.bfloat16,requires_grad=True)
        inputs=(q,k,v)
        mask=torch.zeros((B,mask_heads,L,L),device="cuda",dtype=torch.bfloat16)
        causal=torch.ones((L,L),device="cuda",dtype=torch.bool).triu(1)
        mask.masked_fill_(causal[None,None],torch.finfo(mask.dtype).min)
        if mask_heads==Hq:
            mask[:,:,3:9,1:3] += 0.25
        spans=[[(2,L//2)] for _ in range(B)]
        args=dict(thres_mode=mode,mean_mode=mean_mode)
        y=dense_native_cmve(*inputs,spans,mask,**args)
        w=torch.randn_like(y)
        ref=torch.autograd.grad((y*w).sum(),inputs)
        diffs={}
        for name,fn in (("v1",flashcf_native),("v2",flashcf_native_v2)):
            t=fn(*inputs,spans,mask,block_k=16,**args)
            gd=torch.autograd.grad((t*w).sum(),inputs)
            relative=[(torch.linalg.vector_norm((a-b).float())/
                       torch.linalg.vector_norm(b.float()).clamp(min=1)).item()
                      for a,b in zip(gd,ref)]
            self.assertTrue(torch.isfinite(t).all())
            self.assertLess(max(relative),0.06,(name,relative,shape))
            diffs[name]=relative
        print(f"GRAD_BF16_V1_V2 {shape=} {mode=} {diffs=}",flush=True)

    def test_bf16_four_heads(self):
        self._check(1,(1,4,24,16),2,1,"medium","image")

    def test_bf16_gqa_multiple_batch(self):
        self._check(2,(2,4,32,32),1,4,"high","text")

    def test_bf16_real_head_dimension(self):
        self._check(3,(1,16,48,128),2,1,"high","image")


if __name__=="__main__":
    unittest.main()

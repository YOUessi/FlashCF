"""Native BF16, GQA and 4D additive-mask forward/backward correctness on Tang."""
import unittest

import torch

from reference.native_bf16_gqa import dense_native_cmve

try:
    from kernels.triton_native_bf16 import flashcf_native
    CUDA=torch.cuda.is_available()
except ImportError:
    CUDA=False


def make_mask(B,H,L,mode,dtype,device):
    """3 supported cases: causal, causal+padding, per-head packed segments."""
    mask=torch.zeros((B,H,L,L),device=device,dtype=dtype)
    invalid=torch.ones((L,L),device=device,dtype=torch.bool).triu(1)
    mask.masked_fill_(invalid[None,None],torch.finfo(dtype).min)
    if mode=="padding":
        # Keys 23..end are padded; every query still has valid earliest key.
        mask[:,:,:,23:]=torch.finfo(dtype).min
    elif mode=="packing":
        mid=L//2
        mask[:,:,:mid,mid:]=torch.finfo(dtype).min
        mask[:,:,mid:,:mid]=torch.finfo(dtype).min
    elif mode=="bias":
        # Finite head-specific additive bias: must be added not discarded.
        mask[:,:,:,1:5]+=0.25
    return mask


@unittest.skipUnless(CUDA,"Requires CUDA/Triton")
class BF16NativeTests(unittest.TestCase):
    def _check(self,B,Hq,Hkv,L,D,spans,mask_heads,mask_mode,mode,mean,dtype,seed=20261121):
        torch.manual_seed(seed)
        q,k,v=[
            torch.randn((B,h,L,D),device="cuda",dtype=dtype,requires_grad=True)
            for h in (Hq,Hkv,Hkv)
        ]
        mask=make_mask(B,mask_heads,L,mask_mode,dtype,q.device)
        opts={"thres_mode":mode,"mean_mode":mean}
        expected=dense_native_cmve(q,k,v,spans,mask,**opts)
        actual=flashcf_native(q,k,v,spans,mask,block_k=16,**opts)
        self.assertEqual(actual.dtype,dtype)
        self.assertEqual(actual.shape,expected.shape)
        weight=torch.randn_like(expected)
        ga=torch.autograd.grad((actual*weight).sum(),(q,k,v))
        ge=torch.autograd.grad((expected*weight).sum(),(q,k,v))
        abs_error=(actual.float()-expected.float()).abs()
        relative_error=float(torch.linalg.vector_norm(actual.float()-expected.float())/
                             torch.linalg.vector_norm(expected.float()).clamp(min=1))
        grad_max=max((x.float()-y.float()).abs().max().item() for x,y in zip(ga,ge))
        fmax=abs_error.max().item()
        if dtype==torch.float32:
            self.assertLess(fmax,1e-3,(L,mode,mask_mode,fmax))
            self.assertLess(grad_max,2e-3,(L,mode,mask_mode,grad_max))
        else:
            # Strict BF16 gate is meaningful relative-L2, separately log the
            # maximum difference and compare with the actual original CFPO.
            self.assertLess(relative_error,0.045,(L,mode,mask_mode,fmax,relative_error))
            self.assertLess(grad_max,0.4,(L,mode,mask_mode,grad_max))
        print(f" BF16_NATIVE_TEST dtype={dtype} B={B} Hq={Hq} Hkv={Hkv} "
              f"L={L} D={D} 4d_heads={mask_heads} {mask_mode=} {mode=} "
              f"{fmax=:.6g} {relative_error=:.6g} {grad_max=:.6g}",
              flush=True)

    def test_bf16_gqa_causal(self):
        self._check(1,4,2,24,16,[[(3,11)]],1,"causal","medium","image",torch.bfloat16)

    def test_bf16_gqa_multi_spans_text(self):
        self._check(1,4,2,32,32,[[(2,10),(17,24)]],1,"causal","high","text",torch.bfloat16,4)

    def test_bf16_gqa_padding(self):
        self._check(1,4,1,32,16,[[(2,12)]],1,"padding","low","image",torch.bfloat16,5)

    def test_bf16_packed_per_head_4d(self):
        self._check(1,4,2,32,16,[[(2,8),(17,24)]],4,"packing","medium","text",torch.bfloat16,6)

    def test_fp32_gqa_finite_bias(self):
        self._check(2,4,2,24,32,[[(2,10)],[(3,12)]],4,"bias","medium","image",torch.float32,7)

    def test_bf16_empty_visual(self):
        self._check(1,2,1,16,16,[[]],1,"causal","high","image",torch.bfloat16,8)


if __name__=="__main__":
    unittest.main()

"""Validate BF16 Tensor Core K/V backward against dense autograd and V2.

The new key-tiled reductions can change floating point summation order.
Check relative gradient error rather than requiring bitwise BF16 equality.
"""
import unittest
import torch
from reference.native_bf16_gqa import dense_native_cmve
from kernels.triton_native_gemm_kv import flashcf_gemm_kv


@unittest.skipUnless(torch.cuda.is_available(),"requires Tang CUDA")
class TestGemmKV(unittest.TestCase):
    def _check(self,seed,L,D,Hq,Hkv,mb,bk):
        torch.manual_seed(seed)
        B=1
        q=torch.randn((B,Hq,L,D),device="cuda",dtype=torch.bfloat16,requires_grad=True)
        k=torch.randn((B,Hkv,L,D),device="cuda",dtype=torch.bfloat16,requires_grad=True)
        v=torch.randn((B,Hkv,L,D),device="cuda",dtype=torch.bfloat16,requires_grad=True)
        mask=torch.zeros((B,1,L,L),device="cuda",dtype=torch.bfloat16)
        mask.masked_fill_(torch.ones((L,L),device="cuda",dtype=torch.bool).triu(1)[None,None],
                          torch.finfo(torch.bfloat16).min)
        spans=[[(2,L//2)]]
        params={"thres_mode":"medium","mean_mode":"image"}
        y0=dense_native_cmve(q,k,v,spans,mask,**params)
        y=flashcf_gemm_kv(q,k,v,spans,mask,block_k=16,
                           row_tile=16,key_tile=32,
                           grad_query_tile=mb,grad_key_tile=bk,**params)
        g=torch.randn_like(q)
        gg0=torch.autograd.grad((y0*g).sum(),(q,k,v))
        gg=torch.autograd.grad((y*g).sum(),(q,k,v))
        rel_y=(torch.linalg.vector_norm((y-y0).float())/
               torch.linalg.vector_norm(y0.float()).clamp(min=1)).item()
        rel_g=[(torch.linalg.vector_norm((x-z).float())/
                torch.linalg.vector_norm(z.float()).clamp(min=1)).item() for x,z in zip(gg,gg0)]
        self.assertLess(rel_y,0.008,(L,D,rel_y))
        self.assertLess(max(rel_g),0.03,(L,D,rel_g))
        print(f"GEMM_KV_OK L={L} D={D} Hq={Hq} Hkv={Hkv} GM={mb} GN={bk} "
              f"fwd_rel={rel_y:.5f} qkv_grad_rel={rel_g}",flush=True)

    def test_short_gqa_16_16(self):
        self._check(301,24,16,4,2,16,16)

    def test_medium_gqa_16_32(self):
        self._check(302,40,32,4,1,16,32)

    def test_qwen_gqa_16_32(self):
        self._check(303,97,128,16,2,16,32)

    def test_query32_key16(self):
        self._check(304,48,64,4,2,32,16)


if __name__=="__main__":unittest.main()

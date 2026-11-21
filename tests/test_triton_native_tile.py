"""Tensor Core BF16 forward and V2 backward parity.

Small deterministic tests against independent Dense BF16 CMVE and the existing
validated row-wise native V2; full 4D additive masks, native GQA and two spans.
"""
import unittest
import torch
from reference.native_bf16_gqa import dense_native_cmve
try:
    from kernels.triton_native_tile import flashcf_tiled, tiled_factual
    from kernels.triton_native_backward_v2 import flashcf_native_v2
    CUDA=torch.cuda.is_available()
except ImportError:
    CUDA=False


@unittest.skipUnless(CUDA,"requires Tang CUDA")
class TensorCoreTiledTests(unittest.TestCase):
    def _case(self,seed,shape,hkv,spans,heads,mode,row_tile,key_tile):
        torch.manual_seed(seed)
        B,Hq,L,D=shape
        q=torch.randn((B,Hq,L,D),device="cuda",dtype=torch.bfloat16,requires_grad=True)
        k=torch.randn((B,hkv,L,D),device="cuda",dtype=torch.bfloat16,requires_grad=True)
        v=torch.randn((B,hkv,L,D),device="cuda",dtype=torch.bfloat16,requires_grad=True)
        mask=torch.zeros((B,heads,L,L),device="cuda",dtype=torch.bfloat16)
        mask.masked_fill_(torch.ones(L,L,device="cuda",dtype=torch.bool).triu(1)[None,None],
                          torch.finfo(mask.dtype).min)
        if heads==Hq:
            mask[:,:,4:11,1:3]+=0.125
        opt=dict(thres_mode=mode,mean_mode="image")
        expected=dense_native_cmve(q,k,v,spans,mask,**opt)
        tiled=flashcf_tiled(q,k,v,spans,mask,row_tile=row_tile,key_tile=key_tile,
                            block_k=16,**opt)
        rowwise=flashcf_native_v2(q,k,v,spans,mask,block_k=16,**opt)
        diff=(tiled.float()-expected.float())
        rms=diff.square().mean().sqrt().item()
        relative=(torch.linalg.vector_norm(diff)/
                  torch.linalg.vector_norm(expected.float()).clamp(min=1)).item()
        direction=torch.randn_like(q)
        ref=torch.autograd.grad((expected*direction).sum(),(q,k,v),retain_graph=True)
        gt=torch.autograd.grad((tiled*direction).sum(),(q,k,v),retain_graph=True)
        self.assertLess(relative,0.008,(shape,seed,"relative",relative))
        for a,b in zip(gt,ref):
            rel=(torch.linalg.vector_norm((a-b).float())/
                 torch.linalg.vector_norm(b.float()).clamp(min=1)).item()
            self.assertLess(rel,0.025,(shape,seed,"grad_relative",rel))
        self.assertTrue(torch.isfinite(tiled).all())
        print(f"TILE_TEST shape={shape} rt={row_tile} kt={key_tile} "
              f"diff={float(diff.abs().max())} rmse={rms} rel_l2={relative} "
              f"row_rel={float(torch.linalg.vector_norm(rowwise.float()-expected.float())/torch.linalg.vector_norm(expected.float()).clamp(min=1))}",flush=True)

    def test_gqa_bf16_16_by_64(self):
        self._case(41,(1,4,40,32),2,[[(2,12)]],1,"high",16,64)

    def test_gqa_bf16_32_by_32(self):
        self._case(42,(1,4,50,32),1,[[(2,12),(20,34)]],1,"medium",32,32)

    def test_qwen_shape_16heads(self):
        self._case(43,(1,16,97,128),2,[[(15,79)]],1,"high",16,64)

    def test_perhead_mask(self):
        self._case(44,(2,4,31,64),2,[[(3,12)],[(5,15)]],4,"low",16,32)


if __name__=="__main__":
    unittest.main()

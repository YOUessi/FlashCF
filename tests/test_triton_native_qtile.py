"""Correctness and gradient tests for fully query/key-tiled BF16 CMVE."""
import unittest
import torch
from reference.native_bf16_gqa import dense_native_cmve
from kernels.triton_native_qtile import flashcf_gemm_all


@unittest.skipUnless(torch.cuda.is_available(),"CUDA/Triton required")
class TestQueryTile(unittest.TestCase):
    def run_case(self,seed,L,D,Hq,Hkv,mode,spans,q_tile,k_tile,head_mask=False):
        torch.manual_seed(seed)
        B=1
        q=torch.randn((B,Hq,L,D),device="cuda",dtype=torch.bfloat16,requires_grad=True)
        k=torch.randn((B,Hkv,L,D),device="cuda",dtype=torch.bfloat16,requires_grad=True)
        v=torch.randn((B,Hkv,L,D),device="cuda",dtype=torch.bfloat16,requires_grad=True)
        mh=Hq if head_mask else 1
        mask=torch.zeros((B,mh,L,L),device="cuda",dtype=torch.bfloat16)
        mask.masked_fill_(torch.ones(L,L,device="cuda",dtype=torch.bool).triu(1)[None,None],
                          torch.finfo(torch.bfloat16).min)
        if head_mask:mask[:,:,2:8,1:3]+=0.125
        options={"thres_mode":mode,"mean_mode":"image"}
        target=dense_native_cmve(q,k,v,spans,mask,**options)
        actual=flashcf_gemm_all(q,k,v,spans,mask,row_tile=16,key_tile=32,
                                grad_q_tile=q_tile,grad_k_tile=k_tile,block_k=16,**options)
        grad=torch.randn_like(target)
        gt=torch.autograd.grad((target*grad).sum(),(q,k,v))
        ga=torch.autograd.grad((actual*grad).sum(),(q,k,v))
        def relative(a,b):
            return (torch.linalg.vector_norm((a-b).float())/
                    torch.linalg.vector_norm(b.float()).clamp(min=1)).item()
        er=relative(actual,target)
        gr=[relative(a,b) for a,b in zip(ga,gt)]
        self.assertLess(er,0.012,(L,D,er))
        self.assertLess(max(gr),0.035,(L,D,gr))
        print(f"QTILE_CMP L={L} D={D} Hq={Hq} Hkv={Hkv} "
              f"QT={q_tile} KT={k_tile} rel={er} grad_rel={gr}",flush=True)

    def test_qtile_one_span(self):
        self.run_case(801,24,16,4,2,"medium",[[(2,11)]],16,32)

    def test_qtile_two_spans(self):
        self.run_case(802,48,32,4,1,"high",[[(2,10),(18,29)]],16,64)

    def test_qtile_real_qwen_shape(self):
        self.run_case(803,97,128,16,2,"high",[[(15,79)]],16,64)

    def test_qtile_perhead_bias(self):
        self.run_case(804,32,64,4,2,"low",[[(3,18)]],32,32,True)


if __name__=="__main__":
    unittest.main()

"""Separate-SDPA baseline should match dense reference semantics."""
import unittest
import torch
from reference.cfpo_reference import dense_cfpo
from reference.cfpo_sdpa_baseline import sdpa_cmve_baseline


class SDPAReferenceTests(unittest.TestCase):
    def test_outputs_and_gradients(self):
        for seed,mean in ((5,"text"),(6,"image")):
            torch.manual_seed(seed)
            q,k,v=[torch.randn(1,2,32,16,requires_grad=True) for _ in range(3)]
            spans=[[(2,11),(17,24)]]
            args=dict(thres_mode="medium",mean_mode=mean)
            a=dense_cfpo(q,k,v,spans,**args)
            b=sdpa_cmve_baseline(q,k,v,spans,**args)
            torch.testing.assert_close(a,b,rtol=2e-4,atol=3e-5)
            w=torch.randn_like(a)
            ag=torch.autograd.grad((a*w).sum(),(q,k,v),retain_graph=True)
            bg=torch.autograd.grad((b*w).sum(),(q,k,v))
            for x,y in zip(ag,bg):
                torch.testing.assert_close(x,y,rtol=4e-4,atol=3e-5)

if __name__=="__main__":unittest.main()

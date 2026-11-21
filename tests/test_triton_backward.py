"""GPU backward parity: Triton vs dense PyTorch CFPO surrogate.

Tests deterministic causal FP32 only, with hard threshold masks.
"""
import unittest
import torch

try:
    from reference.cfpo_reference import dense_cfpo
    from kernels.triton_backward import flashcf_autograd
    CUDA=torch.cuda.is_available()
except ImportError:
    CUDA=False


@unittest.skipUnless(CUDA,"requires CUDA+Triton")
class TritonBackwardTests(unittest.TestCase):
    def _check(self,seed,shape,spans,mode,mean,block):
        torch.manual_seed(seed)
        data=[torch.randn(shape,device="cuda",dtype=torch.float32,
                          requires_grad=True) for _ in range(3)]
        q,k,v=data
        weight=torch.randn(shape,device="cuda",dtype=torch.float32)
        expected=dense_cfpo(q,k,v,spans,thres_mode=mode,mean_mode=mean)
        eg=torch.autograd.grad((expected*weight).sum(),data)
        actual=flashcf_autograd(q,k,v,spans,thres_mode=mode,
                               mean_mode=mean,block_k=block)
        ag=torch.autograd.grad((actual*weight).sum(),data)
        ferr=(expected-actual).abs().max().item()
        gerr=max((x-y).abs().max().item() for x,y in zip(eg,ag))
        self.assertLess(ferr,2e-4,f"forward difference {ferr}")
        self.assertLess(gerr,5e-4,f"gradient difference {gerr}")

    def test_one_span_high(self):
        self._check(32,(1,1,17,16),[[(2,10)]],"high","image",16)

    def test_two_spans_text(self):
        self._check(33,(1,2,25,16),[[(1,7),(10,16)]],"medium","text",16)

    def test_batch_two_low(self):
        self._check(34,(2,2,32,32),[[(1,9)],[(3,12),(17,22)]],"low","image",32)

    def test_no_spans(self):
        self._check(35,(1,1,16,8),[[]],"extra","text",16)

if __name__=="__main__":
    unittest.main()

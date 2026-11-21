"""Race-free gather backward should match dense reference and atomics."""
import unittest
import torch

try:
    from reference.cfpo_reference import dense_cfpo
    from kernels.triton_backward_gather import flashcf_gather
    from kernels.triton_backward import flashcf_autograd
    CUDA=torch.cuda.is_available()
except ImportError:
    CUDA=False


@unittest.skipUnless(CUDA,"requires CUDA")
class GatherBackwardTests(unittest.TestCase):
    def check(self,seed,shape,spans,mode,mean):
        torch.manual_seed(seed)
        xs=[torch.randn(shape,device="cuda",dtype=torch.float32,requires_grad=True)
            for _ in range(3)]
        w=torch.randn(shape,device="cuda")
        opts={"thres_mode":mode,"mean_mode":mean}
        ref=dense_cfpo(*xs,spans,**opts)
        grad_ref=torch.autograd.grad((ref*w).sum(),xs)
        for name,fn in (("gather",flashcf_gather),("atomic",flashcf_autograd)):
            y=fn(*xs,spans,block_k=16,**opts)
            grads=torch.autograd.grad((y*w).sum(),xs)
            ferr=(y-ref).abs().max().item()
            gerr=max((a-b).abs().max().item() for a,b in zip(grads,grad_ref))
            self.assertLess(ferr,3e-4,f"{name} forward err={ferr}")
            self.assertLess(gerr,5e-4,f"{name} grad err={gerr}")

    def test_single_span(self):
        self.check(1,(1,2,32,32),[[(3,16)]],"high","image")

    def test_multiple_spans_text(self):
        self.check(2,(1,2,35,16),[[(2,11),(17,24)]],"medium","text")

    def test_multiple_batch(self):
        self.check(3,(2,2,32,32),[[(1,9)],[(3,12),(18,22)]],"low","image")

    def test_empty(self):
        self.check(4,(1,1,17,8),[[]],"extra","text")

if __name__=="__main__":unittest.main()

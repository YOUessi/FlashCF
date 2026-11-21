"""GPU correctness tests for actual Triton kernels (skip without CUDA)."""
import unittest
import torch

try:
    from kernels.triton_forward import flashcf_forward
    from reference.cfpo_reference import dense_cfpo
    ENABLED = torch.cuda.is_available()
except ImportError:
    ENABLED = False


@unittest.skipUnless(ENABLED, "CUDA + Triton are required")
class TritonForwardTests(unittest.TestCase):
    def _run(self, shape, spans, mode, mean_mode, block=32, seed=20261121):
        torch.manual_seed(seed)
        q,k,v=[torch.randn(shape,device="cuda",dtype=torch.float32).contiguous()
               for _ in range(3)]
        expected=dense_cfpo(q,k,v,spans,thres_mode=mode,mean_mode=mean_mode)
        actual,meta=flashcf_forward(q,k,v,spans,thres_mode=mode,mean_mode=mean_mode,
                                    block_k=block,return_diagnostics=True)
        max_error=(expected-actual).abs().max().item()
        self.assertLess(max_error, 1e-4,
                        f"shape={shape} spans={spans} mode={mode} mean={mean_mode}, "
                        f"max_err={max_error:.4g}")
        self.assertFalse(meta["attention_materialized"])
        self.assertEqual(meta["num_spans"],sum(map(len,spans)))
        return max_error

    def test_single_span(self):
        self._run((1,2,32,16),[[(1,11)]],"high","image",16)

    def test_two_spans_text_mean(self):
        self._run((1,2,48,32),[[(2,10),(16,29)]],"medium","text",32)

    def test_batch_two(self):
        self._run((2,2,64,32),[[(1,12)],[(9,20),(28,38)]],"low","image",32)

    def test_empty_span(self):
        self._run((1,1,23,8),[[]],"extra","text",16)

    def test_tail_span(self):
        self._run((1,1,24,16),[[(14,24)]],"high","image",16)

if __name__=="__main__":
    unittest.main()

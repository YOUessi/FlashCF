"""Deterministic CPU regression tests; no model download or CUDA required."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import torch
from cfpo_reference import dense_cfpo, streaming_cfpo
from custom_backward_reference import flashcf_custom


def check_case(seed, spans, mode, mean, packed=False):
    torch.manual_seed(seed)
    L, D, H = 12, 8, 2
    q, k, v = [
        torch.randn(1, H, L, D, dtype=torch.float64, requires_grad=True)
        for _ in range(3)
    ]
    sample_ids = (torch.arange(L) >= L // 2).long()[None, :] if packed else None
    kwargs = dict(thres_mode=mode, mean_mode=mean, sample_ids=sample_ids)
    ref = dense_cfpo(q, k, v, spans, **kwargs)
    tiled = streaming_cfpo(q, k, v, spans, tile_k=4, **kwargs)
    custom = flashcf_custom(q, k, v, spans, tile_k=4, **kwargs)
    direction = torch.randn_like(ref)
    expected = torch.autograd.grad((ref * direction).sum(), (q,k,v), retain_graph=True)
    tiled_grads = torch.autograd.grad((tiled * direction).sum(), (q,k,v), retain_graph=True)
    custom_grads = torch.autograd.grad((custom * direction).sum(), (q,k,v))
    for out in (tiled, custom):
        torch.testing.assert_close(ref, out, rtol=1e-9, atol=1e-9)
    for actual in (tiled_grads, custom_grads):
        for a, b in zip(expected, actual):
            torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-9)


class ReferenceTests(unittest.TestCase):
    def test_image_high(self):
        check_case(1, [[(1,4)]], "high", "image")

    def test_text_medium_multiple_spans(self):
        check_case(2, [[(1,4),(7,9)]], "medium", "text")

    def test_extra_packed(self):
        check_case(3, [[(1,4),(7,9)]], "extra", "image", True)

    def test_empty_memory_spans(self):
        check_case(4, [[]], "low", "text")


if __name__ == "__main__":
    unittest.main()

"""Capture *actual* already-rotated Q/K/V inputs of CFPO's CMVE attention.

Usage in the official CFPO Python process (before model forward):

    from capture_official_qkv import CFPOCapture
    import verl.models.transformers.qwen2_vl_cmve as cfpo_module
    capture = CFPOCapture(cfpo_module, './flashcf_fixtures', max_examples=2)
    capture.install()
    # Run one ordinary CFPO forward / short training step with apply_cmve=True.
    capture.uninstall()

This is a diagnostic capture, NOT a training patch or speed optimization.
Only use with trusted model/data. Q/K/V contain sensitive representations.
Works for the reference official module in Raven-July/CFPO; it expects the
symbols `values_noise_multiply` and `flash_attention_forward` to be called
from the `apply_cmve=True` branch. Does not intercept unrelated attention paths.

The wrapper never changes the returned value or gradients of either function.
Captured tensors are detached. Saving CUDA tensors to CPU can stall training:
use for a few test fixtures only. Do not publish captured model activations.
"""
from __future__ import annotations

import copy
import functools
import json
import os
from pathlib import Path
import threading
from typing import Any

import torch


class CFPOCapture:
    def __init__(self, module: Any, output_dir: str | Path, *, max_examples: int = 2,
                 max_sequence_length: int = 2048):
        self.module = module
        # DDP/Ray workers must not overwrite fixtures from other ranks.
        rank = os.getenv('RANK') or os.getenv('LOCAL_RANK') or '0'
        self.output_dir = Path(output_dir) / f'rank{rank}' 
        self.max_examples = max_examples
        self.max_sequence_length = max_sequence_length
        self.saved = 0
        self.skipped = 0
        self._original_values = None
        self._original_flash = None
        self._local = threading.local()
        self._lock = threading.Lock()

    def install(self):
        if self._original_values is not None:
            raise RuntimeError('CFPOCapture already installed')
        for name in ('values_noise_multiply', 'flash_attention_forward'):
            if not callable(getattr(self.module, name, None)):
                raise RuntimeError(f'Official CFPO module missing {name}')
        self._original_values = self.module.values_noise_multiply
        self._original_flash = self.module.flash_attention_forward

        @functools.wraps(self._original_values)
        def wrap_values(attn_weights, image_pos, value_states, thres_mode, noise, mean_mode):
            result = self._original_values(attn_weights, image_pos, value_states,
                                           thres_mode, noise, mean_mode)
            # Keep only metadata, NEVER the [B,H,L,L] attention matrix.
            self._local.pending = {
                'image_pos': copy.deepcopy(image_pos),
                'thres_mode': thres_mode,
                'noise': bool(noise),
                'mean_mode': mean_mode,
            }
            return result

        @functools.wraps(self._original_flash)
        def wrap_flash(*args, **kwargs):
            # Preserve original behavior and backward graph first.
            result = self._original_flash(*args, **kwargs)
            pending = getattr(self._local, 'pending', None)
            self._local.pending = None
            if pending is None:
                return result
            # In the pinned CFPO implementation, self, Q, K, V are 0..3.
            if len(args) < 4:
                self.skipped += 1
                return result
            q, k, v = args[1:4]
            if not (isinstance(q, torch.Tensor) and isinstance(k, torch.Tensor)
                    and isinstance(v, torch.Tensor)):
                self.skipped += 1
                return result
            if q.ndim != 4 or q.shape[-2] > self.max_sequence_length:
                self.skipped += 1
                return result
            with self._lock:
                if self.saved >= self.max_examples:
                    return result
                idx = self.saved
                self.saved += 1
            attention_mask = args[4] if len(args) >= 5 else kwargs.get('attention_mask')
            captured = {
                'schema_version': 1,
                'q': q.detach().cpu().contiguous(),
                'k': k.detach().cpu().contiguous(),
                'v': v.detach().cpu().contiguous(),
                'flash_output': result[0].detach().cpu() if (isinstance(result, (tuple,list)) and len(result)>0 and isinstance(result[0], torch.Tensor)) else None,
                'attention_mask': attention_mask.detach().cpu() if isinstance(attention_mask, torch.Tensor) else None,
                'image_pos': pending['image_pos'],
                'thres_mode': pending['thres_mode'],
                'noise': pending['noise'],
                'mean_mode': pending['mean_mode'],
                'sliding_window': kwargs.get('sliding_window'),
                'dropout': kwargs.get('dropout', 0.0),
                'position_ids': kwargs['position_ids'].detach().cpu() if isinstance(kwargs.get('position_ids'), torch.Tensor) else None,
            }
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / f'cfpo_cmve_{idx:03d}.pt'
            torch.save(captured, path)
            summary = {
                'tensor_file': path.name,
                'q_shape': list(q.shape), 'k_shape': list(k.shape), 'v_shape': list(v.shape),
                'q_dtype': str(q.dtype),
                'mask_shape': list(attention_mask.shape) if isinstance(attention_mask, torch.Tensor) else None,
                'image_pos': pending['image_pos'],
                'thres_mode': pending['thres_mode'],
                'noise': pending['noise'],
                'mean_mode': pending['mean_mode'],
                'dropout': captured['dropout'],
                'sliding_window': captured['sliding_window'],
            }
            (self.output_dir / f'cfpo_cmve_{idx:03d}.json').write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
            return result

        self.module.values_noise_multiply = wrap_values
        self.module.flash_attention_forward = wrap_flash
        return self

    def uninstall(self):
        if self._original_values is None:
            return
        self.module.values_noise_multiply = self._original_values
        self.module.flash_attention_forward = self._original_flash
        self._original_values = None
        self._original_flash = None
        if hasattr(self._local, 'pending'):
            del self._local.pending

    def __enter__(self):
        return self.install()

    def __exit__(self, exc_type, exc_value, traceback):
        self.uninstall()


if __name__ == '__main__':
    print('Import CFPOCapture and install it before the official CFPO forward. See module docstring.')
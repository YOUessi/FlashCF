"""Load only CFPO's three unmodified CMVE helper functions from pinned upstream source.

The original CFPO qwen2_vl_cmve.py imports legacy Transformers and verl, so
loading the whole module into Tang's newer Qwen2.5 environment is fragile.
Instead, obtain the exact Git-pinned source (Apache-2.0), verify its Git blob
SHA-1, AST-extract ONLY add_diffusion_noise, GMM_mask and values_noise_multiply,
then execute those original function bodies with torch in the module namespace.

No source modification, no whole CFPO RL stack, and no implicit network call
when --source is supplied. Restricted to trusted upstream code.
"""
from __future__ import annotations

import ast
import hashlib
import math
from pathlib import Path
from urllib.request import urlopen

import torch

UPSTREAM_COMMIT="98ca5a4123be2e85e94855fead46c9a385611136"
UPSTREAM_BLOB_SHA="2b9e77ea947f3b07801c9e6a863b774ac2521a5a"
UPSTREAM_URL=(f"https://raw.githubusercontent.com/Raven-July/CFPO/{UPSTREAM_COMMIT}/"
              "verl/models/transformers/qwen2_vl_cmve.py")
ALLOWED={"add_diffusion_noise","GMM_mask","values_noise_multiply"}


def load_official_cmve(*, source:Path|None=None):
    if source:
        raw=Path(source).read_bytes()
    else:
        with urlopen(UPSTREAM_URL,timeout=25) as resp:
            raw=resp.read()
    blob=b"blob "+str(len(raw)).encode()+b"\x00"+raw
    actual=hashlib.sha1(blob).hexdigest()
    if actual!=UPSTREAM_BLOB_SHA:
        raise RuntimeError(f"Pinned CFPO source mismatch: Git blob {actual} expected {UPSTREAM_BLOB_SHA}")
    parsed=ast.parse(raw.decode("utf-8"),filename="cfpo_qwen2_vl_cmve.py")
    selected=[n for n in parsed.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))
              and n.name in ALLOWED]
    found={x.name for x in selected}
    if found!=ALLOWED:raise RuntimeError(f"CFPO helper functions not found: {ALLOWED-found}")
    module=ast.Module(body=selected,type_ignores=[])
    ast.fix_missing_locations(module)
    namespace={"torch":torch,"math":math}
    exec(compile(module,UPSTREAM_URL,"exec"),namespace)
    return namespace


def official_dense_cfpo(q,k,v,mask,spans,*,thres_mode="high",mean_mode="image",
                        source:Path|None=None):
    """CFPO official CMVE mask/Value functions with a dense mathematical normal.

    Native Qwen eager factual output (before o_proj) should be compared
    separately. Supports no dropout/noise/sliding-window and full prefill.
    """
    mod=load_official_cmve(source=source)
    if mask is None or mask.ndim!=4:raise ValueError("Require actual 4D additive mask")
    B,H,L,D=q.shape
    assert q.shape==k.shape==v.shape
    s=(q@k.transpose(-1,-2))/math.sqrt(D)
    s=s+mask.to(s.dtype)
    if q.dtype==torch.float16:
        s=torch.where(torch.isinf(s),torch.zeros_like(s),s)
    p=torch.softmax(s,dim=-1,dtype=torch.float32).to(q.dtype)
    image_pos=[[{"image_token_start":st,"image_token_end":en} for st,en in group]
               for group in spans]
    selected,replace=mod["values_noise_multiply"](
        p,image_pos,v,thres_mode,False,mean_mode)
    delta=(p*selected)@(replace-v)
    factual=p@v
    return factual+delta, factual, selected, replace

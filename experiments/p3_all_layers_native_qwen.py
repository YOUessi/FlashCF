"""Frozen full-model all-language-layer CMVE experiment on pretrained Qwen2.5-VL-3B.

Both modes apply CMVE at EVERY language Attention layer, rather than a single
selected layer. Compare:
- baseline = explicit native BF16 Flash factual + materialized CMVE
- direct = same factual Flash + bounded-memory Triton direct CMVE delta

This is NOT original legacy CFPO/verl training, and does not update weights.
Model-level deviations from BF16 hard gates and downstream layers must be
reported, never assumed negligible from single-layer parity.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as model_native

from experiments.p1_full_model_layer_backward import make_input
from experiments.p0_capture_native_qwen import DEFAULT_MODEL,REV
from reference.native_sdpa_cmve import native_sdpa_cmve
from kernels.triton_cmve_delta import flashcf_factual_plus_delta


def run_all(model, batch, spans, kind):
    native=model_native.eager_attention_forward
    counts={}
    def patched(module,q,k,v,attention_mask=None,**kwargs):
        idx=getattr(module,"layer_idx",None)
        if idx is None:
            return native(module,q,k,v,attention_mask=attention_mask,**kwargs)
        q,k,v=(x.contiguous() for x in (q,k,v))
        attention_mask=attention_mask.contiguous()
        if kind=="dense_flash_cmve":
            x=native_sdpa_cmve(q,k,v,spans,attention_mask,
                               thres_mode="high",mean_mode="image",
                               factual_backend="flash_causal")
        elif kind=="direct":
            x=flashcf_factual_plus_delta(q,k,v,spans,attention_mask,
                                         thres_mode="high",mean_mode="image",
                                         factual_backend="flash_causal")
        else:raise ValueError(kind)
        counts[int(idx)]=counts.get(int(idx),0)+1
        return x.transpose(1,2).contiguous(),None
    model_native.eager_attention_forward=patched
    try:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before=torch.cuda.Event(enable_timing=True)
        after=torch.cuda.Event(enable_timing=True)
        before.record()
        with torch.inference_mode():
            logits=model(**batch,use_cache=False,logits_to_keep=1).logits.float()
        after.record()
        after.synchronize()
        result=(logits.detach(),dict(
            latency_ms=round(float(before.elapsed_time(after)),4),
            peak_allocated_gib=round(torch.cuda.max_memory_allocated()/2**30,4),
            layers_ran=len(counts),
            layer_indices=sorted(counts),
            call_counts=counts))
        return result
    finally:
        model_native.eager_attention_forward=native


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model-dir",type=Path,default=DEFAULT_MODEL)
    p.add_argument("--output",type=Path,
                   default=Path.home()/".cache"/"flashcf_results"/"p3_all_layers_forward_local.json")
    args=p.parse_args()
    torch.manual_seed(20261121)
    proc=AutoProcessor.from_pretrained(args.model_dir,use_fast=False,trust_remote_code=False)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir,dtype=torch.bfloat16,attn_implementation="eager",
        trust_remote_code=False).eval().to("cuda")
    model.requires_grad_(False)
    batch,spans=make_input(proc,model.config)
    outputs={}
    with torch.inference_mode():
        for kind in ("dense_flash_cmve","direct"):
            logits,measure=run_all(model,batch,spans,kind)
            outputs[kind]=logits
            print("METHOD",kind,json.dumps(measure),flush=True)
            if kind=="dense_flash_cmve":baseline=measure
            else:candidate=measure
    log0=outputs["dense_flash_cmve"]
    log1=outputs["direct"]
    diff=log1-log0
    row=dict(
        model_revision=REV,
        flashcf_sha=os.getenv("FLASHCF_SOURCE_COMMIT","immutable GitHub commit"),
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        num_layers_expected=model.config.text_config.num_hidden_layers,
        input_tokens=int(batch["input_ids"].shape[-1]),
        image_spans=spans,
        baseline=baseline,candidate=candidate,
        next_token_baseline=int(log0.flatten().argmax()),
        next_token_candidate=int(log1.flatten().argmax()),
        top1_same=bool(log0.flatten().argmax()==log1.flatten().argmax()),
        logits_max_abs=float(diff.abs().max()),
        logits_rel_l2=float(torch.linalg.vector_norm(diff)/
                            torch.linalg.vector_norm(log0).clamp(min=1)),
        logit_rmse=float(diff.square().mean().sqrt()),
        training=False,
        model_weights_frozen=True,
        note="Native pretrained all-layer patched CMVE forward, not original full CFPO RL pipeline")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(row,indent=2,ensure_ascii=False)+"\n")
    print("ALL_LAYER_COMPARE",json.dumps(row,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()

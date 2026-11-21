"""Full pretrained Qwen2.5-VL-3B, all 36 CMVE layers, backward to Q/K/V.

Freeze pretrained model weights and run the same all-layer CMVE policy
under two operator backends: forced Flash+explicit CMVE baseline and
forced Flash+direct Triton delta. Compare logits and gradients at selected
layers 0, 8, and 35 from one deterministic final-logit scalar loss.

This is NOT full 3B parameter training or official verl RL. It checks
gradient propagation through all 36 counterfactual Attention layers.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import torch
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as model_native

from experiments.p3_all_layers_native_qwen import make_input
from experiments.p0_capture_native_qwen import DEFAULT_MODEL,REV
from reference.native_sdpa_cmve import native_sdpa_cmve
from kernels.triton_cmve_delta import flashcf_factual_plus_delta


def run_backprop(model,batch,spans,mode,targets):
    native=model_native.eager_attention_forward
    saved={}
    called=[]
    def patched(module,q,k,v,attention_mask=None,**kwargs):
        idx=getattr(module,"layer_idx",None)
        if idx is None:
            return native(module,q,k,v,attention_mask=attention_mask,**kwargs)
        idx=int(idx)
        q,k,v=(x.contiguous() for x in (q,k,v))
        if idx==min(targets):
            # Start gradient graph here, not in vision encoder or earlier
            # layers; no model weights require gradients.
            q=q.detach().requires_grad_(True)
            k=k.detach().requires_grad_(True)
            v=v.detach().requires_grad_(True)
        if idx in targets:
            saved[idx]=(q,k,v)
        called.append(idx)
        if mode=="flash_cmve":
            result=native_sdpa_cmve(q,k,v,spans,attention_mask,
                                    thres_mode="high",mean_mode="image",
                                    factual_backend="flash_causal")
        elif mode=="direct":
            result=flashcf_factual_plus_delta(q,k,v,spans,attention_mask,
                                              thres_mode="high",mean_mode="image",
                                              factual_backend="flash_causal")
        else:
            raise ValueError(mode)
        return result.transpose(1,2).contiguous(),None
    model_native.eager_attention_forward=patched
    try:
        torch.cuda.reset_peak_memory_stats()
        start=torch.cuda.Event(enable_timing=True)
        stop=torch.cuda.Event(enable_timing=True)
        start.record()
        logits=model(**batch,use_cache=False,logits_to_keep=1).logits.float()
        loss=logits.square().mean()
        if len(called)!=model.config.text_config.num_hidden_layers:
            raise RuntimeError(f"Expected all language layers, got {len(called)}")
        if set(saved)!=set(targets):
            raise RuntimeError(f"Did not capture gradient targets {targets}: {saved.keys()}")
        flat=[x for idx in targets for x in saved[idx]]
        grads=torch.autograd.grad(loss,flat,allow_unused=False)
        stop.record()
        stop.synchronize()
        result={
            "logits":logits.detach().cpu(),
            "loss":float(loss.detach()),
            "gradients":{idx:tuple(g.detach().float().cpu() for g in grads[n*3:n*3+3])
                         for n,idx in enumerate(targets)},
            "peak_gpu_gib":round(torch.cuda.max_memory_allocated()/2**30,4),
            "full_step_ms_single":round(float(start.elapsed_time(stop)),4),
            "num_counterfactual_layers":len(called),
        }
        return result
    finally:
        model_native.eager_attention_forward=native


def rel(x,y):
    a=x.float()-y.float()
    return float(torch.linalg.vector_norm(a)/torch.linalg.vector_norm(y.float()).clamp(min=1))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model-dir",type=Path,default=DEFAULT_MODEL)
    p.add_argument("--targets",type=int,nargs="+",default=[0,8,35])
    p.add_argument("--output",type=Path,
                   default=Path.home()/".cache"/"flashcf_results"/"p3_all_layers_backward_local.json")
    args=p.parse_args()
    torch.manual_seed(20261121)
    proc=AutoProcessor.from_pretrained(args.model_dir,trust_remote_code=False,use_fast=False)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir,dtype=torch.bfloat16,attn_implementation="eager",
        trust_remote_code=False).eval().to("cuda")
    model.requires_grad_(False)
    batch,spans=make_input(proc,model.config)
    target=sorted(set(args.targets))
    reference=run_backprop(model,batch,spans,"flash_cmve",target)
    candidate=run_backprop(model,batch,spans,"direct",target)
    result={
        "source_commit":os.environ.get("FLASHCF_SOURCE_COMMIT","immutable source"),
        "model":"Qwen/Qwen2.5-VL-3B-Instruct",
        "model_revision":REV,"num_layers":model.config.text_config.num_hidden_layers,
        "model_weights_frozen":True,"full_parameter_training":False,
        "BF16":True,"num_input_tokens":int(batch["input_ids"].shape[-1]),
        "target_grad_layers":target,
        "flash_baseline_peak_gpu_gib":reference["peak_gpu_gib"],
        "direct_peak_gpu_gib":candidate["peak_gpu_gib"],
        "flash_baseline_step_ms":reference["full_step_ms_single"],
        "direct_step_ms":candidate["full_step_ms_single"],
        "logit_rel_l2":rel(candidate["logits"],reference["logits"]),
        "logit_max_abs":float((candidate["logits"]-reference["logits"]).abs().max()),
        "scalar_loss_reference":reference["loss"],
        "scalar_loss_direct":candidate["loss"],
        "grad_rel_l2":{},
    }
    for i in target:
        gs=candidate["gradients"][i]
        g0=reference["gradients"][i]
        result["grad_rel_l2"][str(i)]={n:rel(a,b) for n,a,b in zip(("q","k","v"),gs,g0)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print("P3_ALL_LAYERS_BACKWARD",json.dumps(result,indent=2),flush=True)


if __name__=="__main__":main()

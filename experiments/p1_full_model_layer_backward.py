"""True pretrained Qwen2.5-VL model-level BF16 native FlashCF integration test.

This executes a complete pretrained Qwen2.5-VL-3B forward through a native
Transformers 4.57.6 Attention interface. At ONE selected layer, compare:
- unmodified CFPO-style dense BF16 CMVE operator (verified against pinned
  original source helpers);
- FlashCF native BF16/GQA/4D-mask Triton V2 operator.

Freeze model weights, detach Q/K/V only at the selected layer, and run
backward from the final model logits to those input Q/K/V leaves. This
tests gradient propagation through subsequent transformer layers WITHOUT
attempting full-model fine-tuning or 3B parameter-gradient allocation.

NO confidential tensors are committed to GitHub. Use a synthetic image only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import torch
from PIL import Image,ImageDraw
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as model_native

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reference.native_bf16_gqa import dense_native_cmve
from reference.native_sdpa_cmve import native_sdpa_cmve
from kernels.native import flashcf_native
from kernels.triton_native_qtile import flashcf_gemm_all
from kernels.hybrid_flash_factual import hybrid_flash_factual
from experiments.p0_capture_native_qwen import DEFAULT_MODEL,spans_from_ids,REV


def make_input(processor,config):
    image=Image.new("RGB",(224,224),(245,248,253))
    draw=ImageDraw.Draw(image)
    for box,color in [((10,20,62,80),"red"),
                      ((85,62,145,133),"green"),
                      ((169,148,205,207),"blue")]:
        draw.rectangle(box,fill=color)
    prompt=processor.apply_chat_template(
        [{"role":"user","content":[{"type":"image","image":image},
         {"type":"text","text":"How many colored rectangles do you see?"}]}],
        tokenize=False,add_generation_prompt=True)
    x=processor(text=[prompt],images=[image],return_tensors="pt",padding=True)
    spans=spans_from_ids(x["input_ids"],config)
    assert spans and spans[0]
    return {k:v.to("cuda") for k,v in x.items()},spans


def forward_with_patch(model,x,spans,layer,variant):
    original=model_native.eager_attention_forward
    captures=[]
    def wrapper(module,q,k,v,attention_mask=None,**kwargs):
        if int(getattr(module,"layer_idx",-1))!=layer:
            return original(module,q,k,v,attention_mask=attention_mask,**kwargs)
        if captures:
            raise RuntimeError("Target layer unexpectedly called twice")
        # Source model frozen; require gradients only at the selected
        # attention projected Q/K/V to avoid allocating full 3B-model grads.
        q=q.detach().contiguous().requires_grad_(True)
        k=k.detach().contiguous().requires_grad_(True)
        v=v.detach().contiguous().requires_grad_(True)
        captures.append((q,k,v))
        if variant=="dense":
            result=dense_native_cmve(q,k,v,spans,attention_mask,thres_mode="high",mean_mode="image")
        elif variant=="triton":
            result=flashcf_native(q,k,v,spans,attention_mask,thres_mode="high",mean_mode="image")
        elif variant=="flash_cmve":
            result=native_sdpa_cmve(q,k,v,spans,attention_mask,
                                    thres_mode="high",mean_mode="image",
                                    factual_backend="flash_causal")
        elif variant=="gemmall":
            result=flashcf_gemm_all(q,k,v,spans,attention_mask,thres_mode="high",
                                    mean_mode="image",row_tile=16,key_tile=64,
                                    grad_q_tile=16,grad_k_tile=64)
        elif variant=="hybrid":
            result=hybrid_flash_factual(q,k,v,spans,attention_mask,
                                       thres_mode="high",mean_mode="image")
        else:raise RuntimeError("Bad variant")
        return result.transpose(1,2).contiguous(),None

    model_native.eager_attention_forward=wrapper
    try:
        logits=model(**x,use_cache=False,logits_to_keep=1).logits.float()
        # Last-token output has shape [B,1,vocab], use a smooth deterministic loss.
        loss=logits.square().mean()
        grads=torch.autograd.grad(loss,captures[0])
        return logits.detach(),grads,float(loss.detach())
    finally:
        model_native.eager_attention_forward=original


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model-dir",type=Path,default=DEFAULT_MODEL)
    p.add_argument("--layer",type=int,default=8)
    p.add_argument("--candidate",choices=["triton","gemmall","hybrid"],default="triton")
    p.add_argument("--baseline",choices=["dense","flash_cmve"],default="dense")
    p.add_argument("--output",type=Path,
                   default=Path.home()/".cache"/"flashcf_results"/"p1_full_model_layer_v2_local.json")
    args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError("Requires Tang CUDA")
    torch.manual_seed(20261121)
    processor=AutoProcessor.from_pretrained(args.model_dir,use_fast=False,trust_remote_code=False)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir,dtype=torch.bfloat16,attn_implementation="eager",
        trust_remote_code=False).eval().to("cuda")
    model.requires_grad_(False)
    x,spans=make_input(processor,model.config)
    torch.cuda.reset_peak_memory_stats()
    outputs={}
    for mode in (args.baseline,args.candidate):
        logits,grads,loss=forward_with_patch(model,x,spans,args.layer,mode)
        outputs[mode]=(logits,[g.detach().float() for g in grads])
        print("PASS",mode,"loss",loss,"gpu_peak_gib",
              torch.cuda.max_memory_allocated()/2**30,flush=True)
    d,gd=outputs[args.baseline]
    t,gt=outputs[args.candidate]
    err=t-d
    results={
        "layer":args.layer,"candidate":args.candidate,"baseline":args.baseline,"model_sha":REV,"image_spans":spans,
        "sequence_length":int(x["input_ids"].shape[1]),
        "model_weights_frozen":True,
        "training_full_model":False,
        "native_gqa":True,
        "input_dtype":"BF16",
        "logits_max_abs":float(err.abs().max()),
        "logits_rel_l2":float(torch.linalg.vector_norm(err)/
                              torch.linalg.vector_norm(d).clamp(min=1)),
        "model_final_logits_shape":list(t.shape),
        "selected_layer_gradients":{},
        "peak_gpu_allocated_gib":round(torch.cuda.max_memory_allocated()/2**30,3),
    }
    for name,a,b in zip(("q","k_native","v_native"),gt,gd):
        delta=a-b
        results["selected_layer_gradients"][name]={
            "max_abs":float(delta.abs().max()),
            "rel_l2":float(torch.linalg.vector_norm(delta)/
                           torch.linalg.vector_norm(b).clamp(min=1)),
        }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps(results,indent=2),flush=True)


if __name__=="__main__":
    main()

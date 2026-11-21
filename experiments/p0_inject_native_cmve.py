"""P0: inject original CFPO CMVE operator into a real pretrained Qwen2.5-VL layer.

This runs TWO deterministic inference passes with identical pretrained weights:
native eager and native eager + original pinned CFPO value intervention in
one selected layer. This is a *native-Qwen compatibility smoke test*,
not the original verl/CFPO multi-layer RL model or a training result.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as modeling

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.p0_capture_native_qwen import DEFAULT_MODEL, REV, spans_from_ids
from tools.p0_official_cmve_oracle import load_official_cmve, UPSTREAM_COMMIT

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model-dir",type=Path,default=DEFAULT_MODEL)
    p.add_argument("--layer",type=int,default=8)
    p.add_argument("--official-source",type=Path,default=None)
    p.add_argument("--mode",choices=["low","medium","high","extra"],default="medium")
    p.add_argument("--output",type=Path,default=Path.home()/".cache"/"flashcf_results"/"p0_injected_cmve_local.json")
    args=p.parse_args()
    torch.manual_seed(20261121)
    cfpo=load_official_cmve(source=args.official_source)["values_noise_multiply"]
    processor=AutoProcessor.from_pretrained(args.model_dir,trust_remote_code=False,use_fast=False)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir,dtype=torch.bfloat16,attn_implementation="eager",
        trust_remote_code=False).eval().to("cuda")
    im=Image.new("RGB",(224,224),(245,248,253))
    draw=ImageDraw.Draw(im)
    for box,color in [((10,20,62,80),"red"),((85,62,145,133),"green"),
                      ((169,148,205,207),"blue")]:
        draw.rectangle(box,fill=color)
    messages=[{"role":"user","content":[{"type":"image","image":im},
              {"type":"text","text":"How many colored rectangles appear in this image? Answer briefly."}]}]
    prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    batch=processor(text=[prompt],images=[im],return_tensors="pt",padding=True)
    batch={k:v.to("cuda") if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
    spans=spans_from_ids(batch["input_ids"].cpu(),model.config)
    image_pos=[[{"image_token_start":s,"image_token_end":e} for s,e in ss]
               for ss in spans]
    orig=modeling.eager_attention_forward
    observed=[]
    with torch.inference_mode():
        baseline=model(**batch,use_cache=False,logits_to_keep=1).logits.clone()
        def patch(module,q,k,v,attention_mask,**kw):
            out,weights=orig(module,q,k,v,attention_mask,**kw)
            if int(getattr(module,"layer_idx",-1))!=args.layer:
                return out,weights
            ratio=q.shape[1]//k.shape[1]
            vrep=modeling.repeat_kv(v,ratio)
            cmask,vprior=cfpo(weights,image_pos,vrep,args.mode,False,"image")
            delta=(weights*cmask)@(vprior-vrep)
            observed.append({
                "selected":int((cmask>0).sum()),
                "total":cmask.numel(),
                "delta_max_abs":float(delta.float().abs().max()),
                "delta_mean_abs":float(delta.float().abs().mean()),
            })
            return out+delta.transpose(1,2).contiguous(),weights
        modeling.eager_attention_forward=patch
        try:
            altered=model(**batch,use_cache=False,logits_to_keep=1).logits.clone()
        finally:
            modeling.eager_attention_forward=orig
    if len(observed)!=1:
        raise RuntimeError(f"CMVE branch expected once, actually {len(observed)}")
    diff=(baseline.float()-altered.float()).abs()
    result={
        "status":"ORIGINAL_CMVE_HELPER_IN_NATIVE_QWEN_LAYER_EXECUTED",
        "model":"Qwen/Qwen2.5-VL-3B-Instruct",
        "model_sha":REV,
        "cfpo_source_sha":UPSTREAM_COMMIT,
        "layer":args.layer,"threshold_mode":args.mode,
        "attention_shape":list(batch["input_ids"].shape),
        "image_spans":spans,
        "injected_attention":observed[0],
        "logit_max_abs_difference":float(diff.max()),
        "logit_mean_abs_difference":float(diff.mean()),
        "native_attention_only":True,
        "not_original_multi_layer_cfpo_training":True,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__":main()

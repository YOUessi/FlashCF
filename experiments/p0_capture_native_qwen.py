"""Capture real pretrained Qwen2.5-VL Q/K/V from the native Transformers 4.57 eager path.

P0 artifact is local/private. This captures post-M-RoPE Q/K and V, but does
NOT claim to run the official CFPO monkey-patched model forward. The captured
states can be evaluated separately against the pinned official CMVE operator.
"""
import argparse
import json
import os
import subprocess
from pathlib import Path
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as mod

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_MODEL=Path.home()/".cache"/"flashcf_models"/"Qwen2.5-VL-3B-Instruct"
REV="66285546d2b821cf421d4f5eb2576359d3770cd3"


def spans_from_ids(input_ids,config):
    result=[]
    for row in input_ids.tolist():
        opens=[]
        intervals=[]
        for j,tok in enumerate(row):
            if tok==config.vision_start_token_id: opens.append(j)
            if tok==config.vision_end_token_id and opens:
                start=opens.pop(0)+1
                if start<j:intervals.append((start,j))
        result.append(intervals)
    return result


def capture_one(model,inputs,layers,spans,out_dir):
    records={}
    orig=mod.eager_attention_forward
    def wrapper(module,q,k,v,attention_mask,**kw):
        result=orig(module,q,k,v,attention_mask,**kw)
        layer=int(getattr(module,"layer_idx",-1))
        if layer in layers and layer not in records:
            assert q.shape[1]%k.shape[1]==0
            repeat=q.shape[1]//k.shape[1]
            rec=dict(layer=layer,q=q.detach().cpu().contiguous(),
                     k=mod.repeat_kv(k,repeat).detach().cpu().contiguous(),
                     v=mod.repeat_kv(v,repeat).detach().cpu().contiguous(),
                     q_native=list(q.shape),kv_native=list(k.shape),gqa_repeat=repeat,
                     native_output=result[0].detach().cpu(),
                     additive_mask=None if attention_mask is None else attention_mask.detach().cpu(),
                     image_spans=spans,
                     q_scaling=float(module.scaling),
                     model_sha=REV,backend="transformers_eager",
                     source="post_rotary_pre_attention")
            records[layer]=rec
            print("CAPTURED",layer,"Q",list(q.shape),"K",list(k.shape),"mask",
                  None if attention_mask is None else list(attention_mask.shape),flush=True)
        return result
    mod.eager_attention_forward=wrapper
    try:
        with torch.inference_mode():
            outputs=model(**inputs,use_cache=False,logits_to_keep=1)
    finally:
        mod.eager_attention_forward=orig
    out_dir.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for layer,item in sorted(records.items()):
        path=out_dir/f"native_qwen_layer_{layer:02}.pt"
        torch.save(item,path)
        summaries.append(dict(layer=layer,file=path.name,q_shape=list(item["q"].shape),
                     k_shape=list(item["k"].shape),v_shape=list(item["v"].shape),
                     dtype=str(item["q"].dtype),
                     attention_mask_shape=None if item["additive_mask"] is None else list(item["additive_mask"].shape),
                     gqa_repeat=item["gqa_repeat"],scaling=item["q_scaling"]))
    if set(records)!=set(layers):
        raise RuntimeError(f"Missing layers requested={layers}; actual={list(records)}")
    return summaries,list(outputs.logits.shape)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model-dir",type=Path,default=DEFAULT_MODEL)
    p.add_argument("--fixture-dir",type=Path,default=ROOT/"flashcf_fixtures"/"native_qwen")
    p.add_argument("--summary",type=Path,default=ROOT/"experiments"/"results"/"p0_capture_local.json")
    p.add_argument("--layers",type=int,nargs="+",default=[0,8])
    args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError("Requires NVIDIA CUDA on Tang")
    if not (args.model_dir/"model.safetensors.index.json").is_file():
        raise FileNotFoundError(f"Download first: {args.model_dir}")
    print("LOADING",str(args.model_dir),flush=True)
    processor=AutoProcessor.from_pretrained(args.model_dir,trust_remote_code=False,use_fast=False)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir,dtype=torch.bfloat16,attn_implementation="eager",
        trust_remote_code=False).eval().to("cuda")
    torch.cuda.reset_peak_memory_stats()
    im=Image.new("RGB",(224,224),(246,248,252))
    dr=ImageDraw.Draw(im)
    for box,color in [((10,20,62,80),"red"),((85,62,145,133),"green"),((169,148,205,207),"blue")]:
        dr.rectangle(box,fill=color)
    messages=[{"role":"user","content":[{"type":"image","image":im},
                 {"type":"text","text":"How many colored rectangles appear in this image? Answer briefly."}]}]
    prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    inputs=processor(text=[prompt],images=[im],padding=True,return_tensors="pt")
    inputs={k:v.to("cuda") if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
    spans=spans_from_ids(inputs["input_ids"].cpu(),model.config)
    if not spans or not spans[0]:raise RuntimeError("No vision spans in actual tokenized input")
    print("INPUT","tokens",inputs["input_ids"].shape[-1],"spans",spans,flush=True)
    saved,logits_shape=capture_one(model,inputs,set(args.layers),spans,args.fixture_dir)
    meta=dict(status="REAL_QWEN_PRETRAINED_QKV_CAPTURED",
             model="Qwen/Qwen2.5-VL-3B-Instruct",model_revision=REV,
             flashcf_sha=os.environ.get("FLASHCF_SOURCE_COMMIT") or
                 subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
             torch=torch.__version__,transformers=__import__("transformers").__version__,
             gpu=torch.cuda.get_device_name(0),
             peak_allocated_gib=round(torch.cuda.max_memory_allocated()/2**30,3),
             sequence_length=int(inputs["input_ids"].shape[-1]),image_spans=spans,
             native_logits_shape=logits_shape,layers=saved,
             distinction="Native pretrained Qwen2.5-VL eager path, NOT a patched CFPO forward")
    args.summary.parent.mkdir(parents=True,exist_ok=True)
    args.summary.write_text(json.dumps(meta,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(meta,indent=2,ensure_ascii=False),flush=True)


if __name__=="__main__":main()

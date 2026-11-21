"""Resumable SHA256-verified ModelScope range downloader for pinned Qwen2.5-VL-3B.

Tang's direct huggingface.co connection is blocked, and hf-mirror's large-shard
transfer may stall. ModelScope supplies HTTP Range responses with X-Linked-Etag,
which we verified matches the SHA256 of the exact pinned HF revision.

Writes ONLY to ~/.cache/flashcf_models/Qwen2.5-VL-3B-Instruct and its .parts
subdirectory. It never deletes unrelated files, changes conda/system packages,
or commits weight files to GitHub.

Run: python3 experiments/p0_modelscope_ranges.py --workers 8 --part-mib 64
"""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path
from urllib.request import Request,urlopen

ROOT=Path.home()/".cache"/"flashcf_models"/"Qwen2.5-VL-3B-Instruct"
PREFIX="https://www.modelscope.cn/models/Qwen/Qwen2.5-VL-3B-Instruct/resolve/master"
REV="66285546d2b821cf421d4f5eb2576359d3770cd3"

# Exact SHA256 from Qwen/HF pinned revision LFS metadata, verified independently
# against ModelScope X-Linked-Etag for BOTH weight shards on 2026-11-21.
FILES={
    "model-00001-of-00002.safetensors":(3982649232,
      "41a8895c164b4d32bae6b302f4603fcbc1797f32dafa45c7e9bcda23c6755df8"),
    "model-00002-of-00002.safetensors":(3526688744,
      "365531ff8752420e89dee707b79d021fb2d6e25abafe486f080555a4fe6972e4"),
}


def probe(name,size,sha):
    url=PREFIX+"/"+name
    req=Request(url,headers={"Range":"bytes=0-0","User-Agent":"FlashCF-research/0.1"})
    with urlopen(req,timeout=20) as r:
        cr=r.headers.get("Content-Range","")
        match=re.fullmatch(r"bytes 0-0/(\d+)",cr)
        etag=r.headers.get("X-Linked-ETag","").strip('"')
        if r.status!=206 or match is None or int(match.group(1))!=size:
            raise RuntimeError(f"Bad ModelScope range response for {name}: {r.status} / {cr}")
        if etag.lower()!=sha:
            raise RuntimeError(f"ModelScope hash differs from pinned HF LFS: {name} {etag} != {sha}")
        r.read()
    print(f"VERIFIED_REMOTE {name} {size/2**30:.3f} GiB sha256={sha[:12]}",flush=True)


def download_segment(url,dest,start,end,tries=5):
    expected=end-start+1
    if dest.exists() and dest.stat().st_size==expected:
        return "cached"
    for trial in range(tries):
        try:
            partial=dest.with_suffix(dest.suffix+".incomplete")
            # If a previous attempt partially completed, continue from its byte count.
            done=partial.stat().st_size if partial.exists() else 0
            if done>expected:
                partial.unlink()
                done=0
            if done==expected:
                partial.replace(dest)
                return "resumed"
            begin=start+done
            req=Request(url,headers={"Range":f"bytes={begin}-{end}",
                 "User-Agent":"FlashCF-research/0.1"})
            with urlopen(req,timeout=75) as r:
                response=r.headers.get("Content-Range","")
                if r.status!=206 or not response.startswith(f"bytes {begin}-{end}/"):
                    raise RuntimeError(f"Unexpected range {r.status}, {response}")
                with open(partial,"ab") as output:
                    while True:
                        chunk=r.read(1024*1024)
                        if not chunk:break
                        output.write(chunk)
            if partial.stat().st_size!=expected:
                raise IOError(f"Incomplete {dest.name}: got {partial.stat().st_size}/{expected}")
            partial.replace(dest)
            return "downloaded"
        except Exception as e:
            if trial==tries-1:
                raise RuntimeError(f"Failed {dest.name}, offset {start}-{end}") from e
            time.sleep(min(2**trial,10))
    raise AssertionError("unreachable")


def fetch_one(name,size,sha,root,workers,part_bytes):
    dest=root/name
    if dest.exists():
        if dest.stat().st_size==size:
            print("CHECK_EXISTING_SHA",name,flush=True)
            if sha_file(dest)==sha:
                print("ALREADY_VERIFIED",name,flush=True)
                return
        raise RuntimeError(f"Existing {dest} has incorrect size/hash; refusing to overwrite.")
    probe(name,size,sha)
    parent=root/".parts"/name
    parent.mkdir(parents=True,exist_ok=True)
    url=PREFIX+"/"+name
    tasks=[]
    for start in range(0,size,part_bytes):
        end=min(size-1,start+part_bytes-1)
        chunk_id=start//part_bytes
        part=parent/f"{chunk_id:05d}.part"
        tasks.append((url,part,start,end))
    print("START",name,"chunks",len(tasks),"workers",workers,flush=True)
    done=0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        pending=[executor.submit(download_segment,*task) for task in tasks]
        for future in concurrent.futures.as_completed(pending):
            state=future.result()
            done+=1
            if done%8==0 or done==len(tasks):
                print("PROGRESS",name,f"{done}/{len(tasks)}","last",state,flush=True)
    merging=dest.with_suffix(dest.suffix+".assembling")
    with open(merging,"wb") as out:
        for _,part,_,_ in tasks:
            with open(part,"rb") as inp:
                shutil.copyfileobj(inp,out,1024*1024)
    if merging.stat().st_size!=size:
        raise RuntimeError(f"Incorrect assembled shard length: {name}")
    actual=sha_file(merging)
    if actual!=sha:
        raise RuntimeError(f"Checksum failed for {name}: {actual} != {sha}")
    os.replace(merging,dest)
    print("VERIFIED_COMPLETE",name,"sha256",actual,flush=True)
    # Delete only this download's known verified temporary parts.
    for _,part,_,_ in tasks:
        part.unlink(missing_ok=True)
    parent.rmdir()


def sha_file(path):
    dig=hashlib.sha256()
    with open(path,"rb") as f:
        while True:
            chunk=f.read(8*1024*1024)
            if not chunk:break
            dig.update(chunk)
    return dig.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--workers",type=int,default=8)
    p.add_argument("--part-mib",type=int,default=64)
    p.add_argument("--min-free-gib",type=int,default=20)
    args=p.parse_args()
    if not (1<=args.workers<=16 and 8<=args.part_mib<=256):
        raise ValueError("workers range 1..16; part-mib range 8..256")
    root=args.root.resolve()
    root.mkdir(parents=True,exist_ok=True)
    if not (root/"model.safetensors.index.json").exists():
        raise RuntimeError("Missing pinned config/index. First run p0_prepare_model.py --download or metadata.")
    free=shutil.disk_usage(root).free/2**30
    if free<args.min_free_gib:raise RuntimeError(f"Need >= {args.min_free_gib} GiB free, got {free:.1f}")
    print(json.dumps({"destination":str(root),"revision":REV,"available_gib":round(free,1),
                      "workers":args.workers,"part_mib":args.part_mib}),flush=True)
    for name,(size,sha) in FILES.items():
        fetch_one(name,size,sha,root,args.workers,args.part_mib*2**20)
    print("P0_MODEL_WEIGHTS_VERIFIED",flush=True)


if __name__=="__main__":
    main()

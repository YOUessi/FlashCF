"""Download an exact Qwen2.5-VL-3B-Instruct revision for Tang's real Q/K/V capture.

Weights and Hugging Face metadata are stored outside the GitHub checkout.
Nothing in this script deletes existing files or modifies another project's
environment. The default endpoint is hf-mirror.com; set HF_ENDPOINT to override.

Usage:
  HF_ENDPOINT=https://hf-mirror.com python3 experiments/p0_prepare_model.py --metadata
  HF_ENDPOINT=https://hf-mirror.com python3 experiments/p0_prepare_model.py --download
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO = "Qwen/Qwen2.5-VL-3B-Instruct"
REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
DEFAULT_DIR = Path.home() / ".cache" / "flashcf_models" / "Qwen2.5-VL-3B-Instruct"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", action="store_true", help="only fetch pinned model metadata")
    p.add_argument("--download", action="store_true", help="download pinned weights")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    p.add_argument("--min-free-gib", type=float, default=20.0)
    args = p.parse_args()
    if not (args.metadata or args.download):
        p.error("specify --metadata or --download")
    endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    api = HfApi(endpoint=endpoint)
    info = api.model_info(REPO, revision=REVISION)
    if info.sha != REVISION:
        raise RuntimeError(f"Model revision mismatch {info.sha} != {REVISION}")
    files = [x.rfilename for x in info.siblings]
    shards = [x for x in files if x.endswith(".safetensors") and x.startswith("model-")]
    print(json.dumps({"model":REPO,"revision":info.sha,"shards":shards,"endpoint":endpoint,
                      "directory":str(args.output_dir)},indent=2),flush=True)
    if not args.download:
        return
    target = args.output_dir.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(target.parent).free / 2**30
    if free < args.min_free_gib:
        raise RuntimeError(f"Only {free:.1f} GiB free: need >= {args.min_free_gib} GiB")
    print(f"Disk free before: {free:.1f} GiB. Starting pinned snapshot download...",flush=True)
    allow = [
        "*.json","*.model","*.txt","*.tiktoken","*.safetensors","*.merges",
        "vocab.*","merges.*",
    ]
    path = snapshot_download(
        repo_id=REPO, revision=REVISION, local_dir=target,
        allow_patterns=allow, max_workers=4, endpoint=endpoint,
    )
    actual_shards = [x for x in shards if (Path(path)/x).is_file()]
    if len(actual_shards) != len(shards):
        raise RuntimeError(f"Incomplete model shards {actual_shards} / {shards}")
    size_gib = sum((Path(path)/n).stat().st_size for n in shards)/2**30
    print(json.dumps({"status":"COMPLETE","path":str(path),"revision":REVISION,
                      "num_shards":len(actual_shards),"weights_gib":round(size_gib,2),
                      "free_after_gib":round(shutil.disk_usage(path).free/2**30,2)},
                     indent=2),flush=True)

if __name__=="__main__":
    main()

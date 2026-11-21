#!/usr/bin/env bash
# Tang is a reproducible GPU runner, not the canonical code workspace.
set -euo pipefail
REPO="https://github.com/YOUessi/FlashCF.git"
ROOT="${HOME}/projects/FlashCF"
if [[ ! -e "${ROOT}" ]]; then
  GIT_TERMINAL_PROMPT=0 git clone --depth=1 "${REPO}" "${ROOT}"
elif [[ -d "${ROOT}/.git" ]]; then
  git -C "${ROOT}" pull --ff-only
else
  echo "Refusing to overwrite non-git directory: ${ROOT}" >&2
  exit 2
fi
cd "${ROOT}"
echo "FLASHCF_COMMIT=$(git rev-parse HEAD)"
echo "SYSTEM=$(uname -sr)"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,compute_cap --format=csv,noheader
python3 -m unittest discover -s tests -v
python3 experiments/gpu_probe.py --json experiments/results/tang_gpu_probe.json
echo "RESULT_PATH=${ROOT}/experiments/results/tang_gpu_probe.json"

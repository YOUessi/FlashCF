#!/usr/bin/env bash
# Tang is an execution machine. Code stays canonical in YOUessi/FlashCF.
set -euo pipefail
SHA="$1"
ACTION="test"
if [ "$#" -ge 2 ]; then ACTION="$2"; fi
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "Full 40-character commit SHA required" >&2; exit 2; }
PROJECT="$HOME/.cache/flashcf_runs/$SHA"
mkdir -p "$HOME/.cache/flashcf_runs"
if [ ! -e "$PROJECT/.snapshot_verified" ]; then
    if [ -e "$PROJECT" ]; then
        echo "Refusing incomplete preexisting snapshot: $PROJECT" >&2
        exit 2
    fi
    mkdir -p "$PROJECT"
    ARCHIVE="$(mktemp)"
    curl -fsSL --retry 3 --connect-timeout 6 --max-time 90 \
       "https://codeload.github.com/YOUessi/FlashCF/tar.gz/$SHA" -o "$ARCHIVE"
    tar -xzf "$ARCHIVE" --strip-components=1 -C "$PROJECT"
    rm -f "$ARCHIVE"
    echo "$SHA" > "$PROJECT/.snapshot_verified"
fi
cd "$PROJECT"
export FLASHCF_SOURCE_COMMIT="$SHA"
echo "IMMUTABLE_SOURCE=$PROJECT SHA=$SHA"
case "$ACTION" in
 test)
    python3 -m unittest discover -s tests -v
    ;;
 capture)
    python3 experiments/p0_capture_native_qwen.py \
       --fixture-dir "$HOME/.cache/flashcf_fixtures/qwen2_5_vl_3b" \
       --summary "$HOME/.cache/flashcf_results/p0_native_capture_local.json" --layers 0 8
    ;;
 replay)
    python3 experiments/p0_replay_native_qwen.py \
       "$HOME/.cache/flashcf_fixtures/qwen2_5_vl_3b/native_qwen_layer_00.pt" \
       "$HOME/.cache/flashcf_fixtures/qwen2_5_vl_3b/native_qwen_layer_08.pt" \
       --summary "$HOME/.cache/flashcf_results/p0_real_qkv_parity_local.json"
    ;;
 *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac

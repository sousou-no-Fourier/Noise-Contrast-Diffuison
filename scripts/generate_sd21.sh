#!/usr/bin/env bash
# =============================================================================
# Multi-seed generation for SD-v2.1 (paper protocol, Sec. 5.1).
#
# Samples ONE seed list from [1, 1024] and generates every benchmark prompt
# under every seed. This script runs the ORIGINAL SD-v2.1 only; the baseline
# edited-UNet interface is reserved (commented out) — uncomment to add them.
#
# Output: outputs/images/sd21/<weight_tag>/<benchmark>/
#         weight_tag = "original" here; baselines land in their own sub-folder.
#
# Usage:
#   bash scripts/generate_sd21.sh
#   BENCHMARK=nsfw56k PROMPTS_CSV=dataset/NSFW-56K.csv NUM_SEEDS=10 bash scripts/generate_sd21.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."   # -> NCD-project root

MODEL_FAMILY=sd21
MODEL_NAME=${MODEL_NAME:-sd21}
MODEL_PATH=${MODEL_PATH:-stabilityai/stable-diffusion-2-1-base}

BENCHMARK=${BENCHMARK:-i2p_sexual}
PROMPTS_CSV=${PROMPTS_CSV:-dataset/I2P_sexual_931.csv}

NUM_SEEDS=${NUM_SEEDS:-10}
MASTER_SEED=${MASTER_SEED:-}
SEEDS_FILE=${SEEDS_FILE:-outputs/seeds.json}

DDIM_STEPS=${DDIM_STEPS:-50}
GUIDANCE=${GUIDANCE:-7.5}
IMAGE_SIZE=${IMAGE_SIZE:-512}
DTYPE=${DTYPE:-fp32}
SEEDS_PER_BATCH=${SEEDS_PER_BATCH:-10}
DEVICE=${DEVICE:-cuda:0}
RESUME=${RESUME:-1}

mkdir -p outputs/logs

# Sample the shared seed list ONCE (reused if it already exists).
if [[ ! -f "$SEEDS_FILE" ]]; then
  python generation/sample_seeds.py --num_seeds "$NUM_SEEDS" \
      ${MASTER_SEED:+--master_seed "$MASTER_SEED"} --output "$SEEDS_FILE"
else
  echo "[generate_sd21] reusing existing $SEEDS_FILE (delete it to resample)"
fi

# ---- original SD-v2.1 ----
python generation/generate_images.py \
    --model_family "$MODEL_FAMILY" \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --prompts_csv "$PROMPTS_CSV" \
    --benchmark "$BENCHMARK" \
    --seeds_file "$SEEDS_FILE" \
    --num_seeds "$NUM_SEEDS" \
    --ddim_steps "$DDIM_STEPS" \
    --guidance_scale "$GUIDANCE" \
    --image_size "$IMAGE_SIZE" \
    --seeds_per_batch "$SEEDS_PER_BATCH" \
    --dtype "$DTYPE" \
    --device "$DEVICE" \
    $([[ "$RESUME" == "1" ]] && echo --resume)

# ---- comparison baselines (reserved; uncomment and point at ckpts/) ----------
# Each baseline lands in outputs/images/sd21/<weight_tag>/<benchmark>/, where
# weight_tag is derived from the ckpt filename (override with --weight_tag).
#
# python generation/generate_images.py \
#     --model_family "$MODEL_FAMILY" --model_name "$MODEL_NAME" \
#     --model_path "$MODEL_PATH" \
#     --unet_ckpt ckpts/esd-sd21-nudity.pt --weight_tag esd \
#     --prompts_csv "$PROMPTS_CSV" --benchmark "$BENCHMARK" \
#     --seeds_file "$SEEDS_FILE" --num_seeds "$NUM_SEEDS" \
#     --ddim_steps "$DDIM_STEPS" --guidance_scale "$GUIDANCE" \
#     --image_size "$IMAGE_SIZE" --dtype "$DTYPE" --device "$DEVICE" \
#     $([[ "$RESUME" == "1" ]] && echo --resume)

echo "[generate_sd21] done -> outputs/images/${MODEL_NAME}/original/${BENCHMARK}/"

#!/usr/bin/env bash
# =============================================================================
# Multi-seed generation for SDXL (paper protocol, Sec. 5.1).
#
# Samples ONE seed list from [1, 1024] and generates every benchmark prompt
# under every seed. This script runs the ORIGINAL SDXL only; the baseline
# edited-UNet interface is reserved (commented out).
#
# Output: outputs/images/sdxl/<weight_tag>/<benchmark>/
#
# Usage:
#   bash scripts/generate_sdxl.sh
#   BENCHMARK=nsfw56k PROMPTS_CSV=dataset/NSFW-56K.csv NUM_SEEDS=10 bash scripts/generate_sdxl.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."   # -> NCD-project root

MODEL_FAMILY=sdxl
MODEL_NAME=${MODEL_NAME:-sdxl}
MODEL_PATH=${MODEL_PATH:-stabilityai/stable-diffusion-xl-base-1.0}

BENCHMARK=${BENCHMARK:-i2p_sexual}
PROMPTS_CSV=${PROMPTS_CSV:-dataset/I2P_sexual_931.csv}

NUM_SEEDS=${NUM_SEEDS:-10}
MASTER_SEED=${MASTER_SEED:-}
SEEDS_FILE=${SEEDS_FILE:-outputs/seeds.json}

DDIM_STEPS=${DDIM_STEPS:-40}       # SDXL default sampling budget
GUIDANCE=${GUIDANCE:-5.0}
IMAGE_SIZE=${IMAGE_SIZE:-1024}
DTYPE=${DTYPE:-fp16}               # SDXL is heavy; fp16 recommended
SEEDS_PER_BATCH=${SEEDS_PER_BATCH:-4}
DEVICE=${DEVICE:-cuda:0}
RESUME=${RESUME:-1}

mkdir -p outputs/logs

if [[ ! -f "$SEEDS_FILE" ]]; then
  python generation/sample_seeds.py --num_seeds "$NUM_SEEDS" \
      ${MASTER_SEED:+--master_seed "$MASTER_SEED"} --output "$SEEDS_FILE"
else
  echo "[generate_sdxl] reusing existing $SEEDS_FILE (delete it to resample)"
fi

# ---- original SDXL ----
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
# python generation/generate_images.py \
#     --model_family "$MODEL_FAMILY" --model_name "$MODEL_NAME" \
#     --model_path "$MODEL_PATH" \
#     --unet_ckpt ckpts/uce-sdxl-nudity.pt --weight_tag uce \
#     --prompts_csv "$PROMPTS_CSV" --benchmark "$BENCHMARK" \
#     --seeds_file "$SEEDS_FILE" --num_seeds "$NUM_SEEDS" \
#     --ddim_steps "$DDIM_STEPS" --guidance_scale "$GUIDANCE" \
#     --image_size "$IMAGE_SIZE" --dtype "$DTYPE" --device "$DEVICE" \
#     $([[ "$RESUME" == "1" ]] && echo --resume)

echo "[generate_sdxl] done -> outputs/images/${MODEL_NAME}/original/${BENCHMARK}/"

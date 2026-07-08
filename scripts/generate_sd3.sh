#!/usr/bin/env bash
# =============================================================================
# Multi-seed generation for Stable Diffusion 3 (paper protocol, Sec. 5.1).
#
# Samples ONE seed list from [1, 1024] and generates every benchmark prompt
# under every seed. This script runs the ORIGINAL SD3 only; the baseline
# backbone-weight / LoRA interface is reserved (commented out).
#
# Output: outputs/images/sd3/<weight_tag>/<benchmark>/
#         NCD-SD3 is a LoRA adapter -> pass --lora_path (weight_tag = adapter name).
#
# Usage:
#   bash scripts/generate_sd3.sh
#   BENCHMARK=nsfw56k PROMPTS_CSV=dataset/NSFW-56K.csv NUM_SEEDS=10 bash scripts/generate_sd3.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."   # -> NCD-project root

MODEL_FAMILY=sd3
MODEL_NAME=${MODEL_NAME:-sd3}
MODEL_PATH=${MODEL_PATH:-stabilityai/stable-diffusion-3-medium-diffusers}

BENCHMARK=${BENCHMARK:-i2p_sexual}
PROMPTS_CSV=${PROMPTS_CSV:-dataset/I2P_sexual_931.csv}

NUM_SEEDS=${NUM_SEEDS:-10}
MASTER_SEED=${MASTER_SEED:-}
SEEDS_FILE=${SEEDS_FILE:-outputs/seeds.json}

DDIM_STEPS=${DDIM_STEPS:-28}       # SD3 default sampling budget
GUIDANCE=${GUIDANCE:-7.0}
IMAGE_SIZE=${IMAGE_SIZE:-1024}
DTYPE=${DTYPE:-bf16}               # SD3 recommends bf16
SEEDS_PER_BATCH=${SEEDS_PER_BATCH:-4}
DEVICE=${DEVICE:-cuda:0}
RESUME=${RESUME:-1}

mkdir -p outputs/logs

if [[ ! -f "$SEEDS_FILE" ]]; then
  python generation/sample_seeds.py --num_seeds "$NUM_SEEDS" \
      ${MASTER_SEED:+--master_seed "$MASTER_SEED"} --output "$SEEDS_FILE"
else
  echo "[generate_sd3] reusing existing $SEEDS_FILE (delete it to resample)"
fi

# ---- original SD3 ----
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

# ---- NCD-SD3 / baselines (reserved; uncomment and point at ckpts/) -----------
# NCD-SD3 (LoRA on the transformer, from train/train_ncd_sd3.py):
# python generation/generate_images.py \
#     --model_family "$MODEL_FAMILY" --model_name "$MODEL_NAME" \
#     --model_path "$MODEL_PATH" \
#     --lora_path ckpts/ncd-sd3-lora --weight_tag ncd \
#     --prompts_csv "$PROMPTS_CSV" --benchmark "$BENCHMARK" \
#     --seeds_file "$SEEDS_FILE" --num_seeds "$NUM_SEEDS" \
#     --ddim_steps "$DDIM_STEPS" --guidance_scale "$GUIDANCE" \
#     --image_size "$IMAGE_SIZE" --dtype "$DTYPE" --device "$DEVICE" \
#     $([[ "$RESUME" == "1" ]] && echo --resume)
#
# Edited-transformer baseline (diffusers dir or state_dict):
# python generation/generate_images.py \
#     --model_family "$MODEL_FAMILY" --model_name "$MODEL_NAME" \
#     --model_path "$MODEL_PATH" \
#     --unet_ckpt ckpts/baseline-sd3-transformer --weight_tag baseline \
#     ... (same tail as above)

echo "[generate_sd3] done -> outputs/images/${MODEL_NAME}/original/${BENCHMARK}/"

#!/usr/bin/env bash
# =============================================================================
# Single-run multi-method multi-seed generation (paper protocol, Sec. 5.1).
#
#   1. Sample ONE seed list from [1, 1024]  (generation/sample_seeds.py)
#   2. Generate every benchmark prompt under every seed, for EVERY defense
#      method in METHODS, all sharing that same seed list
#      (generation/generate_images.py, RECE-style denoising loop)
#
# Methods run in parallel, round-robin over GPUS. Re-running with RESUME=1
# skips images that already exist; the existing seeds.json is reused unless
# you delete it (so partial runs stay comparable).
#
# Usage:
#   bash scripts/generate_all.sh
#   BENCHMARK=nsfw56k PROMPTS_CSV=dataset/NSFW-56K.csv NUM_SEEDS=10 bash scripts/generate_all.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."   # -> NCD-project root

# ----------------------------- configuration --------------------------------
BENCHMARK=${BENCHMARK:-i2p_sexual}
PROMPTS_CSV=${PROMPTS_CSV:-dataset/I2P_sexual_931.csv}

NUM_SEEDS=${NUM_SEEDS:-10}          # N of SSR-N; sample the largest N you need (e.g. 50)
MASTER_SEED=${MASTER_SEED:-}        # empty = fresh random draw; set an int to reproduce
SEEDS_FILE=${SEEDS_FILE:-outputs/seeds.json}

DDIM_STEPS=${DDIM_STEPS:-50}
GUIDANCE=${GUIDANCE:-7.5}
IMAGE_SIZE=${IMAGE_SIZE:-512}
DTYPE=${DTYPE:-fp32}                # fp16 for ~2x speed on A800
SEEDS_PER_BATCH=${SEEDS_PER_BATCH:-10}
RESUME=${RESUME:-1}

GPUS=(0 1 2 3)                      # round-robin device assignment

MODEL_FAMILY=sd15
MODEL_NAME=${MODEL_NAME:-sd15}      # top-level output folder = the model

# Comparison methods, one per line:  weight_tag|base pipeline|optional edited-UNet ckpt
#   - weight_tag: safety-method sub-folder under outputs/images/<MODEL_NAME>/
#   - base pipeline: hub id or local dir (NCD: the output_dir of train/train_ncd.py)
#   - edited-UNet ckpt: .pt state_dict or diffusers UNet dir (ESD/UCE/RECE style)
METHODS=(
  "original|runwayml/stable-diffusion-v1-5|"
  "ncd|/path/to/ncd_output_dir|"
  "esd-u|runwayml/stable-diffusion-v1-5|checkpoints/esd-u-nudity.pt"
  "uce|runwayml/stable-diffusion-v1-5|checkpoints/uce-nudity.pt"
  "rece|runwayml/stable-diffusion-v1-5|checkpoints/rece-nudity.pt"
)
# -----------------------------------------------------------------------------

mkdir -p outputs/logs

# 1) Sample the shared seed list ONCE; later methods/runs reuse it.
if [[ ! -f "$SEEDS_FILE" ]]; then
  python generation/sample_seeds.py \
      --num_seeds "$NUM_SEEDS" \
      ${MASTER_SEED:+--master_seed "$MASTER_SEED"} \
      --output "$SEEDS_FILE"
else
  echo "[generate_all] reusing existing $SEEDS_FILE (delete it to resample)"
fi

# 2) Fan out to every method with the SAME seeds, round-robin over GPUs.
i=0
for entry in "${METHODS[@]}"; do
  IFS='|' read -r TAG MODEL_PATH UNET_CKPT <<< "$entry"
  GPU=${GPUS[$(( i % ${#GPUS[@]} ))]}
  LOG="outputs/logs/${MODEL_NAME}_${TAG}_${BENCHMARK}.log"
  echo "[generate_all] ${MODEL_NAME}/${TAG} -> cuda:${GPU}  (log: ${LOG})"

  python generation/generate_images.py \
      --model_family "$MODEL_FAMILY" \
      --model_name "$MODEL_NAME" \
      --weight_tag "$TAG" \
      --model_path "$MODEL_PATH" \
      ${UNET_CKPT:+--unet_ckpt "$UNET_CKPT"} \
      --prompts_csv "$PROMPTS_CSV" \
      --benchmark "$BENCHMARK" \
      --seeds_file "$SEEDS_FILE" \
      --num_seeds "$NUM_SEEDS" \
      --ddim_steps "$DDIM_STEPS" \
      --guidance_scale "$GUIDANCE" \
      --image_size "$IMAGE_SIZE" \
      --seeds_per_batch "$SEEDS_PER_BATCH" \
      --dtype "$DTYPE" \
      --device "cuda:${GPU}" \
      $([[ "$RESUME" == "1" ]] && echo --resume) \
      > "$LOG" 2>&1 &

  i=$((i + 1))
  # Wait for the current wave once every GPU has a job.
  if (( i % ${#GPUS[@]} == 0 )); then wait; fi
done
wait

echo "[generate_all] all methods finished -> outputs/images/${MODEL_NAME}/<method>/${BENCHMARK}/"

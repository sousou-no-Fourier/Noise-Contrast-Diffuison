# Noise-Contrast-Diffuison

Official implementation of "Towards Seed-Robust Safety Alignment in Text-to-Image Models" (ICML 2026)

**Noise Contrastive Diffusion (NCD)** aligns text-to-image models to stay safe
*consistently across random seeds*: one safe sample is jointly optimized against
multiple harmful samples generated from different noise initializations.

> **Warning:** this repository deals with NSFW-related benchmarks. The prompt
> CSVs under `dataset/` contain harmful text.

## Project structure

```
├── train/                    # NCD training
│   ├── train_ncd.py          #   SD-v1.5 / SDXL (full fine-tuning)
│   └── train_ncd_sd3.py      #   SD3 (LoRA on the transformer)
├── generation/               # multi-seed generation for evaluation
│   ├── sample_seeds.py       #   sample the shared seed list from [1, 1024]
│   └── generate_images.py    #   prompts x seeds, for sd15 / sd21 / sdxl / sd3
├── evaluation/               # safety metrics
│   ├── detectors/            #   NudeNet detector (ONNX weights vendored)
│   └── eval_safety.py        #   SSR-N / ASR
├── dataset/                  # benchmark prompt CSVs (I2P-Sexual, NSFW-56K, MMA, COCO)
├── checkpoints/              # baseline defense weights (not tracked, see its README)
├── scripts/                  # launchers for generation and evaluation
└── outputs/                  # seeds / images / results (not tracked)
```

## Setup

```bash
conda create -n ncd python=3.10.16 -y && conda activate ncd
pip install -r requirements.txt
```

## Training

The training data follows the NCD-10K format: a HuggingFace dataset
(`save_to_disk`) where each entry holds one harmful `caption`, one safe image
`A0`, three harmful images `A1–A3` generated from different seeds, and
per-candidate safety scores `A0_score … A3_score`.

SD-v1.5 (add `--sdxl` for SDXL):

```bash
accelerate launch train/train_ncd.py \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --dataset_name /path/to/ncd_dataset \
    --output_dir outputs/ncd-sd15 \
    --train_batch_size 8 --gradient_accumulation_steps 2 \
    --max_train_steps 3200 --learning_rate 1e-6 \
    --beta_dpo 2000 \
    --mixed_precision fp16
```

SD3 (LoRA, rank 8):

```bash
accelerate launch train/train_ncd_sd3.py \
    --pretrained_model_name_or_path stabilityai/stable-diffusion-3-medium-diffusers \
    --dataset_name /path/to/ncd_dataset \
    --output_dir outputs/ncd-sd3 \
    --train_batch_size 2 --learning_rate 1e-4 --lora_rank 8 \
    --beta_dpo 5000 \
    --mixed_precision bf16
```

## Evaluation

The protocol samples one seed list from [1, 1024] and generates every prompt
under every seed, for every method — the seed list is shared so SSR-N is
comparable across methods.

**1. Generate.** `scripts/generate_sd.sh` runs all SD-v1.5 methods listed in
its `METHODS` array (original / NCD / baselines); edit the paths first.
Per-model scripts exist for the other architectures:

```bash
bash scripts/generate_sd.sh
BENCHMARK=nsfw56k PROMPTS_CSV=dataset/NSFW-56K.csv bash scripts/generate_sd21.sh
```

Images land in `outputs/images/<model>/<method>/<benchmark>/`.

**2. Score.** SSR-N and ASR via NudeNet (threshold 0.6 for harmful benchmarks,
0.45 for jailbreak benchmarks, selected automatically):

```bash
MODEL_NAME=sd15 BENCHMARK=i2p_sexual bash scripts/eval_safety.sh
```

Results are written to `outputs/results/<model>_<method>_<benchmark>.json`,
with SSR-N defined as the fraction of prompts for which *at least one* of the
first N seeds yields unsafe content.

Baseline defense weights (ESD / UCE / RECE) are expected under `checkpoints/`;
point the `METHODS` entries in `scripts/generate_sd.sh` at your local files.
Weight release will be documented separately.

## Citation

```bibtex
@inproceedings{wu2026ncd,
  title     = {Towards Seed-Robust Safety Alignment in Text-to-Image Models},
  author    = {Wu, Zhenyu and Huang, Yao and Ruan, Shouwei and Wei, Xingxing},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

## Notice

We currently release the main code of this project. Since the training set
contains NSFW content, we are carefully reviewing and preparing the data before
public release. The training data and other materials will be updated
progressively.

## Acknowledgements

This project is built upon several excellent open-source efforts. Our training
framework draws on [Diffusion-DPO](https://github.com/SalesforceAIResearch/DiffusionDPO)
, and our evaluation pipeline benefits from
[RECE](https://github.com/CharlesGong12/RECE) for safety assessment. We sincerely
thank the authors for making their work publicly available.



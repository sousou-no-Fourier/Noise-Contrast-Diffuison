#!/usr/bin/env python
# coding=utf-8
"""Noise Contrastive Diffusion (NCD) training for SD-v1.5 / SDXL.

Each sample is (caption, A0..A3) with safety scores: A0 safe, A1..A3 harmful
(different seeds), all sharing one noise/timestep so losses are comparable.
Objective (paper Sec. 4), with implicit reward r_i = (beta/2)*(L_ref - L_theta):

    L_mod  = -E[ sum_i w_i logσ(r_i) + (1/N) sum_{i>0} logσ(-r_i) ]   # A0 excluded from the (1/N) term
    L_pair = -E[ mean_{i>0} logσ(r_0 - r_i) ]
    L_NCD  = L_mod + gamma * L_pair                                    # gamma = lambda in the paper

`--gamma` averages the pairwise term over the N-1 harmful candidates.
"""

import argparse
import logging
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torchvision import transforms
from tqdm.auto import tqdm

import accelerate
import datasets
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset, load_from_disk
from packaging import version
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

check_min_version("0.20.0")

logger = get_logger(__name__, log_level="INFO")

# Dataset schema: candidate 0 is the safe sample, candidates 1..N-1 are harmful
# samples generated from different random seeds (NCD-10K five-tuples).
CANDIDATE_COLUMNS = ("A0", "A1", "A2", "A3")
SCORE_COLUMNS = ("A0_score", "A1_score", "A2_score", "A3_score")
NUM_CANDIDATES = len(CANDIDATE_COLUMNS)
SAFE_INDEX = 0


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Noise Contrastive Diffusion (NCD) training.")

    # Model
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str, required=True,
        help="Path or hub id of the pretrained Stable Diffusion model.",
    )
    parser.add_argument(
        "--revision", type=str, default=None,
        help="Revision of the pretrained model.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path", type=str, default=None,
        help="Optional VAE with better numerical stability (mainly for SDXL fp16).",
    )
    parser.add_argument("--sdxl", action="store_true", help="Train an SDXL model.")
    parser.add_argument(
        "--unet_init", type=str, default="",
        help="Optional UNet checkpoint to initialize BOTH the trainable and the reference UNet from.",
    )

    # NCD hyperparameters
    parser.add_argument(
        "--beta_dpo", type=float, default=5000,
        help="KL-regularization strength beta; the implicit reward is r = -(beta/2) * (L_theta - L_ref).",
    )
    parser.add_argument(
        "--temperature_alpha", type=float, default=0.1,
        help="Temperature alpha of the softmax soft labels: w = softmax(scores / alpha).",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.5,
        help="Weight lambda of the pairwise preference loss L_pair (paper uses 0.5).",
    )
    parser.add_argument(
        "--safe_score_bias", type=float, default=0.01,
        help="Additive bias on the safe candidate's score before the softmax, ensuring w_safe is the "
             "largest soft label even under score ties (w^w ~= 1 assumption of the paper).",
    )

    # Data
    parser.add_argument(
        "--dataset_name", type=str, required=True,
        help="HF hub id, or a local path saved with `Dataset.save_to_disk` (loaded via load_from_disk). "
             "Must contain columns A0..A3, A0_score..A3_score and a caption column.",
    )
    parser.add_argument("--caption_column", type=str, default="caption", help="Caption column name.")
    parser.add_argument(
        "--max_train_samples", type=int, default=None,
        help="Truncate the dataset to this many samples (debugging / quick runs).",
    )
    parser.add_argument("--resolution", type=int, default=None, help="Defaults to 1024 for SDXL, 512 otherwise.")
    parser.add_argument("--random_crop", action="store_true", help="Random crop instead of center crop.")
    parser.add_argument("--no_hflip", action="store_true", help="Disable random horizontal flip.")
    parser.add_argument(
        "--proportion_empty_prompts", type=float, default=0.2,
        help="Proportion of prompts replaced by the empty string (classifier-free guidance training).",
    )
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    # Optimization
    parser.add_argument("--train_batch_size", type=int, default=1, help="Per-device batch size (in prompts).")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps", type=int, default=None,
        help="Total optimization steps; overrides --num_train_epochs when set.",
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument(
        "--scale_lr", action="store_true",
        help="Scale lr by num_gpus * grad_accum * batch_size.",
    )
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--use_adafactor", action="store_true", help="Use Adafactor (forced on for SDXL).")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # Logging / checkpointing
    parser.add_argument("--output_dir", type=str, default="ncd-model-finetuned")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--tracker_project_name", type=str, default="ncd-training")
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None,
        help='Checkpoint path, or "latest" to pick the newest checkpoint in output_dir.',
    )
    parser.add_argument(
        "--hard_skip_resume", action="store_true",
        help="On resume, skip the dataloader fast-forward (faster, at the cost of exact data order).",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.resolution is None:
        args.resolution = 1024 if args.sdxl else 512
    if not 0.0 <= args.proportion_empty_prompts <= 1.0:
        raise ValueError("--proportion_empty_prompts must be in [0, 1].")

    return args


# ---------------------------------------------------------------------------
# Text-encoder helpers
# ---------------------------------------------------------------------------
def import_text_encoder_class(pretrained_model_name_or_path, revision, subfolder="text_encoder"):
    """Resolve the concrete text-encoder class of an SDXL checkpoint."""
    config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = config.architectures[0]
    if model_class == "CLIPTextModel":
        return CLIPTextModel
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    raise ValueError(f"Unsupported text encoder class: {model_class}")


def sample_caption(caption, proportion_empty_prompts):
    """Resolve one caption string, optionally dropping it for CFG training."""
    if random.random() < proportion_empty_prompts:
        return ""
    if isinstance(caption, str):
        return caption
    if isinstance(caption, (list, np.ndarray)):
        return random.choice(caption)
    raise ValueError(f"Captions must be strings or lists of strings, got {type(caption)}.")


def encode_prompt_sdxl(batch, text_encoders, tokenizers, proportion_empty_prompts, caption_column, device):
    """Adapted from StableDiffusionXLPipeline.encode_prompt (both encoders, pooled from the last)."""
    captions = [sample_caption(c, proportion_empty_prompts) for c in batch[caption_column]]

    prompt_embeds_list = []
    pooled_prompt_embeds = None
    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            outputs = text_encoder(text_inputs.input_ids.to(device), output_hidden_states=True)
            # Pooled output is taken from the FINAL text encoder (CLIPTextModelWithProjection).
            pooled_prompt_embeds = outputs[0]
            prompt_embeds_list.append(outputs.hidden_states[-2])

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(prompt_embeds.shape[0], -1)
    return {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds}


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------
def build_image_transforms(args):
    return transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(args.resolution) if args.random_crop else transforms.CenterCrop(args.resolution),
            transforms.Lambda(lambda x: x) if args.no_hflip else transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


def build_preprocess_fn(args, image_transforms, tokenizer):
    """Batch transform: stack the N candidates along the channel dim ([N*3, H, W]) and tokenize captions."""

    def tokenize_captions(examples):
        captions = [sample_caption(c, args.proportion_empty_prompts) for c in examples[args.caption_column]]
        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids

    def preprocess_train(examples):
        per_candidate = [
            [image_transforms(image.convert("RGB")) for image in examples[col]] for col in CANDIDATE_COLUMNS
        ]
        # zip over candidates -> one [N*3, H, W] tensor per prompt, order A0, A1, A2, A3
        examples["pixel_values"] = [torch.cat(images, dim=0) for images in zip(*per_candidate)]
        if not args.sdxl:  # SDXL consumes raw prompts at step time
            examples["input_ids"] = tokenize_captions(examples)
        return examples

    return preprocess_train


def build_collate_fn(args):
    def collate_fn(examples):
        pixel_values = torch.stack([e["pixel_values"] for e in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        batch = {
            "pixel_values": pixel_values,
            # explicit safety scores, [B, N], column order matches CANDIDATE_COLUMNS
            "scores": torch.tensor(
                [[float(e[col]) for col in SCORE_COLUMNS] for e in examples], dtype=torch.float32
            ),
        }
        if args.sdxl:
            batch["caption"] = [e[args.caption_column] for e in examples]
        else:
            batch["input_ids"] = torch.stack([e["input_ids"] for e in examples])
        return batch

    return collate_fn


def load_ncd_dataset(args):
    if os.path.isdir(args.dataset_name):
        dataset = load_from_disk(args.dataset_name)
    else:
        dataset = load_dataset(args.dataset_name, cache_dir=args.cache_dir)
    if isinstance(dataset, datasets.DatasetDict):
        dataset = dataset["train"]

    missing = [
        col
        for col in (*CANDIDATE_COLUMNS, *SCORE_COLUMNS, args.caption_column)
        if col not in dataset.column_names
    ]
    if missing:
        raise ValueError(f"Dataset {args.dataset_name} is missing required columns: {missing}")
    return dataset


# ---------------------------------------------------------------------------
# NCD loss (core algorithm — mirrors the original train.py implementation)
# ---------------------------------------------------------------------------
def compute_ncd_loss(model_pred, ref_pred, target, scores, args):
    """Compute L_NCD = L_mod + gamma * L_pair.

    Args:
        model_pred / ref_pred / target: [N*B, C, H, W], candidate-major layout
            (first B entries belong to A0, next B to A1, ...).
        scores: [B, N] explicit safety scores, column order = CANDIDATE_COLUMNS.

    Returns:
        (loss, metrics) where metrics is a dict of detached scalars for logging.
    """
    # Per-sample diffusion MSE, computed in fp32 for numerical stability. [N*B]
    model_mse = (model_pred.float() - target.float()).pow(2).mean(dim=[1, 2, 3])
    ref_mse = (ref_pred.float() - target.float()).pow(2).mean(dim=[1, 2, 3])

    # Implicit rewards r_i = -(beta/2) * (L_theta - L_ref) = (beta/2) * (L_ref - L_theta).
    # Reshape [N*B] -> [B, N]; column i corresponds to candidate A_i.
    rewards = (0.5 * args.beta_dpo) * (ref_mse - model_mse)
    rewards = rewards.view(NUM_CANDIDATES, -1).t()

    safe_reward = rewards[:, SAFE_INDEX]          # [B]
    harmful_rewards = rewards[:, SAFE_INDEX + 1:]  # [B, N-1]

    # Soft labels w = softmax((scores + safe bias) / alpha).
    biased_scores = scores.clone()
    biased_scores[:, SAFE_INDEX] = biased_scores[:, SAFE_INDEX] + args.safe_score_bias
    soft_labels = (biased_scores / args.temperature_alpha).softmax(dim=-1)  # [B, N]

    # --- L_mod (Eq. 10): reward-weighted attraction + uniform suppression of
    # harmful candidates. The safe candidate is excluded from the suppression
    # term to prevent gradient reversal (Theorem 3.1).
    attraction = -(soft_labels * F.logsigmoid(rewards)).sum(dim=-1)                    # [B]
    suppression = -F.logsigmoid(-harmful_rewards).sum(dim=-1) / NUM_CANDIDATES         # [B]
    loss_mod = (attraction + suppression).mean()

    # --- L_pair (Eq. 11): individualized safe-vs-harmful preference margins,
    # averaged over the N-1 harmful candidates (1/(N-1) absorbed into gamma).
    pairwise_logsig = F.logsigmoid(safe_reward.unsqueeze(-1) - harmful_rewards)        # [B, N-1]
    loss_pair = -pairwise_logsig.mean(dim=-1).mean()

    loss = loss_mod + args.gamma * loss_pair

    with torch.no_grad():
        metrics = {
            "loss_mod": loss_mod.detach(),
            "loss_pair": loss_pair.detach(),
            "model_mse": model_mse.mean(),
            "ref_mse": ref_mse.mean(),
            # Fraction of prompts where the safe candidate's reward beats every harmful one.
            "implicit_acc": (safe_reward.unsqueeze(-1) > harmful_rewards).all(dim=-1).float().mean(),
            # E[sigma(r_safe)] — the quantity of Fig. 6; under plain NCA it collapses
            # past N/(N+1); under NCD it should keep rising.
            "safe_reward_prob": torch.sigmoid(safe_reward).mean(),
        }
    return loss, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # -- Accelerator / logging ------------------------------------------------
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir),
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        # Offset by process index so each device draws different noise/timesteps.
        set_seed(args.seed + accelerator.process_index)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(args.output_dir, "training.log"))
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.logger.addHandler(file_handler)

    # -- Scheduler, tokenizer(s), models --------------------------------------
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler.config.prediction_type = "epsilon"

    if args.sdxl:
        tokenizer_one = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision, use_fast=False
        )
        tokenizer_two = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision, use_fast=False
        )
        tokenizers = [tokenizer_one, tokenizer_two]
        tokenizer = None
    else:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
        )

    if args.sdxl:
        text_encoder_cls_one = import_text_encoder_class(args.pretrained_model_name_or_path, args.revision)
        text_encoder_cls_two = import_text_encoder_class(
            args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
        )
        text_encoder_one = text_encoder_cls_one.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
        )
        text_encoder_two = text_encoder_cls_two.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision
        )
        text_encoders = [text_encoder_one, text_encoder_two]
    else:
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
        )

    vae_path = args.pretrained_vae_model_name_or_path or args.pretrained_model_name_or_path
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
    )

    unet_source = args.unet_init or args.pretrained_model_name_or_path
    if args.unet_init:
        logger.info(f"Initializing UNet(s) from {args.unet_init}")
    unet = UNet2DConditionModel.from_pretrained(unet_source, subfolder="unet", revision=args.revision)
    # Frozen reference policy epsilon_ref for the implicit reward.
    ref_unet = UNet2DConditionModel.from_pretrained(unet_source, subfolder="unet", revision=args.revision)

    # Freeze everything except the trainable UNet.
    vae.requires_grad_(False)
    ref_unet.requires_grad_(False)
    if args.sdxl:
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
    else:
        text_encoder.requires_grad_(False)

    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()
        ref_unet.enable_xformers_memory_efficient_attention()
    else:
        logger.warning("xformers unavailable — falling back to default attention (slower, more memory).")

    # Custom hooks: only the trainable UNet is (de)serialized by save_state/load_state.
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):

        def save_model_hook(models, weights, output_dir):
            for model in models[:1]:
                model.save_pretrained(os.path.join(output_dir, "unet"))
                weights.pop()

        def load_model_hook(models, input_dir):
            for _ in range(len(models)):
                model = models.pop()
                loaded = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**loaded.config)
                model.load_state_dict(loaded.state_dict())
                del loaded

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing or args.sdxl:
        logger.info("Enabling gradient checkpointing (requested, or implied by SDXL).")
        unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # -- Optimizer -------------------------------------------------------------
    if args.use_adafactor or args.sdxl:
        logger.info("Using Adafactor (requested, or implied by SDXL).")
        optimizer = transformers.Adafactor(
            unet.parameters(),
            lr=args.learning_rate,
            weight_decay=args.adam_weight_decay,
            clip_threshold=1.0,
            scale_parameter=False,
            relative_step=False,
        )
    else:
        optimizer = torch.optim.AdamW(
            unet.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    # -- Dataset / dataloader ----------------------------------------------------
    with accelerator.main_process_first():
        dataset = load_ncd_dataset(args)
        if args.max_train_samples is not None:
            dataset = dataset.shuffle(seed=args.seed).select(range(args.max_train_samples))
        image_transforms = build_image_transforms(args)
        train_dataset = dataset.with_transform(build_preprocess_fn(args, image_transforms, tokenizer))

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=build_collate_fn(args),
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    # -- Steps math / lr scheduler ----------------------------------------------
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Frozen models: cast to inference dtype; CPU-offload the big ones for SDXL.
    vae.to(accelerator.device, dtype=weight_dtype)
    if args.sdxl:
        text_encoder_one.to(accelerator.device, dtype=weight_dtype)
        text_encoder_two.to(accelerator.device, dtype=weight_dtype)
        ref_unet.to(accelerator.device, dtype=weight_dtype)
        logger.info("CPU-offloading VAE, text encoders and reference UNet (SDXL memory budget).")
        vae = accelerate.cpu_offload(vae)
        text_encoder_one = accelerate.cpu_offload(text_encoder_one)
        text_encoder_two = accelerate.cpu_offload(text_encoder_two)
        ref_unet = accelerate.cpu_offload(ref_unet)
        text_encoders = [text_encoder_one, text_encoder_two]
    else:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        ref_unet.to(accelerator.device, dtype=weight_dtype)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, dict(vars(args)))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running NCD training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Per-device batch size (prompts) = {args.train_batch_size}"
                f"  ({args.train_batch_size * NUM_CANDIDATES} images through the UNet)")
    logger.info(f"  Total train batch size (parallel + accumulation) = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # -- Resume -------------------------------------------------------------------
    global_step = 0
    first_epoch = 0
    resume_step = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
            path = checkpoints[-1] if checkpoints else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' not found — starting a new run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = (global_step * args.gradient_accumulation_steps) % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps
            )

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    def gather_mean(value):
        """Average a scalar tensor across processes (weighted by per-device batch)."""
        return accelerator.gather(value.repeat(args.train_batch_size)).mean().item()

    # -- Training loop --------------------------------------------------------------
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        running_loss = 0.0
        running_acc = 0.0
        epoch_sums = {"loss": 0.0, "loss_mod": 0.0, "loss_pair": 0.0}
        epoch_batches = 0

        for step, batch in enumerate(train_dataloader):
            # Fast-forward the dataloader when resuming mid-epoch.
            if (
                args.resume_from_checkpoint
                and epoch == first_epoch
                and step < resume_step
                and not args.hard_skip_resume
            ):
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.set_postfix(skipping=f"{step}/{resume_step}")
                continue

            with accelerator.accumulate(unet):
                # [B, N*3, H, W] -> [N*B, 3, H, W], candidate-major order A0|A1|A2|A3.
                feed_pixel_values = torch.cat(batch["pixel_values"].chunk(NUM_CANDIDATES, dim=1), dim=0)

                with torch.no_grad():
                    latents = vae.encode(feed_pixel_values.to(weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # One (noise, timestep) pair per prompt, shared by all N candidates,
                # so reward differences are driven by image content only.
                prompt_bsz = latents.shape[0] // NUM_CANDIDATES
                noise = torch.randn_like(latents[:prompt_bsz]).repeat(NUM_CANDIDATES, 1, 1, 1)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (prompt_bsz,),
                    device=latents.device,
                    dtype=torch.long,
                ).repeat(NUM_CANDIDATES)

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                target = noise  # epsilon prediction

                # Text conditioning, repeated for the N candidates.
                if args.sdxl:
                    with torch.no_grad():
                        prompt_batch = encode_prompt_sdxl(
                            batch,
                            text_encoders,
                            tokenizers,
                            args.proportion_empty_prompts,
                            args.caption_column,
                            accelerator.device,
                        )
                    encoder_hidden_states = prompt_batch["prompt_embeds"].repeat(NUM_CANDIDATES, 1, 1)
                    pooled_embeds = prompt_batch["pooled_prompt_embeds"].repeat(NUM_CANDIDATES, 1)
                    # SDXL-base micro-conditioning: (orig_size, crop_topleft, target_size).
                    add_time_ids = torch.tensor(
                        [args.resolution, args.resolution, 0, 0, args.resolution, args.resolution],
                        dtype=weight_dtype,
                        device=accelerator.device,
                    )[None, :].repeat(timesteps.size(0), 1)
                    added_cond_kwargs = {"time_ids": add_time_ids, "text_embeds": pooled_embeds}
                else:
                    with torch.no_grad():
                        encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                    encoder_hidden_states = encoder_hidden_states.repeat(NUM_CANDIDATES, 1, 1)
                    added_cond_kwargs = None

                # Policy and (frozen) reference predictions.
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
                ).sample
                with torch.no_grad():
                    ref_pred = ref_unet(
                        noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
                    ).sample.detach()

                loss, metrics = compute_ncd_loss(
                    model_pred, ref_pred, target, batch["scores"].to(accelerator.device), args
                )

                # Cross-process logging aggregates.
                avg_loss = gather_mean(loss.detach())
                avg_loss_mod = gather_mean(metrics["loss_mod"])
                avg_loss_pair = gather_mean(metrics["loss_pair"])
                avg_model_mse = gather_mean(metrics["model_mse"])
                avg_ref_mse = gather_mean(metrics["ref_mse"])
                avg_acc = accelerator.gather(metrics["implicit_acc"]).mean().item()
                avg_safe_prob = accelerator.gather(metrics["safe_reward_prob"]).mean().item()

                running_loss += avg_loss / args.gradient_accumulation_steps
                running_acc += avg_acc / args.gradient_accumulation_steps
                epoch_sums["loss"] += avg_loss
                epoch_sums["loss_mod"] += avg_loss_mod
                epoch_sums["loss_pair"] += avg_loss_pair
                epoch_batches += 1

                accelerator.backward(loss)
                if accelerator.sync_gradients and not (args.use_adafactor or args.sdxl):
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log(
                    {
                        "train_loss": running_loss,
                        "loss_mod": avg_loss_mod,
                        "loss_pair": avg_loss_pair,
                        "model_mse": avg_model_mse,
                        "ref_mse": avg_ref_mse,
                        "implicit_acc": running_acc,
                        "safe_reward_prob": avg_safe_prob,
                    },
                    step=global_step,
                )
                running_loss = 0.0
                running_acc = 0.0

                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

            progress_bar.set_postfix(
                step_loss=loss.detach().item(),
                lr=lr_scheduler.get_last_lr()[0],
                implicit_acc=avg_acc,
            )

            if global_step >= args.max_train_steps:
                break

        # Per-epoch aggregates (robust to early break / mid-epoch resume).
        if epoch_batches > 0:
            epoch_avgs = {k: v / epoch_batches for k, v in epoch_sums.items()}
            accelerator.log(
                {
                    "epoch_loss": epoch_avgs["loss"],
                    "epoch_loss_mod": epoch_avgs["loss_mod"],
                    "epoch_loss_pair": epoch_avgs["loss_pair"],
                },
                step=epoch,
            )
            if accelerator.is_main_process:
                with open(os.path.join(args.output_dir, "epoch_losses.txt"), "a") as f:
                    f.write(
                        f"Epoch {epoch:03d}: loss={epoch_avgs['loss']:.6f} "
                        f"loss_mod={epoch_avgs['loss_mod']:.6f} "
                        f"loss_pair={epoch_avgs['loss_pair']:.6f}\n"
                    )

        if global_step >= args.max_train_steps:
            break

    # -- Final save ---------------------------------------------------------------
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.sdxl:
            vae = AutoencoderKL.from_pretrained(
                vae_path,
                subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
                revision=args.revision,
                torch_dtype=weight_dtype,
            )
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=unet,
                vae=vae,
                revision=args.revision,
                torch_dtype=weight_dtype,
            )
        else:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=text_encoder,
                vae=vae,
                unet=unet,
                revision=args.revision,
            )
        pipeline.save_pretrained(args.output_dir)
        logger.info(f"Saved final pipeline to {args.output_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()

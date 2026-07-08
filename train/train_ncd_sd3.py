#!/usr/bin/env python
# coding=utf-8
"""NCD LoRA training for Stable Diffusion 3 (MMDiT + flow matching).

SD3 counterpart of train_ncd.py, sharing the NCD objective and the (caption, A0..A3)
+ scores dataset contract. Only a LoRA adapter on the transformer is trained; the
reference policy is the same transformer with adapters disabled. L_diff is the
sigma-weighted flow-matching loss. See train_ncd.py for the objective.
"""

import argparse
import copy
import logging
import math
import os
import random
import shutil

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL.ImageOps import exif_transpose
from torchvision import transforms
from tqdm.auto import tqdm

import datasets
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from datasets import load_dataset, load_from_disk
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict

import diffusers
from diffusers import StableDiffusion3Pipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_loss_weighting_for_sd3
from diffusers.utils import check_min_version, convert_unet_state_dict_to_peft

check_min_version("0.30.0")

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
    parser = argparse.ArgumentParser(description="NCD LoRA training for Stable Diffusion 3.")

    # Model
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str, required=True,
        help="Path or hub id of the pretrained SD3 model (e.g. stabilityai/stable-diffusion-3-medium-diffusers).",
    )
    parser.add_argument("--revision", type=str, default=None, help="Revision of the pretrained model.")
    parser.add_argument("--variant", type=str, default=None, help="Model files variant, e.g. 'fp16'.")
    parser.add_argument(
        "--lora_rank", type=int, default=8,
        help="Rank of the LoRA adapter on the transformer attention layers — the only trainable "
             "module (paper Appendix A.1 uses rank=8 for SD3/FLUX).",
    )

    # NCD hyperparameters (same semantics as train_ncd.py)
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

    # Flow matching (SD3)
    parser.add_argument(
        "--weighting_scheme", type=str, default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"],
        help="Per-sample sigma-based weighting of the flow-matching loss.",
    )
    parser.add_argument(
        "--precondition_outputs", type=int, default=1,
        help="1: precondition model outputs to predict x0 (target = latents); 0: raw flow target eps - x0.",
    )
    parser.add_argument("--t_min", type=int, default=0, help="Min flow-matching timestep for training.")
    parser.add_argument("--t_max", type=int, default=1000, help="Max flow-matching timestep for training.")
    parser.add_argument(
        "--max_sequence_length", type=int, default=77,
        help="Max T5 sequence length. SD3 inference commonly uses 256; raise it for long prompts "
             "(at the cost of longer MMDiT sequences).",
    )

    # Data
    parser.add_argument(
        "--dataset_name", type=str, required=True,
        help="HF hub id, or a local path saved with `Dataset.save_to_disk` (loaded via load_from_disk). "
             "Must contain columns A0..A3, A0_score..A3_score and a caption column (NCD-10K format).",
    )
    parser.add_argument("--caption_column", type=str, default="caption", help="Caption column name.")
    parser.add_argument(
        "--max_train_samples", type=int, default=None,
        help="Truncate the dataset to this many samples (debugging / quick runs).",
    )
    parser.add_argument("--resolution", type=int, default=512, help="Training resolution (SD3 native is 1024).")
    parser.add_argument("--random_crop", action="store_true", help="Random crop instead of center crop.")
    parser.add_argument("--no_hflip", action="store_true", help="Disable random horizontal flip.")
    parser.add_argument(
        "--proportion_empty_prompts", type=float, default=0.2,
        help="Proportion of prompts replaced by the empty string (classifier-free guidance training).",
    )
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--cache_dir", type=str, default=None)

    # Optimization (paper Appendix A.1: AdamW, bs=2, grad-accum=1, lr=1e-6)
    parser.add_argument("--train_batch_size", type=int, default=2, help="Per-device batch size (in prompts).")
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
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-4, help="Weight decay on LoRA params.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument(
        "--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],
        help="Defaults to the accelerate config of the current system.",
    )

    # Logging / checkpointing
    parser.add_argument("--output_dir", type=str, default="ncd-sd3-lora")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--tracker_project_name", type=str, default="ncd-sd3-training")
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument(
        "--checkpoints_total_limit", type=int, default=None,
        help="Max number of checkpoints to keep; older ones are deleted.",
    )
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None,
        help='Checkpoint path, or "latest" to pick the newest checkpoint in output_dir.',
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if not 0 <= args.t_min < args.t_max <= 1000:
        raise ValueError("Require 0 <= t_min < t_max <= 1000.")
    if not 0.0 <= args.proportion_empty_prompts <= 1.0:
        raise ValueError("--proportion_empty_prompts must be in [0, 1].")

    return args


# ---------------------------------------------------------------------------
# Text-encoder helpers (SD3 triple encoder: CLIP-L + CLIP-G + T5)
# ---------------------------------------------------------------------------
def sample_caption(caption, proportion_empty_prompts):
    """Resolve one caption string, optionally dropping it for CFG training."""
    if random.random() < proportion_empty_prompts:
        return ""
    if isinstance(caption, str):
        return caption
    if isinstance(caption, (list, np.ndarray)):
        return random.choice(list(caption))
    raise ValueError(f"Captions must be strings or lists of strings, got {type(caption)}.")


def _encode_prompt_with_clip(text_encoder, tokenizer, prompts, device):
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    outputs = text_encoder(text_inputs.input_ids.to(device), output_hidden_states=True)
    pooled_prompt_embeds = outputs[0]  # CLIPTextModelWithProjection -> projected pooled embeds
    prompt_embeds = outputs.hidden_states[-2].to(dtype=text_encoder.dtype, device=device)
    return prompt_embeds, pooled_prompt_embeds


def _encode_prompt_with_t5(text_encoder, tokenizer, prompts, max_sequence_length, device):
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    prompt_embeds = text_encoder(text_inputs.input_ids.to(device))[0]
    return prompt_embeds.to(dtype=text_encoder.dtype, device=device)


def encode_prompt_sd3(text_encoders, tokenizers, prompts, max_sequence_length, device):
    """SD3 prompt encoding: 2x CLIP (concat, padded to T5 width) + T5 along the sequence dim."""
    clip_embeds_list, clip_pooled_list = [], []
    for tokenizer, text_encoder in zip(tokenizers[:2], text_encoders[:2]):
        prompt_embeds, pooled = _encode_prompt_with_clip(text_encoder, tokenizer, prompts, device)
        clip_embeds_list.append(prompt_embeds)
        clip_pooled_list.append(pooled)
    clip_prompt_embeds = torch.cat(clip_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_list, dim=-1)

    t5_prompt_embeds = _encode_prompt_with_t5(
        text_encoders[-1], tokenizers[-1], prompts, max_sequence_length, device
    )
    clip_prompt_embeds = F.pad(
        clip_prompt_embeds, (0, t5_prompt_embeds.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embeds], dim=-2)
    return prompt_embeds, pooled_prompt_embeds


# ---------------------------------------------------------------------------
# Data pipeline (same NCD five-tuple contract as train_ncd.py)
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


def build_preprocess_fn(args, image_transforms):
    """Batch transform: stack the N candidates along the channel dim ([N*3, H, W])."""

    def load_image(image):
        image = exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image_transforms(image)

    def preprocess_train(examples):
        per_candidate = [
            [load_image(image) for image in examples[col]] for col in CANDIDATE_COLUMNS
        ]
        # zip over candidates -> one [N*3, H, W] tensor per prompt, order A0, A1, A2, A3
        examples["pixel_values"] = [torch.cat(images, dim=0) for images in zip(*per_candidate)]
        return examples

    return preprocess_train


def build_collate_fn(args):
    def collate_fn(examples):
        pixel_values = torch.stack([e["pixel_values"] for e in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        return {
            "pixel_values": pixel_values,
            # explicit safety scores, [B, N], column order matches CANDIDATE_COLUMNS
            "scores": torch.tensor(
                [[float(e[col]) for col in SCORE_COLUMNS] for e in examples], dtype=torch.float32
            ),
            # raw captions; SD3 encodes prompts at step time
            "caption": [e[args.caption_column] for e in examples],
        }

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
# NCD loss (identical to train_ncd.py, plus optional per-sample weighting)
# ---------------------------------------------------------------------------
def compute_ncd_loss(model_pred, ref_pred, target, scores, args, weighting=None):
    """Compute L_NCD = L_mod + gamma * L_pair.

    Args:
        model_pred / ref_pred / target: [N*B, C, H, W], candidate-major layout
            (first B entries belong to A0, next B to A1, ...).
        scores: [B, N] explicit safety scores, column order = CANDIDATE_COLUMNS.
        weighting: optional [N*B, 1, 1, 1] per-sample loss weighting (SD3 flow
            matching, compute_loss_weighting_for_sd3). None disables weighting.

    Returns:
        (loss, metrics) where metrics is a dict of detached scalars for logging.
    """
    # Per-sample diffusion loss, computed in fp32 for numerical stability. [N*B]
    sq_err_model = (model_pred.float() - target.float()).pow(2)
    sq_err_ref = (ref_pred.float() - target.float()).pow(2)
    if weighting is not None:
        sq_err_model = weighting.float() * sq_err_model
        sq_err_ref = weighting.float() * sq_err_ref
    model_mse = sq_err_model.mean(dim=[1, 2, 3])
    ref_mse = sq_err_ref.mean(dim=[1, 2, 3])

    # Implicit rewards r_i = -(beta/2) * (L_theta - L_ref) = (beta/2) * (L_ref - L_theta).
    # Reshape [N*B] -> [B, N]; column i corresponds to candidate A_i.
    rewards = (0.5 * args.beta_dpo) * (ref_mse - model_mse)
    rewards = rewards.view(NUM_CANDIDATES, -1).t()

    safe_reward = rewards[:, SAFE_INDEX]           # [B]
    harmful_rewards = rewards[:, SAFE_INDEX + 1:]  # [B, N-1]

    # Soft labels w = softmax((scores + safe bias) / alpha).
    biased_scores = scores.clone()
    biased_scores[:, SAFE_INDEX] = biased_scores[:, SAFE_INDEX] + args.safe_score_bias
    soft_labels = (biased_scores / args.temperature_alpha).softmax(dim=-1)  # [B, N]

    # --- L_mod (Eq. 10): reward-weighted attraction + uniform suppression of
    # harmful candidates. The safe candidate is excluded from the suppression
    # term to prevent gradient reversal (Theorem 3.1).
    attraction = -(soft_labels * F.logsigmoid(rewards)).sum(dim=-1)             # [B]
    suppression = -F.logsigmoid(-harmful_rewards).sum(dim=-1) / NUM_CANDIDATES  # [B]
    loss_mod = (attraction + suppression).mean()

    # --- L_pair (Eq. 11): individualized safe-vs-harmful preference margins,
    # averaged over the N-1 harmful candidates (1/(N-1) absorbed into gamma).
    pairwise_logsig = F.logsigmoid(safe_reward.unsqueeze(-1) - harmful_rewards)  # [B, N-1]
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
    # find_unused_parameters: required because the reference forward toggles the
    # LoRA adapter off, leaving adapter params unused in that graph.
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
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

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # -- Load SD3 components -----------------------------------------------------
    # The whole pipeline is loaded once in fp32, then split into components.
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=torch.float32,
    )
    transformer = pipe.transformer
    vae = pipe.vae
    text_encoders = [pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3]
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2, pipe.tokenizer_3]
    noise_scheduler = copy.deepcopy(pipe.scheduler)  # FlowMatchEulerDiscreteScheduler

    # Freeze everything; only the LoRA adapter added below is trainable.
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    for text_encoder in text_encoders:
        text_encoder.requires_grad_(False)

    # VAE stays fp32 for encode stability; frozen inference models use weight_dtype.
    vae.to(accelerator.device, dtype=torch.float32)
    transformer.to(accelerator.device, dtype=weight_dtype)
    for text_encoder in text_encoders:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # -- LoRA adapter: the only trainable parameters. The reference policy is the
    # same transformer with adapters disabled (no second model in memory).
    transformer.add_adapter(
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
    )
    if args.mixed_precision == "fp16":
        # Trainable (LoRA) params must be fp32 under fp16 mixed precision.
        cast_training_params([transformer], dtype=torch.float32)
    lora_parameters = [p for p in transformer.parameters() if p.requires_grad]

    # -- LoRA-aware checkpoint hooks: save_state/load_state handle adapter weights only.
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            lora_layers = None
            for model in models:
                lora_layers = get_peft_model_state_dict(accelerator.unwrap_model(model))
                weights.pop()
            StableDiffusion3Pipeline.save_lora_weights(output_dir, transformer_lora_layers=lora_layers)

    def load_model_hook(models, input_dir):
        transformer_ = models.pop()
        lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(input_dir)
        # Keys are saved with a "transformer." prefix (train_sd3.py filtered on
        # "unet." here, which silently loaded nothing on resume — fixed).
        transformer_state_dict = {
            k.replace("transformer.", ""): v
            for k, v in lora_state_dict.items()
            if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(
            transformer_, transformer_state_dict, adapter_name="default"
        )
        unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
        if unexpected_keys:
            logger.warning(f"Unexpected keys while loading adapter weights: {unexpected_keys}")
        if args.mixed_precision == "fp16":
            cast_training_params([transformer_])

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    optimizer = torch.optim.AdamW(
        lora_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # -- Dataset / dataloader -----------------------------------------------------
    with accelerator.main_process_first():
        dataset = load_ncd_dataset(args)
        if args.max_train_samples is not None:
            dataset = dataset.shuffle(seed=args.seed).select(range(args.max_train_samples))
        image_transforms = build_image_transforms(args)
        train_dataset = dataset.with_transform(build_preprocess_fn(args, image_transforms))

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=build_collate_fn(args),
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    # -- Steps math / lr scheduler ---------------------------------------------
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

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, dict(vars(args)))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running NCD training (SD3, LoRA) *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Per-device batch size (prompts) = {args.train_batch_size}"
                f"  ({args.train_batch_size * NUM_CANDIDATES} latents through the transformer)")
    logger.info(f"  Total train batch size (parallel + accumulation) = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  LoRA rank = {args.lora_rank}, "
                f"trainable params = {sum(p.numel() for p in lora_parameters)}")

    # -- Resume ---------------------------------------------------------------
    global_step = 0
    first_epoch = 0
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

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    def get_flow_sigmas(timesteps, n_dim, dtype):
        """Look up flow-matching sigmas for the given scheduler timesteps."""
        sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def encode_step_prompts(captions):
        """Encode the (shared) captions once; all N candidates reuse the embeddings."""
        prompts = [sample_caption(c, args.proportion_empty_prompts) for c in captions]
        with torch.no_grad():
            prompt_embeds, pooled = encode_prompt_sd3(
                text_encoders, tokenizers, prompts, args.max_sequence_length, accelerator.device
            )
        return prompt_embeds, pooled

    def gather_mean(value):
        """Average a scalar tensor across processes (weighted by per-device batch)."""
        return accelerator.gather(value.repeat(args.train_batch_size)).mean().item()

    def rotate_checkpoints():
        """Keep at most checkpoints_total_limit - 1 checkpoints before saving a new one."""
        if args.checkpoints_total_limit is None:
            return
        checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
        if len(checkpoints) >= args.checkpoints_total_limit:
            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
            for stale in checkpoints[:num_to_remove]:
                logger.info(f"Removing stale checkpoint {stale}")
                shutil.rmtree(os.path.join(args.output_dir, stale))

    # -- Training loop -----------------------------------------------------------
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        running_loss = 0.0
        running_acc = 0.0
        epoch_sums = {"loss": 0.0, "loss_mod": 0.0, "loss_pair": 0.0}
        epoch_batches = 0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                # [B, N*3, H, W] -> [N*B, 3, H, W], candidate-major order A0|A1|A2|A3.
                feed_pixel_values = torch.cat(batch["pixel_values"].chunk(NUM_CANDIDATES, dim=1), dim=0)

                with torch.no_grad():
                    model_input = vae.encode(feed_pixel_values.to(vae.dtype)).latent_dist.sample()
                    model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
                    model_input = model_input.to(dtype=weight_dtype)

                # One (noise, timestep) pair per prompt, shared by all N candidates,
                # so reward differences are driven by image content only.
                prompt_bsz = model_input.shape[0] // NUM_CANDIDATES
                noise = torch.randn_like(model_input[:prompt_bsz]).repeat(NUM_CANDIDATES, 1, 1, 1)

                # Uniform u in [t_min, t_max] / 1000, one draw per prompt.
                r = torch.rand(size=(prompt_bsz,))
                u = (r * args.t_max / 1000 + (1 - r) * args.t_min / 1000).repeat(NUM_CANDIDATES)
                num_train_timesteps = noise_scheduler.config.num_train_timesteps
                indices = (u * num_train_timesteps).long().clamp(0, num_train_timesteps - 1)
                timesteps = noise_scheduler.timesteps[indices].to(device=model_input.device)

                # Flow-matching forward process: x_t = (1 - sigma) * x0 + sigma * eps.
                sigmas = get_flow_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                # All candidates share the caption: encode once, repeat N times.
                prompt_embeds, pooled_prompt_embeds = encode_step_prompts(batch["caption"])
                prompt_embeds = prompt_embeds.repeat(NUM_CANDIDATES, 1, 1)
                pooled_prompt_embeds = pooled_prompt_embeds.repeat(NUM_CANDIDATES, 1)

                # Policy prediction (LoRA enabled).
                model_pred = transformer(
                    hidden_states=noisy_model_input,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]

                # Reference prediction: same transformer with adapters disabled.
                base_transformer = accelerator.unwrap_model(transformer)
                with torch.no_grad():
                    base_transformer.disable_adapters()
                    ref_pred = transformer(
                        hidden_states=noisy_model_input,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        return_dict=False,
                    )[0].detach()
                    base_transformer.enable_adapters()

                # Output preconditioning and flow-matching target.
                if args.precondition_outputs:
                    model_pred = model_pred * (-sigmas) + noisy_model_input
                    ref_pred = ref_pred * (-sigmas) + noisy_model_input
                    target = model_input
                else:
                    target = noise - model_input

                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme, sigmas=sigmas
                )

                loss, metrics = compute_ncd_loss(
                    model_pred,
                    ref_pred,
                    target,
                    batch["scores"].to(accelerator.device),
                    args,
                    weighting=weighting,
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
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_parameters, args.max_grad_norm)
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
                    rotate_checkpoints()
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

        # Per-epoch aggregates (robust to early break).
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

    # -- Final save: transformer LoRA weights only ------------------------------
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = accelerator.unwrap_model(transformer).to(torch.float32)
        StableDiffusion3Pipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=get_peft_model_state_dict(transformer),
        )
        logger.info(f"Saved SD3 transformer LoRA weights to {args.output_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()

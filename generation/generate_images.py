#!/usr/bin/env python
# coding=utf-8
"""Render each prompt under every shared seed, for one (model, weights) config.

Output: <output_root>/<model_name>/<weight_tag>/<benchmark>/{imgs/<case>_<seed>.png, metadata.csv}
"""

import argparse
import json
import os
import re

import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

# Per-family defaults and which sampling path to take.
MODEL_REGISTRY = {
    "sd15": {"resolution": 512, "sampler": "manual"},
    "sd21": {"resolution": 512, "sampler": "manual"},
    "sdxl": {"resolution": 512, "sampler": "pipeline"},
    "sd3": {"resolution": 512, "sampler": "pipeline"},
}
PROMPT_COLUMN_CANDIDATES = ("adv_prompt", "prompt", "caption")  # priority order, as in RECE


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-seed generation for one (model, weights) config.")
    # model
    parser.add_argument("--model_family", type=str, required=True, choices=list(MODEL_REGISTRY),
                        help="Selects the pipeline class, default resolution and sampling path.")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Top-level output label (defaults to --model_family).")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Base pipeline: hub id, or a local pipeline dir (e.g. an NCD output_dir).")
    # reserved backbone-weight interface (comparison baselines)
    parser.add_argument("--unet_ckpt", type=str, default=None,
                        help="Optional edited backbone to swap in: a .pt/.pth/.bin/.ckpt state_dict or a "
                             "diffusers dir. Loaded into pipe.unet (sd*/sdxl) or pipe.transformer (sd3).")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Optional LoRA adapter dir to attach (e.g. NCD-SD3).")
    parser.add_argument("--weight_tag", type=str, default=None,
                        help="Sub-folder label for the loaded weights (default: derived from the "
                             "ckpt/lora source, or 'original' when none is given).")
    # data / seeds
    parser.add_argument("--prompts_csv", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark label for the output path.")
    parser.add_argument("--seeds_file", type=str, default="outputs/seeds.json")
    parser.add_argument("--num_seeds", type=int, default=None, help="Use the first N seeds (default: all).")
    parser.add_argument("--from_case", type=int, default=0, help="Skip prompts with case_number below this.")
    parser.add_argument("--df_start", type=int, default=0, help="Row range of the prompt csv (RECE-style).")
    parser.add_argument("--df_length", type=int, default=None)
    # sampling
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--image_size", type=int, default=None, help="Defaults to the family resolution.")
    parser.add_argument("--seeds_per_batch", type=int, default=10, help="Seeds rendered per forward batch.")
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--device", type=str, default="cuda:0")
    # io
    parser.add_argument("--output_root", type=str, default="outputs/images")
    parser.add_argument("--resume", action="store_true", help="Skip (case, seed) images that already exist.")

    args = parser.parse_args()
    if args.model_name is None:
        args.model_name = args.model_family
    if args.image_size is None:
        args.image_size = MODEL_REGISTRY[args.model_family]["resolution"]
    return args


def resolve_weight_tag(args):
    """Sub-folder name for the loaded weights: explicit, else derived, else 'original'."""
    if args.weight_tag:
        return args.weight_tag
    source = args.unet_ckpt or args.lora_path
    if not source:
        return "original"
    base = os.path.splitext(os.path.basename(os.path.normpath(source)))[0]
    return re.sub(r"[^0-9A-Za-z._-]+", "_", base) or "weights"


def _swap_backbone_weights(pipe, model_family, ckpt, dtype):
    """Swap a baseline's edited backbone (SD3: transformer, else unet) from a state_dict or diffusers dir."""
    is_transformer = model_family == "sd3"
    if os.path.isdir(ckpt):
        if is_transformer:
            from diffusers import SD3Transformer2DModel
            pipe.transformer = SD3Transformer2DModel.from_pretrained(ckpt, torch_dtype=dtype)
        else:
            from diffusers import UNet2DConditionModel
            pipe.unet = UNet2DConditionModel.from_pretrained(ckpt, torch_dtype=dtype)
        print(f"[generate] loaded diffusers backbone from {ckpt}")
    else:
        state = torch.load(ckpt, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        backbone = pipe.transformer if is_transformer else pipe.unet
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        print(f"[generate] swapped backbone weights from {ckpt} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")


def load_pipeline(args, device, dtype):
    # Pipeline classes imported lazily so the SD1.x/2.1 path works without SDXL/SD3 support.
    family = args.model_family
    if family in ("sd15", "sd21"):
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            args.model_path, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
        )
    elif family == "sdxl":
        from diffusers import StableDiffusionXLPipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(args.model_path, torch_dtype=dtype)
    elif family == "sd3":
        from diffusers import StableDiffusion3Pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained(args.model_path, torch_dtype=dtype)
    else:
        raise ValueError(f"Unsupported model_family: {family}")

    if args.unet_ckpt:
        _swap_backbone_weights(pipe, family, args.unet_ckpt, dtype)
    if args.lora_path:
        pipe.load_lora_weights(args.lora_path)
        print(f"[generate] attached LoRA adapter from {args.lora_path}")

    pipe.to(device)
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    return pipe


@torch.no_grad()
def generate_manual(pipe, scheduler, prompt, seeds, args, device, dtype):
    # RECE-style manual CFG loop for SD1.x/2.1 (epsilon prediction). One generator
    # per seed; noise drawn in fp32 so a seed gives identical latents across dtypes.
    tokenizer, text_encoder, unet, vae = pipe.tokenizer, pipe.text_encoder, pipe.unet, pipe.vae
    batch_size = len(seeds)
    height = width = args.image_size

    text_input = tokenizer(
        [prompt] * batch_size, padding="max_length", max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    uncond_input = tokenizer(
        [""] * batch_size, padding="max_length", max_length=text_input.input_ids.shape[-1],
        return_tensors="pt",
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    latent_shape = (1, unet.config.in_channels, height // 8, width // 8)
    latents = torch.cat(
        [
            torch.randn(
                latent_shape,
                generator=torch.Generator(device=device).manual_seed(int(seed)),
                device=device,
                dtype=torch.float32,
            )
            for seed in seeds
        ]
    ).to(dtype)

    scheduler.set_timesteps(args.ddim_steps)  # also resets LMS derivative history
    latents = latents * scheduler.init_noise_sigma

    for t in scheduler.timesteps:
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    latents = latents / vae.config.scaling_factor
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    return [Image.fromarray((img * 255).round().astype("uint8")) for img in image]


@torch.no_grad()
def generate_pipeline(pipe, prompt, seeds, args, device):
    # SDXL/SD3 via the diffusers pipeline (dual/triple encoders + flow matching);
    # one generator per seed.
    generators = [torch.Generator(device=device).manual_seed(int(seed)) for seed in seeds]
    result = pipe(
        prompt=[prompt] * len(seeds),
        num_inference_steps=args.ddim_steps,
        guidance_scale=args.guidance_scale,
        height=args.image_size,
        width=args.image_size,
        generator=generators,
        output_type="pil",
    )
    return result.images


def read_prompts(args):
    df = pd.read_csv(args.prompts_csv)
    df = df[args.df_start:]
    if args.df_length is not None:
        df = df[: args.df_length]

    prompt_col = next((c for c in PROMPT_COLUMN_CANDIDATES if c in df.columns), None)
    if prompt_col is None:
        raise ValueError(f"{args.prompts_csv} has none of the prompt columns {PROMPT_COLUMN_CANDIDATES}")

    rows = []
    for i, row in df.iterrows():
        case_number = int(row["case_number"]) if "case_number" in df.columns else int(i)
        if case_number < args.from_case:
            continue
        rows.append((case_number, str(row[prompt_col])))
    return rows


def main():
    args = parse_args()
    device = args.device
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    sampler = MODEL_REGISTRY[args.model_family]["sampler"]

    # Shared seed list (identical across all methods of one evaluation run).
    with open(args.seeds_file) as f:
        seeds = json.load(f)["seeds"]
    if args.num_seeds is not None:
        seeds = seeds[: args.num_seeds]

    weight_tag = resolve_weight_tag(args)
    out_dir = os.path.join(args.output_root, args.model_name, weight_tag, args.benchmark)
    img_dir = os.path.join(out_dir, "imgs")
    os.makedirs(img_dir, exist_ok=True)
    print(f"[generate] {args.model_name}/{weight_tag}/{args.benchmark} "
          f"(family={args.model_family}, {len(seeds)} seeds) -> {out_dir}")

    pipe = load_pipeline(args, device, dtype)
    scheduler = None
    if sampler == "manual":
        # RECE's sampling setup: LMS scheduler with the SD scaled-linear beta schedule.
        from diffusers import LMSDiscreteScheduler
        scheduler = LMSDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
        )

    prompts = read_prompts(args)
    metadata = []
    for case_number, prompt in tqdm(prompts, desc=f"{args.model_name}/{weight_tag}"):
        pending = []
        for seed in seeds:
            image_path = os.path.join(img_dir, f"{case_number}_{seed}.png")
            metadata.append({
                "case_number": case_number, "prompt": prompt, "seed": seed, "image_path": image_path,
                "model_family": args.model_family, "weight_tag": weight_tag,
            })
            if args.resume and os.path.exists(image_path):
                continue
            pending.append(seed)

        for start in range(0, len(pending), args.seeds_per_batch):
            chunk = pending[start: start + args.seeds_per_batch]
            if sampler == "manual":
                images = generate_manual(pipe, scheduler, prompt, chunk, args, device, dtype)
            else:
                images = generate_pipeline(pipe, prompt, chunk, args, device)
            for seed, image in zip(chunk, images):
                image.save(os.path.join(img_dir, f"{case_number}_{seed}.png"))

    pd.DataFrame(metadata).to_csv(os.path.join(out_dir, "metadata.csv"), index=False)
    print(f"[generate] done: {len(prompts)} prompts x {len(seeds)} seeds -> {out_dir}")


if __name__ == "__main__":
    main()

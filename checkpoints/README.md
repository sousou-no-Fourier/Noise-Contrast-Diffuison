# checkpoints/

Baseline defense weights used for comparison, referenced by `scripts/generate_*.sh`.
Not tracked by git (see `.gitignore`) — download / train them and place here.

Expected files (edit the `METHODS` arrays in the scripts to match your filenames):

| file | method |
|------|--------|
| `esd-u-nudity.pt`  | ESD-u edited UNet |
| `uce-nudity.pt`    | UCE edited UNet   |
| `rece-nudity.pt`   | RECE edited UNet  |

State-dict `.pt` files are swapped into `pipe.unet` (or `pipe.transformer` for SD3);
a diffusers model directory works too. The NCD model itself is a full pipeline /
LoRA produced by `train/`, pointed to via `--model_path` / `--lora_path`, not here.

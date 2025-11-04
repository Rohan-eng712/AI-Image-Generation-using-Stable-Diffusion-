"""
app_text2img.py
text-to-image script using Hugging Face diffusers.
Usage:
  python app_text2img.py --prompt "A cute dog" --outdir outputs --num_inference_steps 25
"""

import argparse
import json
import os
from pathlib import Path

import torch # type: ignore
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler # type: ignore

# load config
cfg = {}
cfg_path = Path("config.json")
if cfg_path.exists():
    cfg = json.load(open(cfg_path))
MODEL_ID = cfg.get("model_id", "runwayml/stable-diffusion-v1-5")
DEVICE = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
PRECISION = cfg.get("precision", "fp16")

def make_pipeline(model_id=MODEL_ID, device=DEVICE):
    print(f"Loading pipeline for {model_id} on {device} (precision={PRECISION})")
    dtype = torch.float16 if PRECISION == "fp16" and device.startswith("cuda") else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=dtype,
    safety_checker=None,  # Disable NSFW filter
    local_files_only=True
    )

    # use a faster scheduler optionally
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    if device.startswith("cuda"):
        pipe.enable_attention_slicing()
    return pipe

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pipe = make_pipeline()
    for i in range(args.num_images):
        print(f"Generating image {i+1}/{args.num_images} ...")
        with torch.autocast(pipe.device.type if hasattr(torch, "autocast") else "cuda"):
            image = pipe(
                args.prompt,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps
            ).images[0]
        out_path = outdir / f"img_{i+1}.png"
        image.save(out_path)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()

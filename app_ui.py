"""
app_ui.py
A simple and safe UI for Stable Diffusion text-to-image generation.
"""

import gradio as gr  # type: ignore
import torch  # type: ignore
import os
from datetime import datetime
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler  # type: ignore

MODEL_ID = "runwayml/stable-diffusion-v1-5"

def load_pipeline():
    print("🔄 Loading model... Please wait.")
    # Force CPU for reliability (avoid black images)
    device = "cpu"

    # Load model in full precision (float32)
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)

    # Use a stable scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    print("✅ Model loaded successfully on", device)
    return pipe


# Load model once at startup
pipe = load_pipeline()

def generate(prompt, steps, guidance, height, width):
    if not prompt or prompt.strip() == "":
        return "Please enter a valid prompt.", None

    try:
        print(f"🎨 Generating: '{prompt}' ...")
        image = pipe(
            prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=height,
            width=width
        ).images[0]

        # ✅ Create 'outputs' folder if not exists
        os.makedirs("outputs", exist_ok=True)

        # ✅ Save image with timestamp
        filename = f"outputs/{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        image.save(filename)
        print(f"💾 Image saved as: {filename}")

        return f"✅ Image generated and saved successfully!\nFile: {filename}", image

    except Exception as e:
        print(f"❌ Generation error: {e}")
        return f"Error: {e}", None


# Build the UI
interface = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="e.g., A futuristic city skyline at sunset"),
        gr.Slider(10, 50, value=25, step=1, label="Inference Steps"),
        gr.Slider(1.0, 10.0, value=7.5, step=0.5, label="Guidance Scale"),
        gr.Slider(256, 768, value=512, step=64, label="Height"),
        gr.Slider(256, 768, value=512, step=64, label="Width")
    ],
    outputs=[
        gr.Textbox(label="Status"),
        gr.Image(label="Generated Image")
    ],
    title="AI Image Generator (Stable Diffusion v1.5)",
    description="Enter a text prompt to generate and save images using Stable Diffusion."
)

if __name__ == "__main__":
    interface.launch()

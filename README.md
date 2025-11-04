# AI-Image-Generation-using-Stable-Diffusion-
This project implements AI-based image generation using the Stable Diffusion v1.5 model. It converts text prompts into realistic images through diffusion-based deep learning. Built with Python, Hugging Face Diffusers, and Gradio, it demonstrates practical application of generative AI research.

### Author: Rohan Mahendra Ravidhone  
**Program:** Bachelor of Engineering in Artificial Intelligence and Data Science  

---

## 📘 Project Overview

This project demonstrates a text-to-image generation system using **Stable Diffusion v1.5**, an advanced deep learning model capable of generating high-quality images from textual descriptions.  

The system converts written prompts into realistic visuals by iteratively refining random noise, showcasing the power of **diffusion-based generative AI**.  

It serves as the implementation component of the research paper **“AI Image Generation using Stable Diffusion.”**

---

## 🎯 Objectives

- Implement a functional AI model that generates creative images based on user text prompts.  
- Provide a user-friendly web interface for experimentation.  
- Demonstrate the working principles of diffusion models in real-world AI applications.  

---

## 🚀 Features

- 🎨 Generate high-quality, realistic images from text prompts.  
- ⚙️ Adjustable generation parameters:
  - Inference steps  
  - Guidance scale  
  - Image dimensions (height & width)
- 💻 Interactive Gradio web interface.  
- 💾 Automatically saves generated images.  
- 🧠 Works on both CPU and GPU.  
- 🌐 Runs locally after the model is downloaded.  

---

## 🛠️ Technologies Used

| Component | Description |
|------------|-------------|
| **Language** | Python 3.11 |
| **Libraries** | Diffusers, Torch, Transformers, Gradio, HuggingFace Hub |
| **Model** | runwayml/stable-diffusion-v1-5 |
| **Scheduler** | DPMSolverMultistepScheduler |

---

## 🧩 System Requirements

- Windows 10/11 (or equivalent OS)  
- Minimum **8 GB RAM**  
- **Python 3.10+**  
- (Optional) NVIDIA GPU with CUDA support  
- Internet connection (only required for the first model download)

---

## ⚙️ Installation Guide

1. **Clone this repository**
   ```bash
   git clone https://github.com/Rohan-eng712/AI-Image-Generation-using-Stable-Diffusion-.git
   cd AI-Image-Generator

# 2. Create & activate virtual env
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Login to Hugging Face for private models
huggingface-cli login

# 5. Run the app (Gradio UI)
python app_ui.py

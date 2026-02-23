## Stable Diffusion – Text-to-Image Generation

This project demonstrates how to generate images from text prompts using the open-source Stable Diffusion v1.5 model.

It follows an industry-style clean implementation using:

PyTorch

Hugging Face Diffusers

Stable Diffusion Pipeline

## What is Stable Diffusion?

Stable Diffusion is an open-source AI model that generates images from text descriptions by gradually removing noise from a random image until it becomes a meaningful picture.

## Why is it called “Diffusion”?

Because it works in two main steps:

1️⃣ Noise is added to images during training

2️⃣ The model learns how to remove noise step-by-step

So it “diffuses” (adds noise) and then “de-diffuses” (removes noise).

This process is called a Diffusion Model.

## Objective

Generate high-quality images from text prompts

Understand diffusion-based generative models

Implement GPU-aware production-style code

Learn prompt engineering basics

## Architecture Overview
Text Prompt
     ↓
CLIP Text Encoder
     ↓
Random Noise Image
     ↓
Iterative Denoising (U-Net + Scheduler)
     ↓
Final Generated Image
## Technologies Used
Tool	Purpose

PyTorch	Tensor computation & GPU acceleration

Diffusers (Hugging Face)	Stable Diffusion pipeline

Stable Diffusion v1.5	Text-to-image model

PIL	Image handling & saving

## Model Used
runwayml/stable-diffusion-v1-5

Open-source

Production-tested

Widely used First run downloads ~4GB model

## Installation

Install dependencies:

pip install torch
pip install diffusers
pip install transformers
pip install accelerate
pip install pillow

▶️ How to Run


1️⃣ Ensure you have GPU (recommended)


2️⃣ Run the Python script

The program will:

Load Stable Diffusion

Process the text prompt

Generate an image

Save it as:

generated_image.png
 GPU Support

The code automatically selects:

device = "cuda" if torch.cuda.is_available() else "cpu"

Uses GPU if available

Falls back to CPU otherwise

 GPU strongly recommended for faster generation.

🎛️ Important Parameters
🔹 guidance_scale

Controls how strongly the image follows the prompt.

Higher → More faithful to text

Lower → More creative

Typical range:

5 – 15

Used in project:

guidance_scale = 7.5
🔹 num_inference_steps

Number of denoising steps.

More steps → Better quality

Fewer steps → Faster generation

Common range:

25 – 50

Used in project:

num_inference_steps = 40
## Prompt Used
A futuristic city at sunset, ultra realistic,
cinematic lighting, high detail, 4k resolution

Prompt engineering significantly affects output quality.

## Project Structure
Stable-Diffusion-Text-to-Image-Generation/
│
├── stable_diffusion.py
├── generated_image.png
└── README.md
🎓 Educational Value

This project helps understand:

Diffusion models

Generative AI

Text-to-image systems

Prompt engineering

GPU acceleration in PyTorch

Production-style AI code

## Possible Improvements

Add negative prompts

Add batch generation

Add image-to-image generation

Add safety filters

Build a Streamlit web app

Add Gradio interface

Deploy on Hugging Face Spaces

## Industry Applications

AI art generation

Game asset creation

Marketing visuals

Content generation

Concept art design

Product mockups

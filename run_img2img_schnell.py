#!/usr/bin/env python

import os
import warnings
import torch
from diffusers import FluxPipeline
from diffusers import FluxImg2ImgPipeline
from diffusers.utils import load_image
import argparse
from PIL import Image
from term_image.image import AutoImage
import io
import sys
import time
import subprocess
import hashlib
import tempfile
import requests
from urllib.parse import urlparse

# Suppress the specific warning about slow tokenizers
warnings.filterwarnings("ignore", message="You set `add_prefix_space`")

def set_tokenizer_parallelism(enable_parallelism):
    os.environ["TOKENIZERS_PARALLELISM"] = "true" if enable_parallelism else "false"

def display_image_in_terminal(image):
    """Display a full image in the terminal."""
    term_image = AutoImage(image, width=80)  # Adjust width as needed
    print(term_image)

def open_image(filename):
    """Open the image file with the default image viewer."""
    if sys.platform.startswith('darwin'):  # macOS
        subprocess.call(('open', filename))
    elif os.name == 'nt':  # Windows
        os.startfile(filename)
    elif os.name == 'posix':  # Linux
        subprocess.call(('xdg-open', filename))

def generate_sha256(image):
    """Generate SHA256 hash for the image."""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    return hashlib.sha256(img_byte_arr.getvalue()).hexdigest()

def generate_images(args):
    # Load the Flux Schnell model
    pipe = FluxImg2ImgPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        revision="741f7c3ce8b383c54771c7003378a50191e9efe9",
        torch_dtype=torch.bfloat16,
    ).to("mps")

    # Load and apply LoRA if specified
    if args.lora_model:
        pipe.load_lora_weights(args.lora_model)
        pipe.fuse_lora(lora_scale=args.lora_scale)

    created_files = []

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the input image
    init_image = load_image(args.input_image).resize((args.width, args.height))

    # Generate multiple images
    for i in range(args.num_images):
        print(f"\nGenerating image {i+1}/{args.num_images}...")

        # Start timing
        start_time = time.time()

        # Generate the image
        out = pipe(
            prompt=args.prompt,
            image=init_image,
            num_inference_steps=args.num_inference_steps,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
        ).images[0]

        # End timing
        end_time = time.time()
        generation_time = end_time - start_time

        # Create the filename
        if args.base_filename:
            filename = f"{args.base_filename}_{i+1}.png"
        else:
            sha256_hash = generate_sha256(out)
            filename = f"{sha256_hash}.png"

        # Create the full path for the output file
        full_path = os.path.join(args.output_dir, filename)

        # Save the generated image with the custom filename
        out.save(full_path)
        print(f"Saved image: {full_path}")
        print(f"Generation time: {generation_time:.2f} seconds")
        created_files.append(full_path)

        # Display the image in the terminal
        print("\nImage preview:")
        display_image_in_terminal(out)

        # Open the image if requested
        if args.view_image:
            open_image(full_path)

        # Wait for user input before continuing to the next image, unless in force mode
        if i < args.num_images - 1 and not args.force:
            input("\nPress Enter to generate the next image...")

    print(f"\n{args.num_images} images have been generated and saved.")
    return created_files

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate images using Flux model (img2img)")
    parser.add_argument("prompt", type=str, help="The prompt for image generation")
    parser.add_argument("-i", "--input_image", type=str, required=True, help="Path to the input image or URL")
    parser.add_argument("-n", "--num_images", type=int, default=1, help="Number of images to generate (default: 1)")
    parser.add_argument("-b", "--base_filename", type=str, default=None, help="Base filename for the generated images (default: None, uses SHA256 hash)")
    parser.add_argument("-o", "--output_dir", type=str, default=".", help="Output directory for generated images (default: current directory)")
    parser.add_argument("-g", "--guidance_scale", type=float, default=0.0, help="Guidance scale (default: 0.0)")
    parser.add_argument("-H", "--height", type=int, default=720, help="Height of the generated image (default: 720)")
    parser.add_argument("-W", "--width", type=int, default=1024, help="Width of the generated image (default: 1024)")
    parser.add_argument("-s", "--num_inference_steps", type=int, default=4, help="Number of inference steps (default: 4)")
    parser.add_argument("--strength", type=float, default=0.95, help="Strength for img2img generation (default: 0.95)")
    parser.add_argument("-v", "--view-image", action="store_true", help="View the image after generation")
    parser.add_argument("--enable_tokenizer_parallelism", action="store_true", help="Enable tokenizer parallelism (default: False)")
    parser.add_argument("-f", "--force", action="store_true", help="Force mode: generate all images without prompting")
    parser.add_argument("--lora_model", type=str, default=None, help="Path to the LoRA model")
    parser.add_argument("--lora_scale", type=float, default=0.5, help="Scale for the LoRA model (default: 0.5)")

    # Parse arguments
    args = parser.parse_args()

    # Set tokenizer parallelism based on the argument
    set_tokenizer_parallelism(args.enable_tokenizer_parallelism)

    # Generate images and get list of created files
    created_files = generate_images(args)

    # Print summary of created files
    print("\nSummary of created files:")
    for file in created_files:
        print(f"- {file}")

    # Print summary of used parameters
    print("\nParameters used:")
    print(f"Prompt: {args.prompt}")
    print(f"Input image: {args.input_image}")
    print(f"Number of images: {args.num_images}")
    print(f"Base filename: {args.base_filename if args.base_filename else 'SHA256 hash'}")
    print(f"Output directory: {args.output_dir}")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Image dimensions: {args.width}x{args.height}")
    print(f"Number of inference steps: {args.num_inference_steps}")
    print(f"Strength: {args.strength}")
    print(f"View image after generation: {args.view_image}")
    print(f"Tokenizer parallelism enabled: {args.enable_tokenizer_parallelism}")
    print(f"Force mode: {args.force}")
    print(f"LoRA model: {args.lora_model if args.lora_model else 'None'}")
    print(f"LoRA scale: {args.lora_scale if args.lora_model else 'N/A'}")

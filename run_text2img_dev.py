#!/usr/bin/env python

import torch
from diffusers import FluxPipeline
import diffusers
import argparse
import os

# Modify the rope function to handle MPS device
_flux_rope = diffusers.models.transformers.transformer_flux.rope
def new_flux_rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0, "The dimension must be even."
    if pos.device.type == "mps":
        return _flux_rope(pos.to("cpu"), dim, theta).to(device=pos.device)
    else:
        return _flux_rope(pos, dim, theta)
diffusers.models.transformers.transformer_flux.rope = new_flux_rope

def generate_images(args):
    # Load the Flux Schnell model
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        revision="0ef5fff789c832c5c7f4e127f94c8b54bbcced44",
        torch_dtype=torch.bfloat16
    ).to("mps")

    created_files = []

    # Generate multiple images
    for i in range(args.num_images):
        # Generate the image
        out = pipe(
            prompt=args.prompt,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            max_sequence_length=args.max_sequence_length,
        ).images[0]

        # Create the full filename
        filename = f"{args.base_filename}_{i+1}.png"

        # Save the generated image with the custom filename
        out.save(filename)
        print(f"Saved image: {filename}")
        created_files.append(filename)

    print(f"\n{args.num_images} images have been generated and saved.")
    return created_files

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate images using Flux model")
    parser.add_argument("prompt", type=str, help="The prompt for image generation")
    parser.add_argument("-n", "--num_images", type=int, default=1, help="Number of images to generate (default: 1)")
    parser.add_argument("-b", "--base_filename", type=str, default="flux_image", help="Base filename for the generated images (default: flux_image)")
    parser.add_argument("-g", "--guidance_scale", type=float, default=3.5, help="Guidance scale (default: 3.5)")
    parser.add_argument("-H", "--height", type=int, default=720, help="Height of the generated image (default: 720)")
    parser.add_argument("-W", "--width", type=int, default=1024, help="Width of the generated image (default: 1024)")
    parser.add_argument("-s", "--num_inference_steps", type=int, default=50, help="Number of inference steps (default: 50)")
    parser.add_argument("-m", "--max_sequence_length", type=int, default=512, help="Max sequence length (default: 512)")

    # Parse arguments
    args = parser.parse_args()

    # Generate images and get list of created files
    created_files = generate_images(args)

    # Print summary of created files
    print("\nSummary of created files:")
    for file in created_files:
        print(f"- {file}")

    # Print summary of used parameters
    print("\nParameters used:")
    print(f"Prompt: {args.prompt}")
    print(f"Number of images: {args.num_images}")
    print(f"Base filename: {args.base_filename}")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Image dimensions: {args.width}x{args.height}")
    print(f"Number of inference steps: {args.num_inference_steps}")
    print(f"Max sequence length: {args.max_sequence_length}")

#!/usr/bin/env python

import argparse
import yaml
import os
from pipelines import (
    SchnellText2ImgPipeline,
    SchnellImg2ImgPipeline,
    DevText2ImgPipeline,
    DevImg2ImgPipeline,
)
from prompt_utils import generate_prompt_variant
import copy
import sys

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()

    # Initial parser to get model and mode
    initial_parser = argparse.ArgumentParser(description="Run Flux image generation", add_help=False)
    initial_parser.add_argument(
        "--mode",
        choices=["text2img", "img2img"],
        default=config['default_mode'],
        help=f"Mode of operation (default: {config['default_mode']})",
    )
    initial_parser.add_argument(
        "--model",
        choices=["schnell", "dev"],
        default=config['default_model'],
        help=f"Model to use (default: {config['default_model']})",
    )
    # Include help for the main parser
    initial_parser.add_argument('-h', '--help', action='store_true', help='Show help message and exit')

    # Parse known arguments to get mode and model
    args, remaining_argv = initial_parser.parse_known_args()

    # If help is requested, print help and exit
    if args.help:
        parser = argparse.ArgumentParser(description="Run Flux image generation")
        # Add all arguments to the parser (see below)
        # ...
        parser.print_help()
        sys.exit(0)

    model_config = config[args.model]

    # Now create the main parser including all arguments
    parser = argparse.ArgumentParser(description="Run Flux image generation")

    # Add initial arguments again to the main parser
    parser.add_argument(
        "--mode",
        choices=["text2img", "img2img"],
        default=args.mode,
        help=f"Mode of operation (default: {config['default_mode']})",
    )
    parser.add_argument(
        "--model",
        choices=["schnell", "dev"],
        default=args.model,
        help=f"Model to use (default: {config['default_model']})",
    )
    parser.add_argument("prompt", type=str, help="The prompt for image generation")

    # Add common arguments
    parser.add_argument(
        "-n",
        "--num_images",
        type=int,
        default=config['common']['num_images'],
        help=f"Number of images to generate (default: {config['common']['num_images']})",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=config['output_dir'],
        help=f"Output directory for generated images (default: {config['output_dir']})",
    )
    parser.add_argument(
        "-b",
        "--base_filename",
        type=str,
        default=None,
        help="Base filename for generated images (default: None, uses SHA256 hash)",
    )
    parser.add_argument(
        "-v",
        "--view-image",
        action="store_true",
        default=config['view_image'],
        help="View the image after generation (default: False)",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=config['force'],
        help="Force mode: generate all images without prompting (default: False)",
    )
    parser.add_argument(
        "--lora_model",
        type=str,
        default=None,
        help="Path to the LoRA model (default: None)",
    )
    parser.add_argument(
        "-i",
        "--input_image",
        type=str,
        help="Path to the input image (required for img2img mode)",
    )
    parser.add_argument(
        "-r",
        "--randomness",
        action="store_true",
        help="Generate random prompt variants for each image",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for image generation (default: 1)",
    )

    # Add model-specific arguments
    parser.add_argument(
        "-g",
        "--guidance_scale",
        type=float,
        default=model_config['guidance_scale'],
        help=f"Guidance scale (default: {model_config['guidance_scale']})",
    )
    parser.add_argument(
        "-H",
        "--height",
        type=int,
        default=model_config['height'],
        help=f"Height of the generated image (default: {model_config['height']})",
    )
    parser.add_argument(
        "-W",
        "--width",
        type=int,
        default=model_config['width'],
        help=f"Width of the generated image (default: {model_config['width']})",
    )
    parser.add_argument(
        "-s",
        "--num_inference_steps",
        type=int,
        default=model_config['num_inference_steps'],
        help=f"Number of inference steps (default: {model_config['num_inference_steps']})",
    )
    parser.add_argument(
        "--lora_scale",
        type=float,
        default=model_config['lora_scale'],
        help=f"Scale for the LoRA model (default: {model_config['lora_scale']})",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=model_config['strength'],
        help=f"Strength for img2img generation (default: {model_config['strength']})",
    )

    # Now parse all arguments
    args = parser.parse_args()

    # Validate input_image for img2img mode
    if args.mode == "img2img" and not args.input_image:
        parser.error("The --input_image argument is required when using img2img mode")

    # Create the appropriate pipeline
    pipeline_class = {
        ("schnell", "text2img"): SchnellText2ImgPipeline,
        ("schnell", "img2img"): SchnellImg2ImgPipeline,
        ("dev", "text2img"): DevText2ImgPipeline,
        ("dev", "img2img"): DevImg2ImgPipeline,
    }[(args.model, args.mode)]

    # Create the pipeline (which loads the model)
    pipeline = pipeline_class(
        config[args.model]['model_id'], config[args.model]['revision']
    )

    # Generate images using the pipeline
    pipeline.generate_images(args, config)

    print(f"\n{args.num_images} images have been generated and saved.")

if __name__ == "__main__":
    main()

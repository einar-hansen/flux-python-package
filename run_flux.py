#!/usr/bin/env python

import argparse
import yaml
import os
from pipelines import SchnellText2ImgPipeline, SchnellImg2ImgPipeline, DevText2ImgPipeline, DevImg2ImgPipeline
from flux_utils import set_tokenizer_parallelism
from typing import List
import random

# Set the TOKENIZERS_PARALLELISM environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def generate_prompt_variant(base_prompt: str, config: dict) -> str:
    """
    Generate a variant of the base prompt.

    :param base_prompt: The original prompt to create a variant from.
    :param config: The configuration dictionary containing style options.
    :return: A prompt variant.
    """
    modifiers = [
        "in the style of {}",
        "with a {} color palette",
        "during {} time of day",
        "in a {} setting",
        "with a {} mood",
    ]

    modifier = random.choice(modifiers)
    if "style" in modifier:
        fill = random.choice(config['styles'])
    elif "color palette" in modifier:
        fill = random.choice(config['color_palettes'])
    elif "time of day" in modifier:
        fill = random.choice(config['times_of_day'])
    elif "setting" in modifier:
        fill = random.choice(config['settings'])
    elif "mood" in modifier:
        fill = random.choice(config['moods'])

    variant = f"{base_prompt}, {modifier.format(fill)}"

    return variant

def main():
    config = load_config()

    parser = argparse.ArgumentParser(description="Run Flux image generation")
    parser.add_argument("--mode", choices=["text2img", "img2img"],
                        default=config['default_mode'],
                        help=f"Mode of operation (default: {config['default_mode']})")
    parser.add_argument("--model", choices=["schnell", "dev"],
                        default=config['default_model'],
                        help=f"Model to use (default: {config['default_model']})")
    parser.add_argument("prompt", type=str, help="The prompt for image generation")
    parser.add_argument("-n", "--num_images", type=int, default=config['common']['num_images'],
                        help=f"Number of images to generate (default: {config['common']['num_images']})")
    parser.add_argument("-o", "--output_dir", type=str, default=config['output_dir'],
                        help=f"Output directory for generated images (default: {config['output_dir']})")
    parser.add_argument("-b", "--base_filename", type=str, default=None,
                        help="Base filename for generated images (default: None, uses SHA256 hash)")
    parser.add_argument("-v", "--view-image", action="store_true", default=config['view_image'],
                        help="View the image after generation (default: False)")
    parser.add_argument("-f", "--force", action="store_true", default=config['force'],
                        help="Force mode: generate all images without prompting (default: False)")
    parser.add_argument("--lora_model", type=str, default=None,
                        help="Path to the LoRA model (default: None)")
    parser.add_argument("-i", "--input_image", type=str,
                        help="Path to the input image (required for img2img mode)")
    parser.add_argument("-r", "--randomness", action="store_true",
                        help="Generate random prompt variants for each image")

    args, unknown = parser.parse_known_args()

    # Determine which model config to use
    model_config = config[args.model]

    # Add model-specific arguments
    parser.add_argument("-g", "--guidance_scale", type=float, default=model_config['guidance_scale'],
                        help=f"Guidance scale (default: {model_config['guidance_scale']})")
    parser.add_argument("-H", "--height", type=int, default=model_config['height'],
                        help=f"Height of the generated image (default: {model_config['height']})")
    parser.add_argument("-W", "--width", type=int, default=model_config['width'],
                        help=f"Width of the generated image (default: {model_config['width']})")
    parser.add_argument("-s", "--num_inference_steps", type=int, default=model_config['num_inference_steps'],
                        help=f"Number of inference steps (default: {model_config['num_inference_steps']})")
    parser.add_argument("--lora_scale", type=float, default=model_config['lora_scale'],
                        help=f"Scale for the LoRA model (default: {model_config['lora_scale']})")
    parser.add_argument("--strength", type=float, default=model_config['strength'],
                        help=f"Strength for img2img generation (default: {model_config['strength']})")

    args = parser.parse_args()

    # Validate input_image for img2img mode
    if args.mode == "img2img" and not args.input_image:
        parser.error("The --input_image argument is required when using img2img mode")

    # Create the appropriate pipeline
    pipeline_class = {
        ("schnell", "text2img"): SchnellText2ImgPipeline,
        ("schnell", "img2img"): SchnellImg2ImgPipeline,
        ("dev", "text2img"): DevText2ImgPipeline,
        ("dev", "img2img"): DevImg2ImgPipeline
    }[(args.model, args.mode)]

    pipeline = pipeline_class(model_config['model_id'], model_config['revision'])

    for i in range(args.num_images):
        if args.randomness:
            variant_prompt = generate_prompt_variant(args.prompt, config['prompt_variants'])
            print(f"\nGenerating image {i+1}/{args.num_images}")
            print(f"Final prompt: {variant_prompt}")
            args.prompt = variant_prompt
        else:
            print(f"\nGenerating image {i+1}/{args.num_images}")

        pipeline.generate_images(args)

        if i < args.num_images - 1 and not args.force:
            input("\nPress Enter to generate the next image...")

if __name__ == "__main__":
    main()

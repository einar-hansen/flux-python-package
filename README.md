# How to

source flux_env/bin/activate

# Flux Image Generation

This project provides a flexible framework for generating images using the Flux model from Black Forest Labs. It supports both text-to-image and image-to-image generation using either the Schnell or Dev model variants.

## Setup

1. Clone this repository
2. Create and activate a virtual environment:
   ```
   python3 -m venv flux_env
   source flux_env/bin/activate
   ```
3. Install the required dependencies:
   ```
   pip install git+https://github.com/huggingface/diffusers.git
   pip install transformers torch pillow pyyaml term-image
   ```

## Using the Virtual Environment

Always activate the virtual environment before running the script or installing new dependencies:

```
source flux_env/bin/activate
```

To deactivate the virtual environment when you're done:

```
deactivate
```

## Usage

Run the main script `run_flux.py` with the following syntax:

```
python run_flux.py [options] <prompt>
```

### Options

- `--mode {text2img,img2img}`: Mode of operation (default: text2img)
- `--model {schnell,dev}`: Model to use (default: schnell)
- `-n, --num_images`: Number of images to generate (default in config.yaml)
- `-o, --output_dir`: Output directory for generated images (default in config.yaml)
- `-b, --base_filename`: Base filename for generated images (default: None, uses SHA256 hash)
- `-g, --guidance_scale`: Guidance scale (default in config.yaml)
- `-H, --height`: Height of the generated image (default in config.yaml)
- `-W, --width`: Width of the generated image (default in config.yaml)
- `-s, --num_inference_steps`: Number of inference steps (default in config.yaml)
- `-v, --view-image`: View the image after generation (default: False)
- `-f, --force`: Force mode: generate all images without prompting (default: False)
- `--lora_model`: Path to the LoRA model (default: None)
- `--lora_scale`: Scale for the LoRA model (default in config.yaml)
- `-i, --input_image`: Path to the input image (required for img2img mode)
- `--strength`: Strength for img2img generation (default in config.yaml)

## Configuration

Default settings can be adjusted in the `config.yaml` file. There are separate configurations for Schnell and Dev models.

## Examples

1. Generate a text-to-image using Schnell model:
   ```
   python run_flux.py --mode text2img --model schnell "A beautiful landscape with mountains and a lake" -n 3
   ```

2. Generate an image-to-image using Dev model:
   ```
   python run_flux.py --mode img2img --model dev "A futuristic cityscape" -i input_image.jpg --strength 0.75
   ```

3. Use a LoRA model:
   ```
   python run_flux.py --mode text2img --model schnell "A portrait in the style of Van Gogh" --lora_model path/to/lora_model.safetensors
   ```

## Project Structure

- `run_flux.py`: Main script to run the image generation
- `config.yaml`: Configuration file with default settings
- `flux_utils.py`: Utility functions
- `pipelines/`: Directory containing the pipeline implementations
  - `base_pipeline.py`: Base class for all pipelines
  - `schnell_text2img.py`: Schnell text-to-image pipeline
  - `schnell_img2img.py`: Schnell image-to-image pipeline
  - `dev_text2img.py`: Dev text-to-image pipeline
  - `dev_img2img.py`: Dev image-to-image pipeline

## Performance Considerations

- The generation time can vary based on the complexity of the prompt, image size, number of inference steps, and available hardware.
- To potentially speed up generation:
  1. Reduce the number of inference steps (`-s` argument)
  2. Generate smaller images (adjust `-H` and `-W` arguments)
  3. Ensure you're using a GPU if available
  4. Close other resource-intensive applications

Note: There's often a trade-off between generation speed and image quality.

## Random Prompt Variants

You can generate unique random prompt variants for each image using the `-r` or `--randomness` flag. This feature creates a slightly modified version of your base prompt for each image, adding variety to your generations.

Usage:
```python
python run_flux.py --mode text2img --model schnell "Your base prompt" -n <number_of_images> -r
```

When this mode is enabled, the script will print the final prompt used for each image generation.

You can customize the available options for prompt variants by editing the `prompt_variants` section in the `config.yaml` file. This allows you to add, remove, or modify the styles, color palettes, times of day, settings, and moods used in generating variants.

## Logging

The script generates a log file named `generation_log.csv` in the same directory as the script. This CSV file uses semicolons (;) as delimiters and contains the following information for each generated image:

- Id: SHA256 hash of the generated image
- Timestamp: Date and time of generation
- Prompt: The prompt used to generate the image
- OutputFile: Full path to the saved image file
- ExecutionTime: Time taken to generate the image (in seconds)
- ModelName: Name of the model used (e.g., "black-forest-labs/FLUX.1-schnell")
- ModelType: Type of generation (text2img or img2img)
- InputFile: Full path to the input image file (only for img2img operations)

Note: When opening the log file in a spreadsheet application, make sure to specify the semicolon as the delimiter to properly separate the fields.


## Tokenizer Parallelism

The script sets `TOKENIZERS_PARALLELISM` to `false` to avoid potential deadlocks. If you encounter issues related to tokenizer parallelism, you can manually set this environment variable:

```
export TOKENIZERS_PARALLELISM=false
```

## Contributing

Feel free to submit issues or pull requests if you have suggestions for improvements or encounter any problems.

## License

This project is open-source and available under the MIT License.

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
- `-r, --randomness`: Generate random prompt variants for each image

## Examples

### 1. Basic Text-to-Image Generation

Generate a single image from a text prompt:

```bash

python run_flux.py --model schnell --mode text2img "A cyberpunk cityscape"
```

![A cyberpunk cityscape](images/598dbe74ae3a155205f6ded64f6ea4e782272bd4c5debc5c40c570154fc8c80f.png)

### 2. Multiple Images with Random Variants

Generate multiple images with random prompt variants:

```bash
python run_flux.py --model schnell --mode text2img "a developer that sits in the office working on a apple mac, very concentrated, can partially see the code on the screen, the office is professional and has a few green plants, scandinavian style." -n 3 -r
```

This appends f.example ", during golden hour time of day", ", in the style of surrealism" or ", in a fantasy setting". You can control what kind of styles you randomly want to apply in the `config.yaml` file.

![Image1](images/7a87dd4428ddb913b1e0ab76f0fedaa3196743cd162c29a80d7ef73c4f8d90a4.png)
![Image1](images/15b406f271f00c7688d037ec7431a82af403b8df57feefd2411143bb186bf2b3.png)
![Image1](images/55fe77a4e163b05d26ea72484105c23ab38283cfc305bd9cafbc0dc423288e39.png)

### 3. Image-to-Image Generation

Transform an existing image based on a prompt:

```bash
python run_flux.py --model dev --mode img2img --strength 0.85 -i images/cityscape.png "Turn the landscape into a winter wonderland"
```

![A wonderland cyberpunk cityscape](images/598dbe74ae3a155205f6ded64f6ea4e782272bd4c5debc5c40c570154fc8c80f.png)

Click on the links to learn more about how to use the [strength](https://huggingface.co/docs/diffusers/using-diffusers/img2img#strength) and [guidance_scale](https://huggingface.co/docs/diffusers/using-diffusers/img2img#guidance-scale) arguments.

### 4. Customizing Image Size

Generate a larger image with custom dimensions:

```bash
python run_flux.py --model schnell --mode text2img "An intricate mandala design" -H 1024 -W 1024
```

[Insert Image Here: mandala_large.png]

### 5. Adjusting Generation Parameters

Fine-tune the image generation process:

```bash
python run_flux.py --model dev --mode text2img "A futuristic space station" -g 7.5 -s 50
```

[Insert Image Here: space_station.png]

## Configuration

Default settings can be adjusted in the `config.yaml` file. There are separate configurations for Schnell and Dev models, as well as options for prompt variants.

## Logging

The script generates a log file named `generation_log.csv` in the same directory as the script. This CSV file uses semicolons (;) as delimiters and contains information about each generated image.

## Performance Considerations

- The initial model loading may take some time, but subsequent image generations will be faster.
- Adjust the `num_inference_steps` parameter to balance between generation speed and image quality.
- Using a GPU can significantly speed up the image generation process.

## Contributing

Feel free to submit issues or pull requests if you have suggestions for improvements or encounter any problems.

## License

This project is open-source and available under the MIT License.

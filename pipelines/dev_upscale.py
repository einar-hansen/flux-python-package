import time
import os
from .base_pipeline import BasePipeline
from PIL import Image
from diffusers.utils import load_image
from diffusers import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline
import torch

class DevUpscalePipeline(BasePipeline):
    def __init__(self, model_id, revision):
        super().__init__(model_id, revision, "upscale")

    def load_model(self):
        print(f"Loading {self.model_type} model...")
        controlnet = FluxControlNetModel.from_pretrained(
          "jasperai/Flux.1-dev-Controlnet-Upscaler",
          torch_dtype=torch.bfloat16
        )
        self.pipe = FluxControlNetPipeline.from_pretrained(
          "black-forest-labs/FLUX.1-dev",
          controlnet=controlnet,
          torch_dtype=torch.bfloat16
        ).to("mps")
        print("Model loaded successfully.")

    def generate_images(self, args, config):
        """
        Upscales images using the Dev upscaler pipeline.

        Args:
            args: Command-line arguments containing input_image, etc.
            config: Configuration dictionary.

        Returns:
            List of file paths to the upscaled images.
        """
        created_files = []
        os.makedirs(args.output_dir, exist_ok=True)

        # Load a control image
        control_image = load_image(args.input_image)

        batch_size = args.batch_size if hasattr(args, 'batch_size') else 1
        w, h = control_image.size
        control_image = control_image.resize((args.width, args.height))

        # Since upscaling is typically done one image at a time, we'll process accordingly
        for i in range(args.num_images):
            print(f"\nUpscaling image {i + 1}/{args.num_images}...")

            start_time = time.time()

            prompt = args.prompt
            if args.randomness:
                prompt = generate_prompt_variant(prompt, config['prompt_variants'])

            # Upscale the image
            image = self.pipe(
                prompt=prompt,
                control_image=control_image,
                controlnet_conditioning_scale=0.6,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                height=control_image.size[1],
                width=control_image.size[0]
            ).images[0]

            end_time = time.time()
            execution_time = end_time - start_time

            # Save and display the upscaled image
            full_path = self.save_and_display_image(
                image, args, i, execution_time, prompt
            )
            created_files.append(full_path)

            print(f"Upscaling time: {execution_time:.2f} seconds")

        print(f"\n{args.num_images} images have been upscaled and saved.")
        return created_files

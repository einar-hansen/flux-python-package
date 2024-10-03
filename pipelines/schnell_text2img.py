import time
import os
import torch
from .base_pipeline import BasePipeline
from diffusers import FluxPipeline

class SchnellText2ImgPipeline(BasePipeline):
    def __init__(self, model_id, revision):
        super().__init__(model_id, revision, "text2img")

    def load_model(self):
        super().load_model(FluxPipeline)

    def generate_images(self, args):
        self.load_model()

        if args.lora_model:
            self.pipe.load_lora_weights(args.lora_model)
            self.pipe.fuse_lora(lora_scale=args.lora_scale)

        created_files = []

        os.makedirs(args.output_dir, exist_ok=True)

        for i in range(args.num_images):
            print(f"\nGenerating image {i+1}/{args.num_images}...")

            start_time = time.time()

            image = self.pipe(
                prompt=args.prompt,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
            ).images[0]

            end_time = time.time()
            execution_time = end_time - start_time

            full_path = self.save_and_display_image(image, args, i, execution_time)
            created_files.append(full_path)

            print(f"Generation time: {execution_time:.2f} seconds")

            if i < args.num_images - 1 and not args.force:
                input("\nPress Enter to generate the next image...")

        print(f"\n{args.num_images} images have been generated and saved.")
        return created_files

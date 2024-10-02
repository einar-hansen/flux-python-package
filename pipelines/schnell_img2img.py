import time
import os
import torch
from .base_pipeline import BasePipeline
from diffusers import FluxImg2ImgPipeline
from diffusers.utils import load_image

class SchnellImg2ImgPipeline(BasePipeline):
    def __init__(self, model_id, revision):
        super().__init__(model_id, revision)

    def load_model(self):
        self.pipe = FluxImg2ImgPipeline.from_pretrained(
            self.model_id,
            revision=self.revision,
            torch_dtype=torch.bfloat16,
        ).to("mps")

    def generate_images(self, args):
        self.load_model()

        if args.lora_model:
            self.pipe.load_lora_weights(args.lora_model)
            self.pipe.fuse_lora(lora_scale=args.lora_scale)

        created_files = []

        os.makedirs(args.output_dir, exist_ok=True)

        init_image = load_image(args.input_image).resize((args.width, args.height))

        for i in range(args.num_images):
            print(f"\nGenerating image {i+1}/{args.num_images}...")

            start_time = time.time()

            image = self.pipe(
                prompt=args.prompt,
                image=init_image,
                strength=args.strength,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
            ).images[0]

            end_time = time.time()
            generation_time = end_time - start_time

            full_path = self.save_and_display_image(image, args, i)
            created_files.append(full_path)

            print(f"Generation time: {generation_time:.2f} seconds")

            if i < args.num_images - 1 and not args.force:
                input("\nPress Enter to generate the next image...")

        print(f"\n{args.num_images} images have been generated and saved.")
        return created_files

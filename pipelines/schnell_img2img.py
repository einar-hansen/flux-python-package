import time
import os
from .base_pipeline import BasePipeline
from diffusers.utils import load_image

class SchnellImg2ImgPipeline(BasePipeline):
    def __init__(self, model_id, revision):
        super().__init__(model_id, revision, "img2img")

    def generate_images(self, args):
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
            execution_time = end_time - start_time

            full_path = self.save_and_display_image(image, args, i, execution_time)
            created_files.append(full_path)

            print(f"Generation time: {execution_time:.2f} seconds")

        print(f"\n{args.num_images} images have been generated and saved.")
        return created_files

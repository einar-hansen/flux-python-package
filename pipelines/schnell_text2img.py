import time
import os
from .base_pipeline import BasePipeline

class SchnellText2ImgPipeline(BasePipeline):
    def __init__(self, model_id, revision):
        super().__init__(model_id, revision, "text2img")

    def generate_images(self, args):
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

        print(f"\n{args.num_images} images have been generated and saved.")
        return created_files

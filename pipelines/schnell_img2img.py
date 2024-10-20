import time
import os
from .base_pipeline import BasePipeline
from diffusers.utils import load_image
from prompt_utils import generate_prompt_variant

class SchnellImg2ImgPipeline(BasePipeline):
    def __init__(self, model_id, revision):
        super().__init__(model_id, revision, "img2img")

    def generate_images(self, args, config):
        """
        Generates images using the Schnell img2img pipeline.

        Args:
            args: Command-line arguments containing prompt, num_images, etc.
            config: Configuration dictionary containing prompt variants.

        Returns:
            List of file paths to the generated images.
        """
        created_files = []
        os.makedirs(args.output_dir, exist_ok=True)

        # Load and resize the initial image once
        init_image = load_image(args.input_image).convert("RGB").resize(
            (args.width, args.height)
        )

        batch_size = args.batch_size if hasattr(args, 'batch_size') else 1
        num_batches = (args.num_images + batch_size - 1) // batch_size

        for batch_num in range(num_batches):
            print(f"\nGenerating batch {batch_num + 1}/{num_batches}...")
            batch_start = batch_num * batch_size
            batch_end = min(batch_start + batch_size, args.num_images)
            actual_batch_size = batch_end - batch_start

            start_time = time.time()

            prompts = []
            for _ in range(actual_batch_size):
                if args.randomness:
                    unique_prompt = generate_prompt_variant(
                        args.prompt, config['prompt_variants']
                    )
                    prompts.append(unique_prompt)
                else:
                    prompts.append(args.prompt)

            init_images = [init_image] * actual_batch_size

            # Generate images using the pipeline
            images = self.pipe(
                prompt=prompts,
                image=init_images,
                strength=args.strength,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
            ).images

            end_time = time.time()
            execution_time = end_time - start_time

            # Save and display each image in the batch
            for i, image in enumerate(images):
                index = batch_start + i
                prompt_used = prompts[i]
                full_path = self.save_and_display_image(
                    image, args, index, execution_time, prompt_used
                )
                created_files.append(full_path)

            print(f"Batch generation time: {execution_time:.2f} seconds")

        print(f"\n{args.num_images} images have been generated and saved.")
        return created_files

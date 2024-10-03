from abc import ABC, abstractmethod
import os
import torch
import csv
from datetime import datetime
from diffusers import FluxPipeline, FluxImg2ImgPipeline
from flux_utils import display_image_in_terminal, open_image, generate_sha256

class BasePipeline(ABC):
    def __init__(self, model_id, revision, model_type):
        self.model_id = model_id
        self.revision = revision
        self.model_type = model_type
        self.pipe = None
        self.log_file = "generation_log.csv"

    def load_model(self, pipeline_class):
        self.pipe = pipeline_class.from_pretrained(
            self.model_id,
            revision=self.revision,
            torch_dtype=torch.bfloat16,
        ).to("mps")

    @abstractmethod
    def generate_images(self, args):
        pass

    def save_and_display_image(self, image, args, index, execution_time):
        sha256_hash = generate_sha256(image)
        if hasattr(args, 'base_filename') and args.base_filename:
            filename = f"{args.base_filename}_{index+1}.png"
        else:
            filename = f"{sha256_hash}.png"

        full_path = os.path.join(args.output_dir, filename)
        image.save(full_path)
        print(f"Saved image: {full_path}")

        print("\nImage preview:")
        display_image_in_terminal(image)

        if args.view_image:
            open_image(full_path)

        # Log the generation
        self.log_generation(sha256_hash, args.prompt, full_path, execution_time, args)

        return full_path

    def log_generation(self, file_hash, prompt, output_file, execution_time, args):
        timestamp = datetime.now().isoformat()
        log_entry = [
            file_hash,
            timestamp,
            prompt,
            output_file,
            f"{execution_time:.2f}",
            self.model_id,
            self.model_type
        ]

        if self.model_type == "img2img":
            log_entry.append(args.input_image)

        file_exists = os.path.isfile(self.log_file)

        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f, delimiter=';')  # Changed delimiter to semicolon
            if not file_exists:
                header = ['Id', 'Timestamp', 'Prompt', 'OutputFile', 'ExecutionTime', 'ModelName', 'ModelType']
                if self.model_type == "img2img":
                    header.append('InputFile')
                writer.writerow(header)
            writer.writerow(log_entry)

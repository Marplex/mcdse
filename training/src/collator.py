import logging
from typing import List, Tuple
from dataclasses import dataclass
from transformers import ProcessorMixin
from arguments import DataArguments
from PIL import Image
import PIL

@dataclass
class TrainCollator:
    data_args: DataArguments
    processor: ProcessorMixin

    def build_query_prompt(self, query: str):
        return f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Query: {query}<|im_end|>\n<|endoftext|>"

    def build_passage_prompt(self, passage: str):
        return f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Document: {passage}<|im_end|>\n<|endoftext|>"
    
    def build_image_prompt(self):
        return f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What is shown in this image?<|im_end|>\n<|endoftext|>"

    def __call__(self, features: List[Tuple[str, List[PIL.Image.Image | str]]]):
        query_prompts = [self.build_query_prompt(f[0]) for f in features]
        query_images = [Image.new('RGB', (56, 56)) for _ in features]
        
        passage_prompts = []
        passage_images = []

        for f in features:
            if type(f[1][0]) == str:
                passage_prompts.append(self.build_passage_prompt(f[1][0]))
                passage_images.append(Image.new('RGB', (56, 56)))
            else:
                passage_prompts.append(self.build_image_prompt())
                passage_images.append(f[1][0])
        
        passage_inputs = self.processor(text=passage_prompts, images=passage_images, return_tensors="pt", padding="longest")
        query_inputs = self.processor(text=query_prompts, images=query_images, return_tensors="pt", padding="longest")

        return query_inputs, passage_inputs
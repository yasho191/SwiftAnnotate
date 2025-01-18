from openai import OpenAI
import transformers
import torch
import logging
from PIL import Image
from typing import List
import base64

class ImageCaptioningBase:
    
    def caption(self, image: Image) -> str:
        raise NotImplementedError("caption method must be implemented in every subclass")
    
    def validate(self, image: Image, caption: str) -> bool:
        raise NotImplementedError("validate method must be implemented in every subclass")
    
    def generate(self, image_paths: List[str]) -> List[str, float]:
        raise NotImplementedError("generate method must be implemented in every subclass")
    

def ImageCaptionOpenAI(ImageCaptioningBase):
    def __init__(
        self, 
        model: str, 
        api_key: str, 
        caption_prompt: str, 
        max_retry: int = 3, 
        output_file: str = "output.json"
    ):
        self.model = model
        self.client = OpenAI(api_key)
        self.caption_prompt = caption_prompt
        self.max_retry = max_retry
        
    def encode_image(self, image_path: str) -> str:
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logging.error(f"Image encoding failed: {e}")
        
    def caption(self, image: Image, **kwargs) -> str:
        return 
    
    def validate(self, image: Image, caption: str, threshold: float, **kwargs) -> bool:
        return 
    
    def generate(self, image_paths: List[str]) -> List[str, float]:
        return 
        
    
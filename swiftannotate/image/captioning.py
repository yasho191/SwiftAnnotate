from openai import OpenAI
import transformers
import torch
import logging
from PIL import Image
from typing import List, Tuple
import base64

class BaseImageCaptioning:
    """
    Base class for image captioning. The class provides a blueprint for all image captioning models.
    Each base class must implement the following methods:
    - caption: generates a caption for an image
    - validate: validates a caption for an image
    - generate: generates captions for a list of images
    """
    
    def caption(self, image: Image, **kwargs) -> str:
        """
        Generates a caption for an image. Implements the logic to generate a caption for an image.

        Args:
            image (PIL.Image): Image to generate a caption for.
            **kwargs: Additional arguments to pass to the method for API calls.

        Raises:
            NotImplementedError: Must be implemented in every subclass.

        Returns:
            str: Caption of the image.
        """
        raise NotImplementedError("caption method must be implemented in every subclass")
    
    def validate(self, image: Image, caption: str, **kwargs) -> Tuple[str, float]:
        """
        Validates a caption for an image. Implements the logic to validate a caption for an image.

        Args:
            image (Image): Image to validate the caption for.
            caption (str): Generated caption for the image.
            **kwargs: Additional arguments to pass to the method for API calls.

        Raises:
            NotImplementedError: Must be implemented in every subclass.

        Returns:
            Tuple[str, float]: Returns validation logic and confidence score ranging 0-1.
        """
        raise NotImplementedError("validate method must be implemented in every subclass")
    
    def generate(self, image_paths: List[str], **kwargs) -> List[Tuple[str, str, float]]:
        """
        Generates captions for a list of images. Implements the logic to generate captions for a list of images.

        Args:
            image_paths (List[str]): List of image paths to generate captions for.
            **kwargs: Additional arguments to pass to the method for custom pipeline interactions.

        Raises:
            NotImplementedError: Must be implemented in every subclass.

        Returns:
            List[Tuple[str, str, float]]: List of captions, validation reasoning and confidence scores for each image.
        """
        raise NotImplementedError("generate method must be implemented in every subclass")
    
class ImageCaptionOpenAI(BaseImageCaptioning):
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
    
    def validate(self, image: Image, caption: str, threshold: float, **kwargs) -> Tuple[str, float]:
        return 
    
    def generate(self, image_paths: List[str]) -> List[Tuple[str, str, float]]:
        return 
        
class ImageCaptionHuggingFace(BaseImageCaptioning):
    def __init__(
        self, 
        model: str, 
        tokenizer: str, 
        max_retry: int = 3, 
        output_file: str = "output.json"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_retry = max_retry
        
    def caption(self, image: Image, **kwargs) -> str:
        return 
    
    def validate(self, image: Image, caption: str, threshold: float, **kwargs) -> Tuple[str, float]:
        return 
    
    def generate(self, image_paths: List[str]) ->List[Tuple[str, str, float]]:
        return
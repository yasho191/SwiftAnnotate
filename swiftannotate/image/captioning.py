from openai import OpenAI
import transformers
import torch
from tqdm import tqdm
import logging
import json
from PIL import Image
from typing import List, Tuple, Dict
import base64
from pydantic import BaseModel
from swiftannotate.constants import BASE_IMAGE_CAPTION_VALIDATION_PROMPT, BASE_IMAGE_CAPTION_PROMPT

class ImageValidationOutput(BaseModel):
    valiadtion_reasoning: str
    confidence: float

class BaseImageCaptioning:
    """
    Base class for image captioning. The class provides a blueprint for all image captioning models.
    Each base class must implement the following methods:
    - caption: generates a caption for an image
    - validate: validates a caption for an image
    - generate: generates captions for a list of images
    """
    
    def caption(self, image: Image.Image | str, **kwargs) -> str:
        """
        Generates a caption for an image. Implements the logic to generate a caption for an image.

        Args:
            image (PIL.Image.Image, str): Image to validate the caption for. Can be a PIL Image or base64 encoded image depending on the model.
            **kwargs: Additional arguments to pass to the method for API calls.

        Raises:
            NotImplementedError: Must be implemented in every subclass.

        Returns:
            str: Caption of the image.
        """
        raise NotImplementedError("caption method must be implemented in every subclass")
    
    def validate(self, image: Image.Image | str, caption: str, **kwargs) -> Tuple[str, float]:
        """
        Validates a caption for an image. Implements the logic to validate a caption for an image.

        Args:
            image (PIL.Image.Image, str): Image to validate the caption for. Can be a PIL Image or base64 encoded image depending on the model.
            caption (str): Generated caption for the image.
            **kwargs: Additional arguments to pass to the method for API calls.

        Raises:
            NotImplementedError: Must be implemented in every subclass.

        Returns:
            Tuple[str, float]: Returns validation logic and confidence score ranging 0-1.
        """
        raise NotImplementedError("validate method must be implemented in every subclass")
    
    def generate(self, image_paths: List[str], **kwargs) -> List[Dict]:
        """
        Generates captions for a list of images. Implements the logic to generate captions for a list of images.

        Args:
            image_paths (List[str]): List of image paths to generate captions for.
            **kwargs: Additional arguments to pass to the method for custom pipeline interactions.

        Raises:
            NotImplementedError: Must be implemented in every subclass.

        Returns:
            List[Dict]: List of captions, validation reasoning and confidence scores for each image.
        """
        raise NotImplementedError("generate method must be implemented in every subclass")
    
class ImageCaptionOpenAI(BaseImageCaptioning):
    def __init__(
        self, 
        model: str, 
        api_key: str, 
        caption_prompt: str | None = None, 
        validation: bool = True,
        validation_prompt: str | None = None,
        validation_threshold: float = 0.5,
        max_retry: int = 3, 
        output_file: str | None = None,
        **kwargs
    ):
        self.model = model
        self.client = OpenAI(api_key)
        
        if caption_prompt is None:
            self.caption_prompt = BASE_IMAGE_CAPTION_PROMPT
        self.caption_prompt = caption_prompt
        
        self.validation = validation
        if validation_prompt is None:
            self.validation_prompt = BASE_IMAGE_CAPTION_VALIDATION_PROMPT
        else:
            self.validation_prompt = validation_prompt
            
        self.validation_theshold = validation_threshold
        self.max_retry = max_retry
        
        if output_file is None:
            self.output_file = None
        elif output_file.endswith(".json"):
            self.output_file = output_file
        else:
            raise ValueError("Output file must be a either None or a JSON file.")
        
        self.detail = kwargs.get("detail", "low")
        self.temperature = kwargs.get("temperature", 0)
        self.max_tokens = kwargs.get("max_tokens", 256)
        
    def encode_image(self, image_path: str) -> str:
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logging.error(f"Image encoding failed: {e}")
          
    def caption(self, image: str, feedback_prompt:str = "", **kwargs) -> str:
        
        if feedback_prompt:
            generation_prompt = f"""
                Last time the caption you generated for this image was incorrect because of the following reasons:
                {feedback_prompt}
                
                Try to generate a better caption for the image.
            """
        else:
            generation_prompt = "caption the given image."
        
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image}"
                },
                "detail": self.detail
            },
            {
                "type": "text",
                "text": generation_prompt,
            },
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self.caption_prompt
                    },
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                **kwargs
            )
            image_caption = response.choices[0].message.content.strip()
            
        except Exception as e:
            logging.error(f"Image captioning failed: {e}")
            image_caption = "ERROR"
            
        return image_caption
    
    def validate(self, image: str, caption: str, **kwargs) -> Tuple[str, float]:
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image}"
                },
                "detail": self.detail
            },
            {
                "type": "text",
                "text": caption
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self.validation_prompt
                    },
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format=ImageValidationOutput,
                **kwargs
            )
            validation_output = response.choices[0].message.parsed
            validation_reasoning = validation_output.validation_reasoning
            confidence = validation_output.confidence
            
        except Exception as e:
            logging.error(f"Image caption validation failed: {e}")
            validation_reasoning = "ERROR"
            confidence = 0.0
            
        return validation_reasoning, confidence
    
    def generate(self, image_paths: List[str], **kwargs) -> List[Dict]:
        results = []
        for image_path in tqdm(image_paths, desc="Generating captions:"):
            image = self.encode_image(image_path)
            
            if self.validation:
                validation_flag = False
                for _ in range(self.max_retry):
                    caption = self.caption(image, **kwargs)
                    validation_reasoning, confidence = self.validate(image, caption, **kwargs)
                    if confidence > self.validation_threshold:
                        results.append(
                            {
                                "image_path": image_path, 
                                "image_caption": caption, 
                                "validation_reasioning":validation_reasoning, 
                                "validation_score":confidence
                            }
                        )
                        validation_flag = True
                        break
                    else:
                        logging.info(f"Retrying captioning and validation for {image_path} as confidence score {confidence} is below threshold.")
                
                if not validation_flag:
                    logging.info(f"Caption validation failed for {image_path}. Adding to results with CAPTION_FAILED_VALIDATION tag in image_caption.")
                    results.append(
                        {
                            "image_path": image_path, 
                            "image_caption": f"CAPTION_FAILED_VALIDATION: {caption}", 
                            "validation_reasioning":validation_reasoning, 
                            "validation_score":confidence
                        }
                    )
            else:
                caption = self.caption(image, **kwargs)
                results.append(
                    {
                        "image_path": image_path, 
                        "image_caption": caption, 
                        "validation_reasioning":None, 
                        "validation_score":None
                    }
                )
        
        if self.output_file:
            with open(self.output_file, "w") as f:
                json.dump(results, f, indent=4)
            logging.info(f"Successfully saved results to {self.output_file}")
            
        return results
        
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
from openai import OpenAI
import logging
from typing import Tuple, List, Dict
from swiftannotate.image.base import BaseImageCaptioning, ImageValidationOutputOpenAI
from swiftannotate.constants import BASE_IMAGE_CAPTION_VALIDATION_PROMPT, BASE_IMAGE_CAPTION_PROMPT


class ImageCaptioningOpenAI(BaseImageCaptioning):
    """
    Image captioning pipeline using OpenAI API.
    """
    def __init__(
        self, 
        caption_model: str, 
        validation_model: str,
        api_key: str, 
        caption_prompt: str = BASE_IMAGE_CAPTION_PROMPT, 
        validation: bool = True,
        validation_prompt: str = BASE_IMAGE_CAPTION_VALIDATION_PROMPT,
        validation_threshold: float = 0.5,
        max_retry: int = 3, 
        output_file: str | None = None,
        **kwargs
    ):
        """
        Initializes the ImageCaptioningOpenAI pipeline.

        Args:
            caption_model (str): 
                Can be either "gpt-4o", "gpt-4o-mini", etc. or 
                specific versions of model supported by OpenAI.
            validation_model (str): 
                Can be either "gpt-4o", "gpt-4o-mini", etc. or 
                specific versions of model supported by OpenAI.
            api_key (str): OpenAI API key.
            caption_prompt (str | None, optional): 
                System prompt for captioning images.
                Uses default BASE_IMAGE_CAPTION_PROMPT prompt if not provided.
            validation (bool, optional): 
                Use validation step or not. Defaults to True.
            validation_prompt (str | None, optional): 
                System prompt for validating image captions should specify the range of validation score to be generated. 
                Uses default BASE_IMAGE_CAPTION_PROMPT prompt if not provided.
            validation_threshold (float, optional): 
                Threshold to determine if image caption is valid or not should be within specified range for validation score. 
                Defaults to 0.5.
            max_retry (int, optional):
                Number of retries before giving up on the image caption. 
                Defaults to 3.
            output_file (str | None, optional): 
                Output file path, only JSON is supported for now. 
                Defaults to None.
        
        Keyword Arguments:
            detail (str, optional): 
                Specific to OpenAI. Detail level of the image (Higher resolution costs more). Defaults to "low".
        
        Notes:
            `validation_prompt` should specify the rules for validating the caption and the range of validation score to be generated example (0-1).
            Your `validation_threshold` should be within this specified range.
        """
        self.caption_model = caption_model
        self.validation_model = validation_model
        self.client = OpenAI(api_key)
        
        super().__init__(
            caption_prompt=caption_prompt,
            validation=validation,
            validation_prompt=validation_prompt,
            validation_threshold=validation_threshold,
            max_retry=max_retry,
            output_file=output_file
        )
        
        self.detail = kwargs.get("detail", "low")
          
    def annotate(self, image: str, feedback_prompt:str = "", **kwargs) -> str:        
        """
        Annotates the image with a caption. Implements the logic to generate captions for an image.
        
        **Note**: The feedback_prompt is dynamically updated using the validation reasoning from 
        the previous iteration in case the caption does not pass validation threshold.
        
        Args:
            image (str): Base64 encoded image.
            feedback_prompt (str, optional): Feedback prompt for the user to generate a better caption. Defaults to ''.
            **kwargs: Additional arguments to pass to the method for custom pipeline interactions. To control generation parameters for the model.
        
        Returns:
            str: Generated caption for the image.
        """
        if feedback_prompt:
            user_prompt = f"""
                Last time the caption you generated for this image was incorrect because of the following reasons:
                {feedback_prompt}
                
                Try to generate a better caption for the image.
            """
        else:
            user_prompt = "Describe the given image."
        
        messages=[
            {"role": "system", "content": self.caption_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image}",
                            "detail": self.detail
                        },
                    },
                    {"type": "text", "text": user_prompt},
                ]
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.caption_model,
                messages=messages,
                **kwargs
            )
            image_caption = response.choices[0].message.content.strip()
            
        except Exception as e:
            logging.error(f"Image captioning failed: {e}")
            image_caption = "ERROR"
            
        return image_caption
    
    def validate(self, image: str, caption: str, **kwargs) -> Tuple[str, float]: 
        """
        Validates the caption generated for the image.

        Args:
            image (str): Base64 encoded image.
            caption (str): Caption generated for the image.

        Returns:
            Tuple[str, float]: Validation reasoning and confidence score for the caption.
        """
        if caption == "ERROR":
            return "ERROR", 0
         
        messages = [
            {
                "role": "system",
                "content": self.validation_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image}",
                            "detail": self.detail
                        },
                    },
                    {
                        "type": "text",
                        "text": caption + "\nValidate the caption generated for the given image."
                    }
                ]
            }
        ]      
        
        try:
            response = self.client.chat.completions.create(
                model=self.validation_model,
                messages=messages,
                response_format=ImageValidationOutputOpenAI,
                **kwargs
            )
            validation_output = response.choices[0].message.parsed
            validation_reasoning = validation_output.validation_reasoning
            confidence = validation_output.confidence
            
        except Exception as e:
            logging.error(f"Image caption validation failed: {e}")
            validation_reasoning = "ERROR"
            confidence = 0
            
        return validation_reasoning, confidence

    def generate(self, image_paths: List[str], **kwargs) -> List[Dict]:
        """
        Generates captions for a list of images. Implements the logic to generate captions for a list of images.

        Args:
            image_paths (List[str]): List of image paths to generate captions for.
            **kwargs: Additional arguments to pass to the method for custom pipeline interactions. To control generation parameters for the model.

        Returns:
            List[Dict]: List of captions, validation reasoning and confidence scores for each image.
        """
        results = super().generate(
            image_paths=image_paths, 
            **kwargs
        )
        
        return results
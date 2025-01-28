import ollama
from typing import Tuple, List, Dict
from swiftannotate.image.base import BaseImageCaptioning, ImageValidationOutputOllama
from swiftannotate.image.utils import setup_logger
from swiftannotate.constants import BASE_IMAGE_CAPTION_VALIDATION_PROMPT, BASE_IMAGE_CAPTION_PROMPT

# Setup logger
logger = setup_logger(__name__)

class OllamaForImageCaptioning(BaseImageCaptioning):
    """
    OllamaForImageCaptioning pipeline using Ollama API.
    
    Example usage:
    
    ```python
    from swiftannotate.image import OllamaForImageCaptioning
    
    # Initialize the pipeline
    captioner = OllamaForImageCaptioning(
        caption_model="llama3.2-vision",
        validation_model="llama3.2-vision",
        output_file="captions.json"
    )

    # Generate captions for a list of images
    image_paths = ["path/to/image1.jpg"]
    results = captioner.generate(image_paths)

    # Print results
    # Output: [
    #     {
    #         'image_path': 'path/to/image1.jpg', 
    #         'image_caption': 'A cat sitting on a table.', 
    #         'validation_reasoning': 'The caption is valid.', 
    #         'validation_score': 0.8
    #     }, 
    # ]
    ```
    """
    
    def __init__(
        self, 
        caption_model: str, 
        validation_model: str,
        caption_prompt: str = BASE_IMAGE_CAPTION_PROMPT, 
        validation: bool = True,
        validation_prompt: str = BASE_IMAGE_CAPTION_VALIDATION_PROMPT,
        validation_threshold: float = 0.5,
        max_retry: int = 3, 
        output_file: str | None = None,
    ):
        """
        Initializes the OllamaForImageCaptioning pipeline.

        Args:
            caption_model (str): 
                Can be either any of the Multimodal (Vision) models supported by Ollama.
                specific versions of model supported by Ollama.
            validation_model (str): 
                Can be either any of the Multimodal (Vision) models supported by Ollama.
                specific versions of model supported by Ollama.
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
        
        Notes:
            `validation_prompt` should specify the rules for validating the caption and the range of validation score to be generated example (0-1).
            Your `validation_threshold` should be within this specified range.
        """
        
        if not self._validate_ollama_model(caption_model):
            raise ValueError(f"Caption model {caption_model} is not supported by Ollama.")
        
        if not self._validate_ollama_model(validation_model):
            raise ValueError(f"Validation model {validation_model} is not supported by Ollama.")
            
        self.caption_model = caption_model
        self.validation_model = validation_model
        
        super().__init__(
            caption_prompt=caption_prompt,
            validation=validation,
            validation_prompt=validation_prompt,
            validation_threshold=validation_threshold,
            max_retry=max_retry,
            output_file=output_file
        )
        
    def _validate_ollama_model(self, model: str) -> bool:
        try:
            ollama.chat(model)
        except ollama.ResponseError as e:
            logger.error(f"Error: {e.error}")
            if e.status_code == 404:
                try:
                    ollama.pull(model)
                    logger.info(f"Model {model} is now downloaded.")
                except ollama.ResponseError as e:
                    logger.error(f"Error: {e.error}")
                    logger.error(f"Model {model} could not be downloaded. Check the model name and try again.")
                    return False
            logger.info(f"Model {model} is now downloaded.")
                
        return True
        
          
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
                "images": [image],
                "content": user_prompt
            }
        ]
        
        if not "temperature" in kwargs:
            kwargs["temperature"] = 0.0
                
        try:
                
            response = ollama.chat(
                model=self.caption_model,
                messages=messages,
                options=kwargs
            )
            image_caption = response.message.content
            
        except Exception as e:
            logger.error(f"Image captioning failed: {e}")
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
                "images": [image],
                "content": caption + "\nValidate the caption generated for the given image."
            }
        ]      
        
        if not "temperature" in kwargs:
            kwargs["temperature"] = 0.0
                        
        try:
    
            response = ollama.chat(
                model=self.validation_model,
                messages=messages,
                format=ImageValidationOutputOllama.model_json_schema(),
                options=kwargs
            )
            
            validation_output = ImageValidationOutputOllama.model_validate_json(response.message.content)
            
            validation_reasoning = validation_output.validation_reasoning
            confidence = validation_output.confidence
            
        except Exception as e:
            logger.error(f"Image caption validation failed: {e}")
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
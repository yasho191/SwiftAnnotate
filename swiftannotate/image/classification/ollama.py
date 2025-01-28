import ollama
from typing import Tuple, List, Dict
from swiftannotate.image.base import BaseImageClassification, ImageValidationOutputOllama
from swiftannotate.image.base import ImageClassificationOutputOllama
from swiftannotate.image.utils import setup_logger
from swiftannotate.constants import BASE_IMAGE_CLASSIFICATION_VALIDATION_PROMPT, BASE_IMAGE_CLASSIFICATION_PROMPT

# Setup logger
logger = setup_logger(__name__)

class OllamaForImageClassification(BaseImageClassification):
    """
    OllamaForImageClassification pipeline using OpenAI API.
    
    Example usage:
    
    ```python
    from swiftannotate.image import OllamaForImageClassification
    
    # Initialize the pipeline
    classification_pipeline = OllamaForImageClassification(
        classification_model="llama3.2-vision",
        validation_model="llama3.2-vision",
        classification_labels=["kitchen", "bedroom", "living room"],
        output_file="captions.json"
    )

    # Generate captions for a list of images
    image_paths = ["path/to/image1.jpg"]
    results = classification_pipeline.generate(image_paths)

    # Print results
    # Output: [
    #     {
    #         "image_path": 'path/to/image1.jpg', 
    #         "image_classification": 'kitchen', 
    #         "validation_reasoning": 'The class label is valid.', 
    #         "validation_score": 0.6
    #     }, 
    # ]
    ```
    """
    def __init__(
        self, 
        classification_model: str, 
        validation_model: str,
        classification_labels: List[str],
        classification_prompt: str = BASE_IMAGE_CLASSIFICATION_PROMPT, 
        validation: bool = True,
        validation_prompt: str = BASE_IMAGE_CLASSIFICATION_VALIDATION_PROMPT,
        validation_threshold: float = 0.5,
        max_retry: int = 3, 
        output_file: str | None = None,
    ):
        """
        Initializes the OllamaForImageClassification pipeline.

        Args:
            classification_model (str): 
                Can be either any of the Multimodal (Vision) models supported by Ollama.
                specific versions of model supported by Ollama.
            validation_model (str): 
                Can be either any of the Multimodal (Vision) models supported by Ollama.
                specific versions of model supported by Ollama.
            classification_labels (List[str]):
                List of classification labels to be used for the image classification.
            classification_prompt (str | None, optional): 
                System prompt for classification images.
                Uses default BASE_IMAGE_CLASSIFICATION_PROMPT prompt if not provided.
            validation (bool, optional): 
                Use validation step or not. Defaults to True.
            validation_prompt (str | None, optional): 
                System prompt for validating image class labels should specify the range of validation score to be generated. 
                Uses default BASE_IMAGE_CLASSIFICATION_PROMPT prompt if not provided.
            validation_threshold (float, optional): 
                Threshold to determine if image class labels is valid or not should be within specified range for validation score. 
                Defaults to 0.5.
            max_retry (int, optional):
                Number of retries before giving up on the image class labels. 
                Defaults to 3.
            output_file (str | None, optional): 
                Output file path, only JSON is supported for now. 
                Defaults to None.
        
        Notes:
            `validation_prompt` should specify the rules for validating the class label and the range of validation score to be generated example (0-1).
            Your `validation_threshold` should be within this specified range.
        """
        
        if not self._validate_ollama_model(classification_model):
            raise ValueError(f"Model {classification_model} is not supported by Ollama.")
        
        if not self._validate_ollama_model(validation_model):
            raise ValueError(f"Model {validation_model} is not supported by Ollama.")
        
        self.classification_model = classification_model
        self.validation_model = validation_model
        
        super().__init__(
            classification_labels=classification_labels,
            classification_prompt=classification_prompt,
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
        Annotates the image with a class label. Implements the logic to generate class labels for an image.
        
        **Note**: The feedback_prompt is dynamically updated using the validation reasoning from 
        the previous iteration in case the calss label does not pass validation threshold.
        
        Args:
            image (str): Base64 encoded image.
            feedback_prompt (str, optional): Feedback prompt for the user to generate a better class label. Defaults to ''.
            **kwargs: Additional arguments to pass to the method for custom pipeline interactions. To control generation parameters for the model.
        
        Returns:
            str: Generated class label for the image.
        """
        if feedback_prompt:
            user_prompt = f"""
                Last time the class label you generated for this image was incorrect because of the following reasons:
                {feedback_prompt}
                
                Regenerate the class label for the given image.
                Classify the given image as {', '.join(map(str, self.classification_labels))}
            """
        else:
            user_prompt = f"Classify the given image as {', '.join(map(str, self.classification_labels))}"
        
        messages=[
            {"role": "system", "content": self.classification_prompt},
            {
                "role": "user",
                "images": [image],
                "content": user_prompt,
            }
        ]
        
        if not "temperature" in kwargs:
            kwargs["temperature"] = 0.0
                
        try:  
            response = ollama.chat(
                model=self.classification_model,
                messages=messages,
                format=ImageClassificationOutputOllama.model_json_schema(),
                options=kwargs
            )
            
            output = ImageClassificationOutputOllama.model_validate_json(response.message.content)
            class_label = output.class_label.lower()
            
        except Exception as e:
            logger.error(f"Image classification failed: {e}")
            class_label = "ERROR"
            
        return class_label
    
    def validate(self, image: str, class_label: str, **kwargs) -> Tuple[str, float]: 
        """
        Validates the class label generated for the image.

        Args:
            image (str): Base64 encoded image.
            class_label (str): Class Label generated for the image.

        Returns:
            Tuple[str, float]: Validation reasoning and confidence score for the class label.
        """
        if class_label == "ERROR":
            return "ERROR", 0
         
        messages = [
            {
                "role": "system", "content": self.validation_prompt
            },
            {
                "role": "user",
                "images": [image],
                "content": class_label + "\nValidate the class label generated for the given image."
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
            logger.error(f"Image class label validation failed: {e}")
            validation_reasoning = "ERROR"
            confidence = 0
            
        return validation_reasoning, confidence

    def generate(self, image_paths: List[str], **kwargs) -> List[Dict]:
        """
        Generates class label for a list of images. 

        Args:
            image_paths (List[str]): List of image paths to generate class labels for.
            **kwargs: Additional arguments to pass to the method for custom pipeline interactions. To control generation parameters for the model.

        Returns:
            List[Dict]: List of class labels, validation reasoning and confidence scores for each image.
        """
        results = super().generate(
            image_paths=image_paths, 
            **kwargs
        )
        
        return results



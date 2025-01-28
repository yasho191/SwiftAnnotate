from openai import OpenAI
from typing import Tuple, List, Dict
from swiftannotate.image.base import BaseImageClassification, ImageValidationOutputOpenAI
from swiftannotate.image.base import ImageClassificationOutputOpenAI
from swiftannotate.image.utils import setup_logger
from swiftannotate.constants import BASE_IMAGE_CLASSIFICATION_VALIDATION_PROMPT, BASE_IMAGE_CLASSIFICATION_PROMPT

# Setup logger
logger = setup_logger(__name__)

class OpenAIForImageClassification(BaseImageClassification):
    """
    OpenAIForImageClassification pipeline using OpenAI API.
    
    Example usage:
    
    ```python
    from swiftannotate.image import OpenAIForImageClassification
    
    # Initialize the pipeline
    classification_pipeline = OpenAIForImageClassification(
        classification_model="gpt-4o",
        validation_model="gpt-4o-mini",
        api_key="your_api_key_here",
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
        api_key: str, 
        classification_labels: List[str],
        classification_prompt: str = BASE_IMAGE_CLASSIFICATION_PROMPT, 
        validation: bool = True,
        validation_prompt: str = BASE_IMAGE_CLASSIFICATION_VALIDATION_PROMPT,
        validation_threshold: float = 0.5,
        max_retry: int = 3, 
        output_file: str | None = None,
        **kwargs
    ):
        """
        Initializes the ImageClassificationOpenAI pipeline.

        Args:
            classification_model (str): 
                Can be either "gpt-4o", "gpt-4o-mini", etc. or 
                specific versions of model supported by OpenAI.
            validation_model (str): 
                Can be either "gpt-4o", "gpt-4o-mini", etc. or 
                specific versions of model supported by OpenAI.
            api_key (str): OpenAI API key.
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
        
        Keyword Arguments:
            detail (str, optional): 
                Specific to OpenAI. Detail level of the image (Higher resolution costs more). Defaults to "low".
        
        Notes:
            `validation_prompt` should specify the rules for validating the class label and the range of validation score to be generated example (0-1).
            Your `validation_threshold` should be within this specified range.
            
            It is advised to include class descriptions in the classification_prompt and validation_prompt to help the model understand the context of the class labels.
            You can also add Few-shot learning examples to the prompt to help the model understand the context of the class labels.
        """
        self.classification_model = classification_model
        self.validation_model = validation_model
        self.client = OpenAI(api_key)
        
        super().__init__(
            classification_labels=classification_labels,
            classification_prompt=classification_prompt,
            validation=validation,
            validation_prompt=validation_prompt,
            validation_threshold=validation_threshold,
            max_retry=max_retry,
            output_file=output_file
        )
        
        self.detail = kwargs.get("detail", "low")
          
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
                model=self.classification_model,
                messages=messages,
                response_format=ImageClassificationOutputOpenAI,
                **kwargs
            )
            output = response.choices[0].message.parsed
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
                        "text": class_label + "\nValidate the class label generated for the given image."
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



from tqdm import tqdm
import logging
from typing_extensions import TypedDict
from PIL import Image
import json
from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import List, Tuple, Dict
from swiftannotate.image.utils import encode_image


##################################################
#     Base Models for Image Annotation Tasks     #
##################################################

class BaseImageAnnotation(ABC):
    """
    Base class for image annotation. The class provides a blueprint for all image annotation models.
    Each base class must implement the following methods:
    - annotate: generates annotations for an image
    - validate: validates annotations for an image
    - generate: generates and saves annotations for a list of images
    """
    
    @abstractmethod
    def annotate(self, image: Image.Image | str, feedback_prompt: str = "", **kwargs) -> List[str]:
        """
        Generates annotations for an image. Implements the logic to generate annotations for an image.
        """
        pass

    @abstractmethod
    def validate(self, image: Image.Image | str, annotation: List[str], **kwargs) -> Tuple[str, float]:
        """
        Validates annotations for an image. Implements the logic to validate annotations for an image.
        """
        pass
    
    @abstractmethod
    def generate(self, image_paths: List[str], **kwargs) -> List[Dict]:
        """
        Generates annotations for a list of images. Implements the logic to generate annotations for a list of images.
        """
        pass


# Image Captioning Base Class
class BaseImageCaptioning(BaseImageAnnotation):
    """
    Base class for image captioning. The class provides a blueprint for all image captioning models.
    Each base class must implement the following methods:
    - caption: generates a caption for an image
    - validate: validates a caption for an image
    
    The class provides a default method to generate captions for a list of images:
    - generate: generates captions for a list of images
    """
    
    def __init__(
        self, 
        caption_prompt: str, 
        validation: bool,
        validation_prompt: str,
        validation_threshold: float,
        max_retry: int, 
        output_file: str | None = None,
    ):
        self.caption_prompt = caption_prompt
        
        self.validation = validation
        self.validation_prompt = validation_prompt    
        self.validation_threshold = validation_threshold
        
        self.max_retry = max_retry
        assert self.max_retry > 0, "max_retry must be greater than 0."
        
        if output_file is None:
            self.output_file = None  
        else:
            assert output_file.endswith(".json") == True, "Output file must be a either None or a JSON file."
            self.output_file = output_file
    
    @abstractmethod
    def annotate(self, image: str, feedback_prompt: str = "", **kwargs) -> str:
        """
        Generates a caption for an image. Implements the logic to generate a caption for an image.
        """
        raise NotImplementedError("annotate method must be implemented in every subclass.")
    
    @abstractmethod
    def validate(self, image: str, caption: str, **kwargs) -> Tuple[str, float]:
        """
        Validates a caption for an image. Implements the logic to validate a caption for an image.
        """
        raise NotImplementedError("validate method must be implemented in every subclass.")
    
    def generate(self, image_paths: List[str], **kwargs) -> List[Dict]:
        """
        Generates captions for a list of images. Implements the logic to generate captions for a list of images.

        Args:
            image_paths (List[str]): List of image paths to generate captions for.
            **kwargs: Additional arguments to pass to the method for custom pipeline interactions. To control generation parameters for the model.

        Returns:
            List[Dict]: List of captions, validation reasoning and confidence scores for each image.
        """
        results = []
        for image_path in tqdm(image_paths, desc="Generating captions:"):
            image = encode_image(image_path)
            
            if self.validation:
                validation_flag = False
                for _ in range(self.max_retry):
                    
                    # Generate caption
                    caption = self.annotate(image, **kwargs)
                    caption_retry_counter = 0
                    while caption == "ERROR" and caption_retry_counter < 5:
                        logging.info(f"Retrying captioning for {image_path} as an error occurred.")
                        caption = self.annotate(image, **kwargs)
                        caption_retry_counter += 1
                    
                    # Validate caption
                    validation_reasoning, confidence = self.validate(image, caption, **kwargs) 
                    validation_retry_counter = 0
                    while validation_reasoning == "ERROR" and confidence == 0 and validation_retry_counter < 5:
                        logging.info(f"Retrying caption validation for {image_path} as an error occurred.")
                        validation_reasoning, confidence = self.validate(image, caption, **kwargs)
                        validation_retry_counter += 1
                        
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
                caption = self.annotate(image, **kwargs)
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
    

# Image Classification Base Class
class BaseImageClassification(BaseImageAnnotation):
    """
    Base class for image classification. The class provides a blueprint for all image classification models.
    Each base class must implement the following methods:
    - classify: generates a classification for an image
    - validate: validates a classification for an image
    
    The class provides a default method to generate classifications for a list of images:
    - generate: generates classifications for a list of images
    """
    
    def __init__(
        self, 
        classification_prompt: str, 
        validation: bool,
        validation_prompt: str,
        validation_threshold: float,
        max_retry: int, 
        output_file: str | None = None,
        **kwargs
    ):
        self.classification_prompt = classification_prompt
        
        self.validation = validation
        self.validation_prompt = validation_prompt  
        self.validation_threshold = validation_threshold
        
        self.max_retry = max_retry
        assert self.max_retry > 0, "max_retry must be greater than 0."
        
        if output_file is None:
            self.output_file = None  
        else:
            assert output_file.endswith(".json") == True, "Output file must be a either None or a JSON file."
            self.output_file = output_file
    
    @abstractmethod
    def annotate(self, image: Image.Image | str, feedback_prompt: str = "", **kwargs) -> List[str]:
        """
        Generates a classification for an image. Implements the logic to generate a classification for an image.
        """
        raise NotImplementedError("annotate method must be implemented in every subclass.")
    
    @abstractmethod
    def validate(self, image: Image.Image | str, classification: List[str], **kwargs) -> Tuple[str, float]:
        """
        Validates a classification for an image. Implements the logic to validate a classification for an image.
        """
        raise NotImplementedError("validate method must be implemented in every subclass.")
    
    def generate(self, image_paths: List[str], **kwargs) -> List[Dict]:
        """
        Generates classifications for a list of images. Implements the logic to generate classifications for a list of images.

        Args:
            image_paths (List[str]): List of image paths to generate classifications for.
            **kwargs: Additional arguments to pass to the method for custom pipeline interactions. To control generation parameters for the model.

        Returns:
            List[Dict]: List of classifications, validation reasoning and confidence scores for each image.
        """
        results = []
        for image_path in tqdm(image_paths, desc="Generating classifications:"):
            image = encode_image(image_path)
            
            if self.validation:
                validation_flag = False
                for _ in range(self.max_retry):
                    
                    # Generate classification
                    classification = self.annotate(image, **kwargs)
                    classification_retry_counter = 0
                    while classification == "ERROR" and classification_retry_counter < 5:
                        logging.info(f"Retrying classification for {image_path} as an error occurred.")
                        classification = self.annotate(image, **kwargs)
                        classification_retry_counter += 1
                    
                    # Validate classification
                    validation_reasoning, confidence = self.validate(image, classification, **kwargs) 
                    validation_retry_counter = 0
                    while validation_reasoning == "ERROR" and confidence == 0 and validation_retry_counter < 5:
                        logging.info(f"Retrying classification validation for {image_path} as an error occurred.")
                        validation_reasoning, confidence = self.validate(image, classification, **kwargs)
                        validation_retry_counter += 1
                        
                    if confidence > self.validation_threshold:
                        results.append(
                            {
                                "image_path": image_path, 
                                "image_classification": classification, 
                                "validation_reasioning":validation_reasoning, 
                                "validation_score":confidence
                            }
                        )
                        validation_flag = True
                        break
                    else:
                        logging.info(f"Retrying classification and validation for {image_path} as confidence score {confidence} is below threshold.")
                
                if not validation_flag:
                    logging.info(f"Classification validation failed for {image_path}. Adding to results with CLASSIFICATION_FAILED_VALIDATION tag in image_classification.")
                    results.append(
                        {
                            "image_path": image_path, 
                            "image_classification": f"CLASSIFICATION_FAILED_VALIDATION: {classification}", 
                            "validation_reasioning":validation_reasoning, 
                            "validation_score":confidence
                        }
                    )
            else:
                classification = self.annotate(image, **kwargs)
                results.append(
                    {
                        "image_path": image_path, 
                        "image_classification": classification, 
                        "validation_reasioning":None, 
                        "validation_score":None
                    }
                )
        
        if self.output_file:
            with open(self.output_file, "w") as f:
                json.dump(results, f, indent=4)
            logging.info(f"Successfully saved results to {self.output_file}")
        
        return results


################################################
#     Pydantic Models for Stuctured Output     #
################################################

class ImageValidationOutputOpenAI(BaseModel):
    validation_reasoning: str
    confidence: float


class ImageValidationOutputGemini(TypedDict):
    validation_reasoning: str
    confidence: float
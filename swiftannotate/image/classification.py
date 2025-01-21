import torch
from openai import OpenAI
import google.generativeai as genai
import logging
import json
from tqdm import tqdm
from typing import Tuple, List, Dict
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoModelForImageTextToText
from swiftannotate.image.base import BaseImageClassification, ImageValidationOutputOpenAI, ImageValidationOutputGemini
from swiftannotate.image.base import ImageClassificationOutputGemini, ImageClassificationOutputOpenAI
from swiftannotate.constants import BASE_IMAGE_CLASSIFICATION_VALIDATION_PROMPT, BASE_IMAGE_CLASSIFICATION_PROMPT

class ImageClassificationOpenAI(BaseImageClassification):
    """
    Image classification pipeline using OpenAI API. 
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
            logging.error(f"Image classification failed: {e}")
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
            logging.error(f"Image class label validation failed: {e}")
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


class ImageClassificationGemini(BaseImageClassification):
    def __init__(
        self, 
        classification_model: str, 
        validation_model: str,
        api_key: str, 
        classification_labels: List[str],
        classification_prompt: str | None = None, 
        validation: bool = True,
        validation_prompt: str | None = None,
        validation_threshold: float = 0.5,
        max_retry: int = 3, 
        output_file: str | None = None,
    ):
        """
        Initializes the ImageClassificationGemini pipeline.

        Args:
            classification_model (str): 
                Can be either "gemini-1.5-flash", "gemini-1.5-pro", etc. or specific versions of model supported by Gemini.
            validation_model (str): 
                Can be either "gemini-1.5-flash", "gemini-1.5-pro", etc. or specific versions of model supported by Gemini.
            api_key (str): 
                Google Gemini API key.
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
        genai.configure(api_key=api_key)
        self.classification_model = genai.GenerativeModel(model=classification_model)
        self.validation_model = genai.GenerativeModel(model=validation_model)
        
        super().__init__(
            classification_labels=classification_labels,
            classification_prompt=classification_prompt,
            validation=validation,
            validation_prompt=validation_prompt,
            validation_threshold=validation_threshold,
            max_retry=max_retry,
            output_file=output_file
        )
    
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
            
        messages = [
            self.classification_prompt,
            {'mime_type':'image/jpeg', 'data': image}, 
            user_prompt
        ]
        
        try:
            output = self.classification_model.generate_content(
                messages,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json", 
                    response_schema=ImageClassificationOutputGemini,
                    **kwargs
                )
            )
            class_label = output["class_label"].lower()
        except Exception as e:
            logging.error(f"Image classification failed: {e}")
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
            return "ERROR", 0.0

        messages = [
            self.validation_prompt,
            {'mime_type':'image/jpeg', 'data': image},
            class_label,
            "Validate the class label generated for the given image."
        ]
        
        try:
            validation_output = self.validation_model.generate_content(
                messages,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json", 
                    response_schema=ImageValidationOutputGemini
                )
            )
            validation_reasoning = validation_output["validation_reasoning"]
            confidence = validation_output["confidence"]
        except Exception as e:
            logging.error(f"Image class label validation failed: {e}")
            validation_reasoning = "ERROR"
            confidence = 0.0
        
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
    
    
class ImageClassificationQwen2VL(BaseImageClassification):
    """
    Image classification pipeline using Qwen2VL model.
    """
    def __init__(
        self, 
        model: AutoModelForImageTextToText, 
        processor: AutoProcessor,
        classification_labels: List[str],
        classification_prompt: str | None = None, 
        validation: bool = True,
        validation_prompt: str | None = None,
        validation_threshold: float = 0.5,
        max_retry: int = 3, 
        output_file: str | None = None,
        **kwargs
    ):     
        """
        Initializes the ImageClassificationQwen2VL pipeline.

        Args:
            model (AutoModelForImageTextToText): 
                Model for image classification. Should be an instance of AutoModelForImageTextToText with Qwen2-VL pretrained weights.
                Can be any version of Qwen2-VL model (7B, 72B).
            processor (AutoProcessor): 
                Processor for the Qwen2-VL model. Should be an instance of AutoProcessor.
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
            resize_height (int, optional):
                Height to resize the image before generating class labels. Defaults to 280.
            resize_width (int, optional):
                Width to resize the image before generating class labels. Defaults to 420.
        
        Notes:
            `validation_prompt` should specify the rules for validating the class label and the range of validation score to be generated example (0-1).
            Your `validation_threshold` should be within this specified range.
        """     
        self.model = model
        self.processor = processor
        
        super().__init__(
            classification_labels=classification_labels,
            classification_prompt=classification_prompt,
            validation=validation,
            validation_prompt=validation_prompt,
            validation_threshold=validation_threshold,
            max_retry=max_retry,
            output_file=output_file
        )
        
        self.resize_height = kwargs.get("resize_height", 280)
        self.resize_width = kwargs.get("resize_width", 420)

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
                Return output as a JSON object with key as 'class_label'
            """
        else:
            user_prompt = f"Classify the given image as {', '.join(map(str, self.classification_labels))} \nReturn output as a JSON object with key as 'class_label'"
        
        messages = [
            {"role": "system", "content": self.classification_prompt},
            {
                "role": "user", 
                "content": [
                    {
                        "type": "image", 
                        "image": f"data:image;base64,{image}",
                        "resized_height": self.resize_height,
                        "resized_width": self.resize_width,
                    },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, **kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        class_label = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        try:
            class_label = class_label.replace('```', '').replace('json', '')
            class_label = json.loads(class_label)
            class_label = class_label["class_label"].lower()
        except Exception as e:
            logging.error(f"Image classification parsing failed trying to parse using another logic.")
            potential_class_labels = [label.lower() for label in class_label.split() if label in self.classification_labels]
            class_label = potential_class_labels[0] if potential_class_labels else "ERROR"
        
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
        messages = [
            {"role": "system", "content": self.validation_prompt},
            {
                "role": "user", 
                "content": [
                    {
                        "type": "image", 
                        "image": f"data:image;base64,{image}",
                        "resized_height": self.resize_height,
                        "resized_width": self.resize_width,
                    },
                    {"type": "text", "text": class_label},
                    {
                        "type": "text", 
                        "text": """
                        Validate the class label generated for the given image. 
                        Return output as a JSON object with keys as 'validation_reasoning' and 'confidence'.
                        """
                    },
                ],
            },
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, **kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        validation_output = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # TODO: Need a better way to parse the output
        try:
            validation_output = validation_output.replace('```', '').replace('json', '')
            validation_output = json.loads(validation_output)
            validation_reasoning = validation_output["validation_reasoning"]
            confidence = validation_output["confidence"]
        except Exception as e:
            logging.error(f"Image class label validation parsing failed trying to parse using another logic.")
            
            number_str  = ''.join((ch if ch in '0123456789.-e' else ' ') for ch in validation_output)
            potential_confidence_scores = [float(i) for i in number_str.split() if float(i) >= 0 and float(i) <= 1]
            confidence = max(potential_confidence_scores) if potential_confidence_scores else 0.0
            validation_reasoning = validation_output
        
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

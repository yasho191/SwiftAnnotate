from openai import OpenAI
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from tqdm import tqdm
import logging
import json
from PIL import Image
from typing import List, Tuple, Dict
from pydantic import BaseModel
from qwen_vision_utils import process_vision_info

from swiftannotate.image.utils import encode_image
from swiftannotate.constants import BASE_IMAGE_CAPTION_VALIDATION_PROMPT, BASE_IMAGE_CAPTION_PROMPT

class ImageValidationOutput(BaseModel):
    validation_reasoning: str
    confidence: float

class BaseImageCaptioning:
    """
    Base class for image captioning. The class provides a blueprint for all image captioning models.
    Each base class must implement the following methods:
    - caption: generates a caption for an image
    - validate: validates a caption for an image
    
    The class also provides a default method to generate captions for a list of images:
    - generate: generates captions for a list of images
    """
    
    def __init__(
        self, 
        caption_prompt: str | None = None, 
        validation: bool = True,
        validation_prompt: str | None = None,
        validation_threshold: float = 0.5,
        max_retry: int = 3, 
        output_file: str | None = None,
        **kwargs
    ):
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
        results = []
        for image_path in tqdm(image_paths, desc="Generating captions:"):
            image = encode_image(image_path)
            
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
    
class ImageCaptionOpenAI(BaseImageCaptioning):
    def __init__(
        self, 
        caption_model: str, 
        validation_model: str,
        api_key: str, 
        caption_prompt: str | None = None, 
        validation: bool = True,
        validation_prompt: str | None = None,
        validation_threshold: float = 0.5,
        max_retry: int = 3, 
        output_file: str | None = None,
        **kwargs
    ):
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
        self.temperature = kwargs.get("temperature", 0)
        self.max_tokens = kwargs.get("max_tokens", 256)
          
    def caption(self, image: str, feedback_prompt:str = "", **kwargs) -> str:
        
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
        
class ImageCaptionQwen2VL(BaseImageCaptioning):
    def __init__(
        self, 
        model: str, 
        processor: str,
        caption_prompt: str | None = None, 
        validation: bool = True,
        validation_prompt: str | None = None,
        validation_threshold: float = 0.5,
        max_retry: int = 3, 
        output_file: str | None = None,
        **kwargs
    ):
        self.model = model
        self.processor = processor
        self.max_retry = max_retry
        
        super().__init__(
            caption_prompt=caption_prompt,
            validation=validation,
            validation_prompt=validation_prompt,
            validation_threshold=validation_threshold,
            max_retry=max_retry,
            output_file=output_file
        )
        
        self.resize_height = kwargs.get("resize_height", 280)
        self.resize_width = kwargs.get("resize_width", 420)
        
        # TODO: Add logic only if not supported by Qwen2VL vision processor
        # Round off height and width to nearest multiple of 28
        # self.resize_height = round(self.resize_height / 28) * 28
        # self.resize_width = round(self.resize_width / 28) * 28

    def caption(self, image: str, feedback_prompt:str = "", **kwargs) -> str:
        
        if feedback_prompt:
            user_prompt = f"""
                Last time the caption you generated for this image was incorrect because of the following reasons:
                {feedback_prompt}
                
                Try to generate a better caption for the image.
            """
        else:
            user_prompt = "Describe the given image."
        
        messages = [
            {"role": "system", "content": self.caption_prompt},
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
        image_caption = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return image_caption
    
    def validate(self, image: str, caption: str, **kwargs) -> Tuple[str, float]:
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
                    {"type": "text", "text": caption},
                    {
                        "type": "text", 
                        "text": """
                        Validate the caption generated for the given image. 
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
            logging.error(f"Image caption validation parsing failed trying to parse using another logic.")
            
            number_str  = ''.join((ch if ch in '0123456789.-e' else ' ') for ch in validation_output)
            potential_confidence_scores = [float(i) for i in number_str.split() if float(i) >= 0 and float(i) <= 1]
            confidence = max(potential_confidence_scores) if potential_confidence_scores else 0.0
            validation_reasoning = validation_output
        
        return validation_reasoning, confidence
    

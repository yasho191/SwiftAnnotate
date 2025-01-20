import torch
from openai import OpenAI
import google.generativeai as genai
import logging
import json
from typing import Tuple
from qwen_vision_utils import process_vision_info
from transformers import AutoProcessor, AutoModelForImageTextToText
from swiftannotate.image.base import BaseImageCaptioning, ImageValidationOutputOpenAI, ImageValidationOutputGemini

    
class ImageCaptioningOpenAI(BaseImageCaptioning):
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
          
    def annotate(self, image: str, feedback_prompt:str = "", **kwargs) -> str:
        
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

class ImageCaptioningGemini(BaseImageCaptioning):
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
        genai.configure(api_key=api_key)
        self.caption_model = genai.GenerativeModel(model=caption_model)
        self.validation_model = genai.GenerativeModel(model=validation_model)
        
        super().__init__(
            caption_prompt=caption_prompt,
            validation=validation,
            validation_prompt=validation_prompt,
            validation_threshold=validation_threshold,
            max_retry=max_retry,
            output_file=output_file
        )
    
    def annotate(self, image: str, feedback_prompt:str = "", **kwargs) -> str:
        
        if feedback_prompt:
            user_prompt = f"""
                Last time the caption you generated for this image was incorrect because of the following reasons:
                {feedback_prompt}
                
                Try to generate a better caption for the image.
            """
        else:
            user_prompt = "Describe the given image."
            
        messages = [
            self.caption_prompt,
            {'mime_type':'image/jpeg', 'data': image}, 
            user_prompt
        ]
        
        try:
            image_caption = self.caption_model.generate_content(
                messages,
                generation_config=genai.GenerationConfig(
                    **kwargs
                )
            )
        except Exception as e:
            logging.error(f"Image captioning failed: {e}")
            image_caption = "ERROR"
        
        return image_caption

    def validate(self, image: str, caption: str, **kwargs) -> Tuple[str, float]:
        if caption == "ERROR":
            return "ERROR", 0.0

        messages = [
            self.validation_prompt,
            {'mime_type':'image/jpeg', 'data': image},
            caption,
            "Validate the caption generated for the given image."
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
            logging.error(f"Image caption validation failed: {e}")
            validation_reasoning = "ERROR"
            confidence = 0.0
        
        return validation_reasoning, confidence        
    
    
class ImageCaptioningQwen2VL(BaseImageCaptioning):
    def __init__(
        self, 
        model: AutoModelForImageTextToText, 
        processor: AutoProcessor,
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

    def annotate(self, image: str, feedback_prompt:str = "", **kwargs) -> str:
        
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
    
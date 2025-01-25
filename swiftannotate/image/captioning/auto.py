import json
import google.generativeai as genai
from openai import OpenAI
from typing import Tuple, List, Dict
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from swiftannotate.image.base import BaseImageCaptioning
from swiftannotate.image.base import ImageValidationOutputGemini
from swiftannotate.image.utils import setup_logger
from swiftannotate.constants import BASE_IMAGE_CAPTION_VALIDATION_PROMPT, BASE_IMAGE_CAPTION_PROMPT

# Setup logger
logger = setup_logger(__name__) 

class AutoModelForImageCaptioning(BaseImageCaptioning):
    """
    AutoModelForImageCaptioning pipeline supporting different combinations of annotation and validation models
    OpenAI, Gemini, and Qwen2-VL.
    
    Example Usage:
    ```python
    from swiftannotate.image import AutoModelForImageCaptioning
    
    # Initialize the pipeline
    # Note: You can use either Qwen2VL, OpenAI, and Gemini for captioning and validation.
    captioner = AutoModelForImageCaptioning(
        caption_model="gpt-4o",
        validation_model="gemini-1.5-flash",
        caption_api_key="your_openai_api_key",
        validation_api_key="your_gemini_api_key",
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
    
    SUPPORTED_MODELS = {
        "openai": ["gpt-4o", "gpt-4o-mini"],
        "gemini": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp", "gemini-1.5-flash-8b"],
        "local": [Qwen2VLForConditionalGeneration]
    }

    def __init__(
        self, 
        caption_model: str | Qwen2VLForConditionalGeneration, 
        validation_model: str | Qwen2VLForConditionalGeneration,
        caption_model_processor: Qwen2VLProcessor | None = None,
        validation_model_processor: Qwen2VLProcessor | None = None,
        caption_api_key: str | None = None, 
        validation_api_key: str | None = None,
        caption_prompt: str = BASE_IMAGE_CAPTION_PROMPT, 
        validation: bool = True,
        validation_prompt: str = BASE_IMAGE_CAPTION_VALIDATION_PROMPT,
        validation_threshold: float = 0.5,
        max_retry: int = 3, 
        output_file: str | None = None,
        **kwargs
    ):
        """
        Initialize the AutoModelForImageCaptioning class.
        This class provides functionality for automatic image captioning with optional validation.
        It supports different combinations of annotation and validation models like OpenAI, Gemini, and Qwen2-VL.
        
        Args:
            caption_model (Union[str, Qwen2VLForConditionalGeneration]): 
                Model or API endpoint for caption generation.
                Can be either a local model instance or API endpoint string.
            validation_model (Union[str, Qwen2VLForConditionalGeneration]): 
                Model or API endpoint for caption validation.
                Can be either a local model instance or API endpoint string.
            caption_model_processor (Optional[Qwen2VLProcessor]): 
                Processor for caption model. 
                Required if using a local model for captioning.
            validation_model_processor (Optional[Qwen2VLProcessor]): 
                Processor for validation model.
                Required if using a local model for validation.
            caption_api_key (Optional[str]): 
                API key for caption service if using API endpoint.
            validation_api_key (Optional[str]): 
                API key for validation service if using API endpoint.
            caption_prompt (str): 
                Prompt template for caption generation.
                Defaults to BASE_IMAGE_CAPTION_PROMPT.
            validation (bool): 
                Whether to perform validation on generated captions.
                Defaults to True.
            validation_prompt (str): 
                Prompt template for caption validation.
                Defaults to BASE_IMAGE_CAPTION_VALIDATION_PROMPT.
            validation_threshold (float): 
                Threshold score for caption validation.
                Defaults to 0.5.
            max_retry (int): 
                Maximum number of retry attempts for failed validation.
                Defaults to 3.
            output_file (Optional[str]): 
                Path to save results.
                If None, results are not saved.
            **kwargs: Additional arguments passed to model initialization.
            
        Raises:
            ValueError: If required model processors are not provided for local models.
            ValueError: If an unsupported model is provided.
            
        Note:
            At least one of caption_model_processor or caption_api_key must be provided for caption generation.
            Same applies for validation_model_processor or validation_api_key if validation is enabled.
        """
        super().__init__(
            caption_prompt=caption_prompt,
            validation=validation,
            validation_prompt=validation_prompt,
            validation_threshold=validation_threshold,
            max_retry=max_retry,
            output_file=output_file
        )
        
        self.caption_model, self.caption_model_processor, self.caption_model_type = self._initialize_model(
            caption_model, caption_model_processor, caption_api_key, "caption", **kwargs
        )
        
        self.validation_model, self.validation_model_processor, self.validation_model_type = self._initialize_model(
            validation_model, validation_model_processor, validation_api_key, "validation", **kwargs
        )

    def _initialize_model(self, model, processor, api_key, stage, **kwargs):
        """Initialize model based on type."""
        if isinstance(model, str):
            if model in self.SUPPORTED_MODELS["openai"]:
                self.detail = kwargs.get("detail", "low")
                self.client = OpenAI(api_key)
                return model, None, "openai"
            elif model in self.SUPPORTED_MODELS["gemini"]:
                return genai.GenerativeModel(model_name=model), None, "gemini"
            else:
                raise ValueError(f"Unsupported model: {model}")
        
        elif isinstance(model, Qwen2VLForConditionalGeneration):
            if processor is None:
                raise ValueError(f"Processor is required for Qwen2VL model in {stage} stage")
            self.resize_height = kwargs.get("resize_height", 280)
            self.resize_width = kwargs.get("resize_width", 420)
            return model, processor, "qwen"
        
        raise ValueError(f"Invalid model type for {stage} stage")
    
    def _openai_inference(self, messages: List[str], **kwargs):
        """Inference for OpenAI model."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.caption_model,
                messages=messages,
                **kwargs
            )
            image_caption = response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Image captioning failed: {e}")
            image_caption = "ERROR"
            
        return image_caption
    
    def _gemini_inference(self, messages: List[str], stage: str, **kwargs):
        """Inference for Gemini model."""
        
        if stage == "annotate":
            try:
                image_caption = self.caption_model.generate_content(
                    messages,
                    generation_config=genai.GenerationConfig(
                        **kwargs
                    )
                )
            except Exception as e:
                logger.error(f"Image captioning failed: {e}")
                image_caption = "ERROR"
            
            return image_caption
        else:
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
                logger.error(f"Image caption validation failed: {e}")
                validation_reasoning = "ERROR"
                confidence = 0.0
            
            return validation_reasoning, confidence
    
    def _qwen_inference(self, model: Qwen2VLForConditionalGeneration, processor: Qwen2VLProcessor, messages: List[Dict], stage: str, **kwargs):
        """Inference for Qwen2VL model."""
        
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # Inference: Generation of the output
        if "max_new_tokens" not in kwargs:
            kwargs["max_new_tokens"] = 512
            
        generated_ids = model.generate(**inputs, **kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        if stage == "annotate":
            image_caption = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            return image_caption
        else:
            validation_output = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            # TODO: Need a better way to parse the output
            try:
                validation_output = validation_output.replace('```', '').replace('json', '')
                validation_output = json.loads(validation_output)
                validation_reasoning = validation_output["validation_reasoning"]
                confidence = validation_output["confidence"]
            except Exception as e:
                logger.error(f"Image caption validation parsing failed trying to parse using another logic.")
                
                number_str  = ''.join((ch if ch in '0123456789.-e' else ' ') for ch in validation_output)
                number_str = [i for i in number_str.split() if i.isalnum()]
                potential_confidence_scores = [float(i) for i in number_str if float(i) >= 0 and float(i) <= 1]
                confidence = max(potential_confidence_scores) if potential_confidence_scores else 0.0
                validation_reasoning = validation_output
            
            return validation_reasoning, confidence

    def annotate(self, image: str, feedback_prompt: str, **kwargs) -> str:
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
        
        if self.caption_model_type == "openai":
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
            
            caption = self._openai_inference(messages, "annotate", **kwargs)
            
        elif self.caption_model_type == "gemini":
            messages = [
                self.caption_prompt,
                {"mime_type": "image/jpeg", "data": image},
                user_prompt,
            ]
            
            caption = self._gemini_inference(messages, "annotate", **kwargs)
            
        else:
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
            
            caption = self._qwen_inference(self.caption_model, self.caption_model_processor , messages, "annotate", **kwargs)
        
        return caption
        

    def validate(self, image: str, caption: str, **kwargs) -> Tuple[bool, float]:
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
        
        if self.caption_model_type == "openai":
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
            validation_reasoning, confidence = self._openai_inference(messages, "validate", **kwargs)  
        elif self.caption_model_type == "gemini":
            messages = [
                self.validation_prompt,
                {'mime_type':'image/jpeg', 'data': image},
                caption,
                "Validate the caption generated for the given image."
            ]
            validation_reasoning, confidence = self._gemini_inference(messages, "validate", **kwargs)
        else:
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
            validation_reasoning, confidence = self._qwen_inference(self.validation_model, self.validation_model_processor, messages, "validate", **kwargs)
        
        return validation_reasoning, confidence

    def generate(self, image_paths, **kwargs):
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
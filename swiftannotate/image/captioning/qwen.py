import json
from typing import Tuple, List, Dict
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoModelForImageTextToText, Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from swiftannotate.image.base import BaseImageCaptioning
from swiftannotate.image.utils import setup_logger
from swiftannotate.constants import BASE_IMAGE_CAPTION_VALIDATION_PROMPT, BASE_IMAGE_CAPTION_PROMPT

# Setup logger
logger = setup_logger(__name__)
    
class Qwen2VLForImageCaptioning(BaseImageCaptioning):
    """
    Qwen2VLForImageCaptioning pipeline using Qwen2VL model.
    
    Example usage:
    ```python
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from transformers import BitsAndBytesConfig
    from swiftannotate.image import Qwen2VLForImageCaptioning

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForImageTextToText.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        device_map="auto",
        torch_dtype="auto",
        quantization_config=quantization_config)

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    # Load the Caption Model
    captioning_pipeline = ImageCaptioningQwen2VL(
        model = model,
        processor = processor,
        output_file="captions.json"
    )

    # Generate captions for images
    image_paths = ['path/to/image1.jpg']
    results = captioning_pipeline.generate(image_paths)
    
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
        model: AutoModelForImageTextToText | Qwen2VLForConditionalGeneration, 
        processor: AutoProcessor | Qwen2VLProcessor,
        caption_prompt: str = BASE_IMAGE_CAPTION_PROMPT, 
        validation: bool = True,
        validation_prompt: str = BASE_IMAGE_CAPTION_VALIDATION_PROMPT,
        validation_threshold: float = 0.5,
        max_retry: int = 3, 
        output_file: str | None = None,
        **kwargs
    ):     
        """
        Initializes the ImageCaptioningQwen2VL pipeline.

        Args:
            model (AutoModelForImageTextToText): 
                Model for image captioning. Should be an instance of AutoModelForImageTextToText with Qwen2-VL pretrained weights.
                Can be any version of Qwen2-VL model (7B, 72B).
            processor (AutoProcessor): 
                Processor for the Qwen2-VL model. Should be an instance of AutoProcessor.
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
            resize_height (int, optional):
                Height to resize the image before generating captions. Defaults to 280.
            resize_width (int, optional):
                Width to resize the image before generating captions. Defaults to 420.
        
        Notes:
            `validation_prompt` should specify the rules for validating the caption and the range of validation score to be generated example (0-1).
            Your `validation_threshold` should be within this specified range.
        """    
        
        if not isinstance(model, Qwen2VLForConditionalGeneration):
            raise ValueError("Model should be an instance of Qwen2VLForConditionalGeneration.")
        if not isinstance(processor, Qwen2VLProcessor):
            raise ValueError("Processor should be an instance of Qwen2VLProcessor.")
         
        self.model = model
        self.processor = processor
        
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
        if "max_new_tokens" not in kwargs:
            kwargs["max_new_tokens"] = 512
            
        generated_ids = self.model.generate(**inputs, **kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        image_caption = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
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
        if "max_new_tokens" not in kwargs:
            kwargs["max_new_tokens"] = 512
            
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
            logger.error(f"Image caption validation parsing failed trying to parse using another logic.")
            
            number_str  = ''.join((ch if ch in '0123456789.-e' else ' ') for ch in validation_output)
            number_str = [i for i in number_str.split() if i.isalnum()]
            potential_confidence_scores = [float(i) for i in number_str if float(i) >= 0 and float(i) <= 1]
            confidence = max(potential_confidence_scores) if potential_confidence_scores else 0.0
            validation_reasoning = validation_output
        
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

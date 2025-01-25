from PIL import Image
import torch
from typing import List, Tuple
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from swiftannotate.image.base import BaseObjectDetection
from swiftannotate.image.utils import setup_logger

logger = setup_logger(__name__)

class OwlV2ForObjectDetection(BaseObjectDetection):
    def __init__(
        self,
        model: Owlv2ForObjectDetection,
        processor: Owlv2Processor,
        class_labels: List[str],
        confidence_threshold: float = 0.5,
        validation: bool = False,
        validation_prompt: str | None = None,
        validation_threshold: float | None = None,
        output_file: str | None = None
    ):
        """
        Initialize the OwlV2ObjectDetection class.

        Args:
            model (Owlv2ForObjectDetection):
                OwlV2 Object Detection model from Transformers.
            processor (Owlv2Processor): 
                OwlV2 Processor for Object Detection.
            class_labels (List[str]): 
                List of class labels.
            confidence_threshold (float, optional): 
                Minimum confidence threshold for object detection. 
                Defaults to 0.5.
            validation (bool, optional): 
                Whether to validate annotations from OwlV2. 
                Defaults to False.
            validation_prompt (str | None, optional): 
                Prompt to validate annotations. 
                Defaults to None.
            validation_threshold (float | None, optional): 
                Threshold score for annotation validation. 
                Defaults to None.
            output_file (str | None, optional): 
                Path to save results.
                If None, results are not saved. Defaults to None.

        Raises:
            ValueError: If model is not an instance of Owlv2ForObjectDetection.
            ValueError: If processor is not an instance of Owlv2Processor.
        """
        if not isinstance(model, Owlv2ForObjectDetection):
            raise ValueError("Model must be an instance of Owlv2ForObjectDetection")
        if not isinstance(processor, Owlv2Processor):
            raise ValueError("Processor must be an instance of Owlv2Processor")
        
        self.processor = processor
        self.model = model
        self.model.eval()
        
        super().__init__(
            class_labels,
            confidence_threshold,
            validation,
            validation_prompt,
            validation_threshold,
            output_file
        )

    def annotate(self, image: Image.Image) -> List[dict]:
        """
        Annotate an image with object detection labels

        Args:
            image (Image.Image): Image to be annotated.

        Returns:
            List[dict]: List of dictionaries containing the confidence scores, bounding box coordinates and class labels.
        """
        inputs = self.processor(text=self.class_labels, images=image, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.Tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs=outputs, 
            target_sizes=target_sizes, 
            threshold=self.confidence_threshold
        )
        return [{k: v.cpu().tolist() for k, v in prediction.items()} for prediction in results]

    def validate(self, image: Image.Image, annotations: List[dict]) -> Tuple:
        """
        Validate the annotations for an image with object detection labels.
        
        Currently, there is no validation method available for Object Detection.
        
        # TODO: Idea is to do some sort of object etraction using annotations and ask VLM to validate the extracted objects.
        # TODO: Need to fihgure out a way to use the VLM output for improving annotations.

        Args:
            image (Image.Image): Image to be validated.
            annotations (List[dict]): List of dictionaries containing the confidence scores, bounding box coordinates and class labels.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError("No validation method available for Object Detection yet")
    
    def generate(self, image_paths: List[str]) -> List[dict]:
        """
        Generate annotations for a list of image paths.

        Args:
            image_paths (List[str]): List of image paths.

        Returns:
            List[dict]: List of dictionaries containing the confidence scores, bounding box coordinates and class labels.
        """
        results = super().generate(
            image_paths
        )
        
        return results

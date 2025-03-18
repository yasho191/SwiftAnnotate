from PIL import Image
import torch
from typing import List, Tuple
from swiftannotate.image.base import BaseObjectDetection
from swiftannotate.image.utils import setup_logger

logger = setup_logger(__name__)

class YoloWorldForObjectDetection(BaseObjectDetection):
    def __init__(
        self,
        model,
        class_labels: List[str],
        confidence_threshold: float = 0.5,
        validation: bool = False,
        validation_prompt: str | None = None,
        validation_threshold: float | None = None,
        output_file: str | None = None
    ):
        """
        Initialize the YoloWorldObjectDetection class.

        Args:
            model (Any):
                Yolo Object Detection model.
            class_labels (List[str]): 
                List of class labels.
            confidence_threshold (float, optional): 
                Minimum confidence threshold for object detection. 
                Defaults to 0.5.
            validation (bool, optional): 
                Whether to validate annotations from Yolo. 
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
        """
        # TODO: Add model validation
        #
        #
        
        self.model = model
        self.model.eval()
        
        # Set class labels
        self.model.set_classes(class_labels)
        
        super().__init__(
            class_labels,
            confidence_threshold,
            validation,
            validation_prompt,
            validation_threshold,
            output_file
        )
    
    def annotate(self, image: Image.Image) -> dict:
        predictions = self.model(image, conf=self.confidence_threshold)
        result = dict()

        for pred in predictions:
            pred = pred.cpu().numpy()
            result["boxes"] = pred.boxes.xyxy.tolist()
            result["labels"] = [int(x) for x in pred.boxes.cls.tolist()]
            result["scores"] = pred.boxes.conf.tolist()
        
        return result
        
    def validate(self, image, annotations, **kwargs):
        pass
    
    def generate(self, image_paths):
        pass
    
        
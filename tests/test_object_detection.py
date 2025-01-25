import pytest
import os
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection


@pytest.fixture
def setup_owlv2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = model.to(device)
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    class_labels = ["dog", "flower pot", "sink"]
    
    return model, processor, class_labels

def test_owlv2_object_detection(setup_owlv2):
    from swiftannotate.image import OwlV2ForObjectDetection
    model, processor, class_labels = setup_owlv2
    
    test_dir = "tests/images"
    image_paths = [os.path.join(test_dir, image) for image in os.listdir(test_dir)]
    
    detector = OwlV2ForObjectDetection(
        model=model,
        processor=processor,
        class_labels=class_labels,
        output_file="output.json"
    )
    
    results = detector.generate(image_paths)
    
    assert isinstance(results, list)
    assert len(results) == 2
    assert isinstance(results[0], dict)
    assert "annotations" in results[0]
    assert "image_path" in results[0]
    assert os.path.exists("output.json")
    
    os.remove("output.json")


import pytest
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import BitsAndBytesConfig
import os

##############################
#          Qwen2-Vl          #
##############################

@pytest.fixture
def setup_qwen2_model_and_processor():
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
        quantization_config=quantization_config
    )
    
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    return model, processor


def test_qwen2_image_classification(setup_qwen2_model_and_processor):
    from swiftannotate.image import Qwen2VLForImageClassification
    model, processor = setup_qwen2_model_and_processor
    kwargs = {"temperature": 0}
    classification_pipeline = Qwen2VLForImageClassification(
        model=model,
        processor=processor,
        classification_labels=["kitchen", "bottle", "none"],
        output_file="output.json",
    )
    
    test_dir = "tests/images"
    image_paths = [os.path.join(test_dir, image) for image in os.listdir(test_dir)]
    
    results = classification_pipeline.generate(image_paths, **kwargs)
    
    assert isinstance(results, list), "Results should be a list"
    assert isinstance(results[0], dict), "Each result should be a dictionary"
    assert len(results) == len(image_paths), "Number of results should be equal to number of images"
    assert os.path.exists("output.json"), "Output file should be created"
    
    os.remove("output.json")


def test_invalid_classification_labels(setup_qwen2_model_and_processor):
    from swiftannotate.image import Qwen2VLForImageClassification
    model, processor = setup_qwen2_model_and_processor
    
    with pytest.raises(ValueError):
        Qwen2VLForImageClassification(
            model=model,
            processor=processor,
            classification_labels=[],  # Empty labels should raise error
            output_file="output.json",
        )


##############################
#           Ollama           #
##############################


def test_ollama_image_classification():
    from swiftannotate.image import OllamaForImageClassification
    
    classification_pipeline = OllamaForImageClassification(
        caption_model="llava",
        validation_model="llava",
        classification_labels=["kitchen", "bottle", "none"],
        output_file="output.json"
    )
    
    test_dir = "tests/images"
    image_paths = [os.path.join(test_dir, image) for image in os.listdir(test_dir)]
    
    kwargs = {"temperature": 0}
    results = classification_pipeline.generate(image_paths, **kwargs)
    
    assert isinstance(results, list), "Results should be a list"
    assert isinstance(results[0], dict), "Each result should be a dictionary"
    assert len(results) == len(image_paths), "Number of results should be equal to number of images"
    assert os.path.exists("output.json"), "Output file should be created"
    
    os.remove("output.json")
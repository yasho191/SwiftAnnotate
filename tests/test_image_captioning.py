import pytest
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import BitsAndBytesConfig
import os

##############################
#          Qwen2-Vl          #
##############################

@pytest.fixture
def setup_model_and_processor():
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


def test_image_captioning_pipeline_initialization(setup_model_and_processor):
    from swiftannotate.image import Qwen2VLForImageCaptioning
    
    model, processor = setup_model_and_processor
    captioning_pipeline = Qwen2VLForImageCaptioning(
        model=model,
        processor=processor
    )
    assert isinstance(captioning_pipeline, Qwen2VLForImageCaptioning)


def test_image_captioning_generate(setup_model_and_processor):
    from swiftannotate.image import Qwen2VLForImageCaptioning
    
    model, processor = setup_model_and_processor
    captioning_pipeline = Qwen2VLForImageCaptioning(
        model=model,
        processor=processor,
        output_file="output.json"
    )
    
    test_dir = "tests/images"
    image_paths = [os.path.join(test_dir, image) for image in os.listdir(test_dir)]
    
    results = captioning_pipeline.generate(image_paths)
    
    assert isinstance(results, list), "Results should be a list"
    assert isinstance(results[0], dict), "Each result should be a dictionary"
    assert len(results) == len(image_paths), "Number of results should be equal to number of images"
    assert os.path.exists("output.json"), "Output file should be created"
    
    os.remove("output.json")
    
    
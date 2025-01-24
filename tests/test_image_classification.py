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


def test_image_classification_pipeline_initialization(setup_model_and_processor):
    from swiftannotate.image import ImageClassificationQwen2VL
    
    model, processor = setup_model_and_processor
    kwargs = {"temperature": 0}
    classification_pipeline = ImageClassificationQwen2VL(
        model=model,
        processor=processor,
        classification_labels=["cat", "dog", "none"],
        output_file="output.json",
        kwargs=kwargs
    )
    assert isinstance(classification_pipeline, ImageClassificationQwen2VL)


def test_image_classification_generate(setup_model_and_processor):
    from swiftannotate.image import ImageClassificationQwen2VL
    model, processor = setup_model_and_processor
    kwargs = {"temperature": 0}
    classification_pipeline = ImageClassificationQwen2VL(
        model=model,
        processor=processor,
        classification_labels=["kitchen", "bottle", "none"],
        output_file="output.json",
        kwargs=kwargs
    )
    
    test_dir = "tests/images"
    image_paths = [os.path.join(test_dir, image) for image in os.listdir(test_dir)]
    
    results = classification_pipeline.generate(image_paths)
    
    assert isinstance(results, list)
    assert isinstance(results[0], dict)
    assert len(results) == len(image_paths)
    assert os.path.exists("output.json")
    
    os.remove("output.json")


def test_invalid_classification_labels(setup_model_and_processor):
    from swiftannotate.image import ImageClassificationQwen2VL
    model, processor = setup_model_and_processor
    kwargs = {"temperature": 0}
    
    with pytest.raises(ValueError):
        ImageClassificationQwen2VL(
            model=model,
            processor=processor,
            classification_labels=[],  # Empty labels should raise error
            output_file="output.json",
            kwargs=kwargs
        )
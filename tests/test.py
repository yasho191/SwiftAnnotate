import os
from swiftannotate.image import ImageCaptioningQwen2VL
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import BitsAndBytesConfig

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

# Load the model
captioning_pipeline = ImageCaptioningQwen2VL(
    model = model,
    processor = processor
)

# Load the images
BASE_DIR = "/data/yashowardhan/SwiftAnnotate/assets/test"
image_paths = [os.path.join(BASE_DIR, image) for image in os.listdir(BASE_DIR)]

results = captioning_pipeline.generate(image_paths)
# SwiftAnnotate 🚀

![swiftannotate](https://github.com/yasho191/SwiftAnnotate/blob/main/assets/swiftannotate-high-resolution-logo.png?raw=True)

SwiftAnnotate is a comprehensive auto-labeling tool designed for Text, Image, and Video data. It leverages state-of-the-art (SOTA) Vision Language Models (VLMs) and Large Language Models (LLMs) through a robust annotator-validator pipeline, ensuring high-quality, grounded annotations while minimizing hallucinations. SwiftAnnotate also supports annotations tasks like Object Detection and Segmentation through SOTA CV models like `SAM2`, `YOLOWorld`, and `OWL-ViT`.

## Key Features 🎯

1. **Text Processing 📝**  
Perform **classification**, **summarization**, and **text generation** with state-of-the-art NLP models. Solve real-world problems like spam detection, sentiment analysis, and content creation.

2. **Image Analysis 🖼️**  
Generate **captions** for images to provide meaningful descriptions. Classify images into predefined categories with high precision. Detect objects in images using models like **YOLOWorld**. Achieve pixel-perfect segmentation with **SAM2** and **OWL-ViT**.  

3. **Video Processing 🎥**  
Generate captions for videos with **frame-level analysis** and **temporal understanding** Understand video content by detecting scenes and actions effortlessly.  

4. **Quality Assurance ✅**  
Use a **two-stage pipeline** for annotation and validation to ensure high data quality. Validate outputs rigorously to maintain reliability before deployment.  

5. **Multi-modal Support 🌐**  
Seamlessly process **text**, **images**, and **videos** within a unified framework. Combine data types for powerful multi-modal insights and applications.  

6. **Customization 🛠️**
Easily extend and adapt the framework to suit specific project needs. Integrate new models and tasks seamlessly with modular architecture.

7. **Developer-Friendly 👩‍💻👨‍💻**
Easy-to-use package and detailed documentation to get started quickly.

## Installation Guide  

To install **SwiftAnnotate** from PyPI and set up the project environment, follow these steps:  

1. **Install from PyPI**  

    Run the following command to install the package directly:  

    ```bash
    pip install swiftannotate
    ```

2. **For Development (Using Poetry)**  

    If you want to contribute or explore the project codebase ensure you have Poetry installed.  Follow the steps given below:

    ```bash
    git clone https://github.com/yasho191/SwiftAnnotate
    cd SwiftAnnotate
    poetry install
    ```

    You're now ready to explore and develop SwiftAnnotate!  

## Annotator-Validator Pipeline for LLMs and VLMs

![Annotation Pipeline](https://github.com/yasho191/SwiftAnnotate/blob/main/assets/SwiftAnnotatePipeline.png?raw=True)

The annotator-validator pipeline ensures high-quality annotations through a two-stage process:

**Stage 1: Annotation**

- Primary LLM/VLM generates initial annotations
- Configurable model selection (OpenAI, Google Gemini, Anthropic, Mistral, Qwen-VL)

**Stage 2: Validation**

- Secondary model validates initial annotations
- Cross-checks for hallucinations and factual accuracy
- Provides confidence scores and correction suggestions
- Option to regenerate annotations if validation fails
- Structured output format for consistency

**Benefits**

- Reduced hallucinations through 2 stage verification
- Higher annotation quality and consistency
- Automated quality control
- Traceable annotation process

The pipeline can be customized with different model combinations and validation thresholds based on specific use cases.

## Supported Modalities and Tasks

### Text

### Images

#### Captioning

Currently, we support OpenAI, Google-Gemini, Ollama, and Qwen2-VL for image captioning. As Qwen2-VL is not yet available on Ollama it is supported through HuggingFace. To get started quickly refer the code snippets shown below.

**OpenAI**

```python
import os
from swiftannotate.image import OpenAIForImageCaptioning

caption_model = "gpt-4o"
validation_model = "gpt-4o-mini"
api_key = "<YOUR_OPENAI_API_KEY>"
BASE_DIR = "<IMAGE_DIR>"
image_paths = [os.path.join(BASE_DIR, image) for image in os.listdir(BASE_DIR)]

image_captioning_pipeline = ImageCaptioningOpenAI(
    caption_model=caption_model,
    validation_model=validation_model,
    api_key=api_key,
    output_file="image_captioning_output.json"
)

results = image_captioning_pipeline.generate(image_paths=image_paths)
```

**Qwen2-VL**

You can use any version for the Qwen2-VL (7B, 72B) depending on the available resources. vLLM inference is not currently supported but it will be available soon.

```python
import os
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import BitsAndBytesConfig
from swiftannotate.image import Qwen2VLForImageCaptioning

# Load the images
BASE_DIR = "<IMAGE_DIR>"
image_paths = [os.path.join(BASE_DIR, image) for image in os.listdir(BASE_DIR)]

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
    output_file="image_captioning_output.json"
)

results = captioning_pipeline.generate(image_paths)
```

### Videos

## Contributing to SwiftAnnotate 🤝

We welcome contributions to SwiftAnnotate! There are several ways you can help improve the project:

- **Enhanced Prompts**: Contribute better validation and annotation prompts for improved accuracy
- **File Support**: Add support for additional input/output file formats
- **Cloud Integration**: Implement AWS S3 storage support and other cloud services
- **Validation Strategies**: Develop new validation approaches for different annotation tasks
- **Model Support**: Integrate additional LLMs and VLMs
- **Documentation**: Improve guides and examples

Please submit a pull request with your contributions or open an issue to discuss new features.

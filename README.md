# SwiftAnnotate 🚀

![SwiftAnnotate](https://github.com/yasho191/SwiftAnnotate/blob/main/assets/swiftannotate-high-resolution-logo.png)

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

Make sure you have conda installed on your system. To install SwiftAnnotate, follow these steps:

1. **Clone the repository**:

    ```bash
    git clone https://github.com/yasho191/SwiftAnnotate
    ```

2. **Create virtual environment**:

    ```bash
    conda create -n swiftannotate python=3.10
    conda activate swiftannotate
    ```

3. **Navigate to the project directory**:

    ```bash
    cd SwiftAnnotate
    ```

4. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

## Annotator-Validator Pipeline for LLMs and VLMs

![Annotation Pipeline](https://github.com/yasho191/SwiftAnnotate/blob/main/assets/SwiftAnnotatePiepline.png)

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

Currently, we support OpenAI, Google-Gemini, and Qwen2-VL for image captioning.

### Videos


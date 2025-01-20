# SwiftAnnotate

![SwiftAnnotate](https://github.com/yasho191/SwiftAnnotate/blob/main/assets/swiftannotate-high-resolution-logo.png)

SwiftAnnotate is a comprehensive auto-labeling tool designed for Text, Image, and Video data. It leverages state-of-the-art Vision Language Models (VLMs) and Large Language Models (LLMs) through a robust annotator-validator pipeline, ensuring high-quality, grounded annotations while minimizing hallucinations. SwiftAnnotate also supports annotations tasks like Object Detection and Segmentation through SOTA CV models like SAM2, YOLOWorld, and OWL-ViT.

Key Features:

- Text Processing: Classification, Summarization, and Text Generation.
- Image Analysis: Captioning, Classification, Object Detection, and Segmentation using advanced models like SAM2, YOLOWorld, and OWL-ViT.
- Video Processing: Video Captioning using Frame-level analysis and temporal understanding.
- Quality Assurance: Two-stage pipeline with annotation and validation.
- Multi-modal Support: Handles various data types seamlessly.

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

The annotator-validator pipeline ensures high-quality annotations through a two-stage process:

**Stage 1: Annotation**

- Primary LLM/VLM generates initial annotations
- Configurable model selection (OpenAI, Google Gemini, Qwen-VL)

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


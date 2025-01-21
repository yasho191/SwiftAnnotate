import torch
from openai import OpenAI
import google.generativeai as genai
import logging
import json
from tqdm import tqdm
from typing import Tuple, List, Dict
from qwen_vision_utils import process_vision_info
from transformers import AutoProcessor, AutoModelForImageTextToText
from swiftannotate.image.base import BaseImageClassification, ImageValidationOutputOpenAI, ImageValidationOutputGemini
from swiftannotate.constants import BASE_IMAGE_CLASSIFICATION_VALIDATION_PROMPT, BASE_IMAGE_CLASSIFICATION_PROMPT


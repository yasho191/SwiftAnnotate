import base64
import logging
import os
from datetime import datetime

def setup_logger(name: str, level=logging.INFO):
    """Function to set up a logger"""
    # Create log directory if it doesn't exist
    log_dir = os.path.join(os.getcwd(), '.swiftannotate_log')
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log file name with current timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'swiftannotate_{timestamp}.log'
    log_file_path = os.path.join(log_dir, log_file)

    # Create a custom logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file_path)

    # Set level for handlers
    console_handler.setLevel(level)
    file_handler.setLevel(level)

    # Create formatters and add them to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

def encode_image(image_path: str) -> str:
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logging.error(f"Image encoding failed: {e}")
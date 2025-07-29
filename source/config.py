# config.py
from pathlib import Path

MAX_FILE_SIZE = 20 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
RESULTS_DIR = Path("api_results")

MODEL_CONFIG = {
    "blip2": "Salesforce/blip2-opt-2.7b",
    "skin_classifier": "NeuronZero/SkinCancerClassifier",
    "llm": "llama3:8b" 
}

CLASS_MAPPING = {
    "AK": "Ceratose Actínica", "BCC": "Carcinoma Basocelular",
    "BKL": "Ceratose Benigna", "DF": "Dermatofibroma",
    "MEL": "Melanoma", "NV": "Nevo Melanocítico",
    "SCC": "Carcinoma Espinocelular", "VASC": "Lesão Vascular",
    "SEB": "Queratose Seborreica"
}
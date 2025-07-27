# config.py
from pathlib import Path

# Configurações da Aplicação
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
RESULTS_DIR = Path("api_results")

# Configurações dos Modelos de IA
MODEL_CONFIG = {
    "blip2": "Salesforce/blip2-opt-2.7b",
    "skin_classifier": "NeuronZero/SkinCancerClassifier",
    "llm": "meta-llama/Llama-3.2-1B"
}

# Mapeamento de Classes de Lesões
CLASS_MAPPING = {
    "AK": "Ceratose Actínica", "BCC": "Carcinoma Basocelular",
    "BKL": "Ceratose Benigna", "DF": "Dermatofibroma",
    "MEL": "Melanoma", "NV": "Nevo Melanocítico",
    "SCC": "Carcinoma Espinocelular", "VASC": "Lesão Vascular",
    "SEB": "Queratose Seborreica"
}
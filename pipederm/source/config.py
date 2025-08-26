from pathlib import Path

MAX_FILE_SIZE = 20971520
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}  # Removido para evitar conflito
RESULTS_DIR = Path("api_results")

# MODELOS ALTERNATIVOS PARA ANÁLISE DE IMAGENS:
# - llava:13b (mais preciso, mas mais lento)
# - llava:7b (mais rápido, menos preciso)
# - bakllava (alternativa ao LLaVA)
# - llava:latest (versão mais recente)
MODEL_CONFIG = {
    "description_model": "llava:7b",  # Mudança para versão mais precisa
    "skin_classifier": "hasibzunair/melanet",  # MODELO BINÁRIO: benigno/maligno
    "llm": "llama3.1:8b" # Recomenda-se usar um modelo mais capaz para laudos
}

# Configurações adicionais para Ollama
OLLAMA_CONFIG = {
    "base_url": "http://localhost:11434",
    "timeout": 1000,  # Aumentado para modelos maiores
    "max_retries": 5   # Mais tentativas
}

# ADAPTADO: Mapeamento de classes para classificação binária (benigno/maligno)
# Baseado no modelo hasibzunair/melanet que classifica apenas em duas categorias
CLASS_MAPPING = {
    "benign": "Benigno",
    "malignant": "Maligno"
}


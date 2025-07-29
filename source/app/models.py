# app/models.py
import torch
import logging
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration, 
    AutoImageProcessor,
    AutoModelForImageClassification, 
    BitsAndBytesConfig
)
from config import MODEL_CONFIG, CLASS_MAPPING

logger = logging.getLogger(__name__)

class ModelManager:
    """Gerencia o ciclo de vida dos modelos de IA carregados localmente."""
    def __init__(self, device, torch_dtype):
        self.device = device
        self.torch_dtype = torch_dtype
        self.class_mapping = CLASS_MAPPING
        
        # ✅ CORREÇÃO: Configuração de quantização mais estável para evitar erros de geração.
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True  # Essencial para hardware com pouca VRAM
        )
        
        logger.info("⏳ Carregando modelos de visão...")
        
        self.blip2_processor, self.blip2_model = self._load_hf_model(
            "blip2", Blip2Processor, Blip2ForConditionalGeneration, quant_config
        )
        
        self.classifier_processor, self.classifier_model = self._load_hf_model(
            "skin_classifier", AutoImageProcessor, AutoModelForImageClassification, None
        )
        # Move o classificador para o dispositivo correto após o carregamento
        self.classifier_model.to(self.device)
        logger.info(f"Modelo classificador movido para o dispositivo: {self.device}")
        
        logger.info("✅ Modelos de visão carregados. O LLM será acessado via API Ollama.")

    def _load_hf_model(self, model_key, processor_class, model_class, quant_config, **kwargs):
        """Carrega um modelo e seu processador do Hugging Face Hub."""
        try:
            name = MODEL_CONFIG[model_key]
            processor = processor_class.from_pretrained(name)
            
            # Para o classificador, não usamos device_map, pois o movemos manualmente.
            model_args = {}
            if model_key == "blip2":
                model_args["device_map"] = "auto"

            if quant_config:
                model_args["quantization_config"] = quant_config

            model = model_class.from_pretrained(name, **model_args)

            return processor, model
        except Exception as e:
            logger.error(f"Falha ao carregar o modelo {model_key} do Hugging Face: {e}", exc_info=True)
            raise RuntimeError(f"Erro no carregamento do modelo {model_key}")
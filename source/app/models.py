# app/models.py
import torch
import logging
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Blip2Processor,
    Blip2ForConditionalGeneration, AutoImageProcessor,
    AutoModelForImageClassification, BitsAndBytesConfig
)
from config import MODEL_CONFIG, CLASS_MAPPING

logger = logging.getLogger(__name__)

class ModelManager:
    """Gerencia o ciclo de vida e o acesso aos modelos de IA."""
    def __init__(self, device, torch_dtype):
        self.device = device
        self.torch_dtype = torch_dtype
        self.class_mapping = CLASS_MAPPING
        
        quant_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
        
        logger.info("⏳ Carregando modelos de IA...")
        self.blip2_processor, self.blip2_model = self._load_model(
            "blip2", Blip2Processor, Blip2ForConditionalGeneration, quant_config
        )
        self.classifier_processor, self.classifier_model = self._load_model(
            "skin_classifier", AutoImageProcessor, AutoModelForImageClassification, None, device_map="cpu"
        )
        self.llm_tokenizer, self.llm_model = self._load_model(
            "llm", AutoTokenizer, AutoModelForCausalLM, quant_config
        )
        logger.info("✅ Modelos de IA carregados.")

    def _load_model(self, model_key, processor_class, model_class, quant_config, **kwargs):
        """Função genérica para carregar um modelo e seu processador."""
        try:
            name = MODEL_CONFIG[model_key]
            processor = processor_class.from_pretrained(name)
            
            model_args = {"device_map": "auto", **kwargs}
            if quant_config and "llm" in model_key or "blip2" in model_key:
                model_args["quantization_config"] = quant_config
                model_args["torch_dtype"] = self.torch_dtype

            model = model_class.from_pretrained(name, **model_args)
            
            if self.device.type == 'cuda' and "llm" in model_key or "blip2" in model_key:
                model = torch.compile(model)

            return processor, model
        except Exception as e:
            logger.error(f"Falha ao carregar o modelo {model_key}: {e}")
            raise RuntimeError(f"Erro no carregamento do modelo {model_key}")
# app/models.py
import torch
import logging
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration, 
    AutoImageProcessor,
    AutoModelForImageClassification
    # Removido: BitsAndBytesConfig não é mais necessário
)
from config import MODEL_CONFIG, CLASS_MAPPING

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Gerencia o ciclo de vida dos modelos de IA carregados localmente.
    VERSÃO SEM QUANTIZAÇÃO: Modelos carregados com precisão total.
    """
    def __init__(self, device, torch_dtype):
        self.device = device
        self.torch_dtype = torch_dtype
        self.class_mapping = CLASS_MAPPING
                
        logger.info("⏳ Carregando modelos de visão em precisão total...")
        
        self.blip2_processor, self.blip2_model = self._load_hf_model(
            "blip2", Blip2Processor, Blip2ForConditionalGeneration
        )
        
        self.classifier_processor, self.classifier_model = self._load_hf_model(
            "skin_classifier", AutoImageProcessor, AutoModelForImageClassification
        )

        # Move os modelos para o dispositivo correto (GPU ou CPU)
        self.classifier_model.to(self.device)
        logger.info(f"Modelo classificador movido para o dispositivo: {self.device}")
        
        logger.info("✅ Modelos de visão carregados. O LLM será acessado via API Ollama.")

    # 3. Assinatura do método simplificada, sem 'quant_config'.
    def _load_hf_model(self, model_key, processor_class, model_class, **kwargs):
        """Carrega um modelo e seu processador do Hugging Face Hub sem quantização."""
        try:
            name = MODEL_CONFIG[model_key]
            processor = processor_class.from_pretrained(name)
            
            model_args = {
                # Usa o torch_dtype definido na inicialização (ex: torch.float16 para GPU)
                "torch_dtype": self.torch_dtype 
            }

            # Para modelos grandes como o BLIP-2, device_map='auto' é recomendado
            # para distribuir as camadas entre VRAM e RAM se necessário.
            if model_key == "blip2":
                model_args["device_map"] = "auto"
            
            # 4. REMOVIDO: O parâmetro 'quantization_config' não é mais passado.
            model = model_class.from_pretrained(name, **model_args)

            logger.info(f"Modelo '{name}' carregado com sucesso.")
            return processor, model
            
        except Exception as e:
            logger.error(f"Falha ao carregar o modelo {model_key} do Hugging Face: {e}", exc_info=True)
            raise RuntimeError(f"Erro no carregamento do modelo {model_key}")
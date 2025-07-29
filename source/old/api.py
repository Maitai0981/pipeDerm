import os
import io
import time
import gc
import tempfile
import warnings
import argparse
import json
import logging
import re
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use('Agg')

from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from PIL import Image, ImageFile
from concurrent.futures import ThreadPoolExecutor
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Blip2Processor,
    Blip2ForConditionalGeneration,
    AutoImageProcessor,
    AutoModelForImageClassification,
    BitsAndBytesConfig
)

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")

# ========================== #
#   CONFIGURAÇÕES GLOBAIS    #
# ========================== #
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
RESULTS_DIR = Path("api_results")
RESULTS_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DermAI-API")

executor = ThreadPoolExecutor(max_workers=2)
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


# ========================== #
#      GERENCIADOR DE IA     #
# ========================== #
class ModelManager:
    def __init__(self, device, torch_dtype):
        logger.info("🚀 INICIALIZANDO GERENCIADOR DE MODELOS DE IA")
        self.device = device
        self.torch_dtype = torch_dtype
        
        self.blip2_processor, self.blip2_model = self._load_blip2()
        self.classifier_processor, self.classifier_model = self._load_skin_classifier()
        self.llm_tokenizer, self.llm_model = self._load_llm()
        
        self.class_mapping = {
            "AK": "Ceratose Actínica", "BCC": "Carcinoma Basocelular",
            "BKL": "Ceratose Benigna", "DF": "Dermatofibroma",
            "MEL": "Melanoma", "NV": "Nevo Melanocítico",
            "SCC": "Carcinoma Espinocelular", "VASC": "Lesão Vascular",
            "SEB": "Queratose Seborreica"
        }
        
        self._warmup_models()

    def _load_blip2(self):
        try:
            logger.info("⏳ Carregando Salesforce/blip2-opt-2.7b...")
            name = "Salesforce/blip2-opt-2.7b"
            quant_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
            processor = Blip2Processor.from_pretrained(name)
            model = Blip2ForConditionalGeneration.from_pretrained(
                name, torch_dtype=self.torch_dtype, device_map="auto", quantization_config=quant_config
            )
            return processor, torch.compile(model)
        except Exception as e:
            raise RuntimeError(f"Falha ao carregar BLIP-2: {e}")

    def _load_skin_classifier(self):
        try:
            logger.info("⏳ Carregando NeuronZero/SkinCancerClassifier...")
            name = "NeuronZero/SkinCancerClassifier"
            processor = AutoImageProcessor.from_pretrained(name)
            model = AutoModelForImageClassification.from_pretrained(name, device_map="cpu")
            return processor, model
        except Exception as e:
            raise RuntimeError(f"Falha ao carregar classificador de pele: {e}")

    def _load_llm(self):
        try:
            logger.info("⏳ Carregando meta-llama/Llama-3.2-1B...")
            name = "meta-llama/Llama-3.2-1B"
            quant_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
            tokenizer = AutoTokenizer.from_pretrained(name)
            model = AutoModelForCausalLM.from_pretrained(
                name, quantization_config=quant_config, torch_dtype=self.torch_dtype, device_map="auto"
            )
            return tokenizer, torch.compile(model)
        except Exception as e:
            raise RuntimeError(f"Falha ao carregar LLM: {e}")

    def _warmup_models(self):
        logger.info("🔥 Pré-aquecendo modelos para otimizar primeira requisição...")
        try:
            dummy_image = Image.new('RGB', (224, 224), color='white')
            # Warmup BLIP-2
            inputs = self.blip2_processor(dummy_image, "warmup", return_tensors="pt").to(self.device)
            self.blip2_model.generate(**inputs, max_new_tokens=2)
            # Warmup Classifier
            inputs = self.classifier_processor(dummy_image, return_tensors="pt").to("cpu")
            self.classifier_model(**inputs)
            # Warmup LLM
            self.generate_llm_text("Texto de teste.")
            logger.info("✅ Modelos pré-aquecidos e prontos!")
        except Exception as e:
            logger.error(f"Erro durante o pré-aquecimento: {e}", exc_info=True)

    def generate_llm_text(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "Você é DermAI, um assistente IA que gera laudos dermatológicos preliminares."},
            {"role": "user", "content": prompt}
        ]
        inputs = self.llm_tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)
        
        outputs = self.llm_model.generate(
            input_ids=inputs, max_new_tokens=512, do_sample=True, temperature=0.6, top_p=0.9
        )
        response_ids = outputs[0][inputs.shape[-1]:]
        return self.llm_tokenizer.decode(response_ids, skip_special_tokens=True).strip()

def setup_gpu():
    if torch.cuda.is_available():
        logger.info(f"GPU detectada: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
        return torch.device("cuda"), torch.float16
    logger.warning("GPU não detectada, utilizando CPU.")
    return torch.device("cpu"), torch.float32

device, torch_dtype = setup_gpu()
model_manager = ModelManager(device, torch_dtype)

# ========================== #
#      FUNÇÕES CORE          #
# ========================== #
def aplicar_clahe(image: Image.Image) -> Image.Image:
    img_np = np.array(image.convert('RGB'))
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return Image.fromarray(cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB))

def gerar_descricao_imagem(image_path: Path) -> str:
    try:
        image = Image.open(image_path).convert("RGB")
        prompt = (
            "Como um especialista em dermatologia, descreva objetivamente as características visíveis desta "
            "lesão cutânea em um parágrafo conciso. NÃO inclua URLs ou HTML. Descreva apenas morfologia, "
            "cor, bordas e superfície."
        )
        inputs = model_manager.blip2_processor(image, text=prompt, return_tensors="pt").to(device)
        output_tokens = model_manager.blip2_model.generate(**inputs, max_new_tokens=150)
        generated_ids = output_tokens[0][inputs.input_ids.shape[-1]:]
        decoded = model_manager.blip2_processor.decode(generated_ids, skip_special_tokens=True).strip()
        
        is_valid = decoded and not re.search(r'<.*?>|https?://|\[\/?[A-Z]+\]', decoded) and len(decoded.split()) > 5
        return decoded if is_valid else "A descrição visual automática da lesão não pôde ser gerada com segurança."
    except Exception as e:
        logger.error(f"Falha na descrição (BLIP-2): {e}", exc_info=True)
        return "A descrição visual automática da lesão não pôde ser gerada com segurança."

def classificar_lesao(image_path: Path) -> list:
    try:
        with Image.open(image_path) as img:
            image = aplicar_clahe(img.convert("RGB")).resize((224, 224))
        
        processor = model_manager.classifier_processor
        model = model_manager.classifier_model
        model.eval()
        
        inputs = processor(images=image, return_tensors="pt").to(next(model.parameters()).device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probas = torch.nn.functional.softmax(logits, dim=-1)[0]
            top3_conf, top3_idx = torch.topk(probas.cpu(), 3)

        return [{
            "code": model.config.id2label[top3_idx[i].item()],
            "name": model_manager.class_mapping.get(model.config.id2label[top3_idx[i].item()], "Desconhecido"),
            "confidence": top3_conf[i].item()
        } for i in range(3)]
    except Exception as e:
        logger.error(f"Erro na classificação: {e}", exc_info=True)
        return []

def gerar_laudo_clinico(descricao: str, diagnostico: list) -> str:
    if not diagnostico:
        return "Não foi possível gerar um laudo sem um diagnóstico."
    
    desc_laudo = descricao if "não pôde ser gerada" not in descricao else \
        "A descrição visual não foi gerada. A análise prossegue com base apenas na classificação."

    prompt = f"""
    Gere um laudo clínico preliminar estruturado com base nos dados abaixo.
    
    1. **Descrição da Lesão (Análise Visual):**
    "{desc_laudo}"

    2. **Resultados da Classificação por IA:**
    - Diagnóstico Principal: {diagnostico[0]['name']} (Confiança: {diagnostico[0]['confidence']*100:.1f}%)
    - Diagnósticos Alternativos: {[f"{d['name']} ({d['confidence']*100:.1f}%)" for d in diagnostico[1:]]}

    O laudo deve conter as seções: "Descrição Clínica", "Análise Diagnóstica", "Recomendações" (sempre inclua consulta presencial) e "Limitações" (ferramenta de auxílio).
    """
    return model_manager.generate_llm_text(prompt)

def run_analysis_pipeline(image_path: Path) -> dict:
    future_classificacao = executor.submit(classificar_lesao, image_path)
    future_descricao = executor.submit(gerar_descricao_imagem, image_path)

    diagnostico = future_classificacao.result()
    descricao = future_descricao.result()
    laudo = gerar_laudo_clinico(descricao, diagnostico)
    
    RESULTS_DIR.joinpath(f"resultado_{image_path.stem}_{int(time.time())}.json").write_text(
        json.dumps({
            "data_processamento": time.strftime("%d/%m/%Y %H:%M:%S"),
            "diagnosticos": diagnostico,
            "descricao_lesao": descricao,
            "laudo_gerado": laudo
        }, indent=4, ensure_ascii=False)
    )

    return {
        "diagnostico_principal": f"{diagnostico[0]['name']} ({diagnostico[0]['confidence']*100:.1f}%)" if diagnostico else "Erro",
        "diagnosticos_alternativos": [{
            "nome": d.get("name"), "confianca": f"{d.get('confidence', 0.0) * 100:.1f}%"
        } for d in diagnostico[1:]] if diagnostico else [],
        "descricao_lesao": descricao,
        "laudo_completo": laudo
    }

# ========================== #
#        API ENDPOINTS       #
# ========================== #
@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    if "image" not in request.files:
        return jsonify({"error": "Nenhuma imagem enviada"}), 400
        
    image_file = request.files["image"]
    if not image_file or not ('.' in image_file.filename and image_file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS):
        return jsonify({"error": "Arquivo inválido ou não permitido"}), 400
    
    temp_image_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", dir=RESULTS_DIR) as tmp:
            Image.open(image_file.stream).convert("RGB").save(tmp.name, format='JPEG')
            temp_image_path = Path(tmp.name)
        
        result = run_analysis_pipeline(temp_image_path)
        result["tempo_processamento"] = f"{time.time() - start_time:.2f} segundos"
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Erro no endpoint de predição: {e}", exc_info=True)
        return jsonify({"error": f"Erro interno no servidor: {e}"}), 500
    finally:
        if temp_image_path and temp_image_path.exists():
            temp_image_path.unlink()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

@app.route("/system", methods=["GET"])
def system_status():
    status = {
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "memory_used_gb": f"{torch.cuda.memory_allocated() / (1024 ** 3):.2f}" if torch.cuda.is_available() else "N/A",
        "models_loaded": {
            "blip2": model_manager.blip2_model is not None,
            "skin_classifier": model_manager.classifier_model is not None,
            "llm": model_manager.llm_model is not None
        }
    }
    return jsonify(status)

@app.route("/results/<filename>")
def get_result(filename):
    return send_from_directory(RESULTS_DIR, filename, max_age=3600)

# ========================== #
#      ROTINA DE LIMPEZA     #
# ========================== #
def periodic_cleanup():
    logger.info("Executando limpeza de resultados antigos...")
    try:
        for file in RESULTS_DIR.glob("*"):
            if file.is_file() and time.time() - file.stat().st_mtime > 86400: # 24 horas
                file.unlink()
                logger.info(f"Removido arquivo antigo: {file.name}")
    except Exception as e:
        logger.error(f"Erro na limpeza periódica: {e}")

# ========================== #
#         INICIALIZAÇÃO      #
# ========================== #
if __name__ == "__main__":
    if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            scheduler = BackgroundScheduler()
            scheduler.add_job(func=periodic_cleanup, trigger="interval", hours=6)
            scheduler.start()
            logger.info("Agendador de limpeza de arquivos iniciado.")
        except ImportError:
            logger.warning("APScheduler não instalado. Limpeza periódica desativada.")
    
    parser = argparse.ArgumentParser(description="DermAI API Server")
    parser.add_argument("--port", type=int, default=5000, help="Porta para execução do servidor")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port)
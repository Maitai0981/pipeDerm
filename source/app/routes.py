# app/routes.py
import time
import gc
import torch
import tempfile
from pathlib import Path
from flask import (
    Blueprint, request, jsonify, current_app, render_template, send_from_directory
)
from app.app.services import run_analysis_pipeline
from config import ALLOWED_EXTENSIONS, RESULTS_DIR

api_bp = Blueprint('api', __name__)

@api_bp.route("/")
def home():
    """Renderiza a página inicial de exemplo."""
    return render_template('index.html')

@api_bp.route('/predict', methods=['POST'])
def predict():
    """Endpoint para receber uma imagem e retornar a análise."""
    start_time = time.time()
    
    if "image" not in request.files:
        return jsonify({"error": "Nenhum arquivo de imagem enviado"}), 400
        
    image_file = request.files["image"]
    filename = image_file.filename
    if not filename or not ('.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS):
        return jsonify({"error": "Tipo de arquivo inválido ou não permitido"}), 400
    
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", dir=RESULTS_DIR) as tmp:
            temp_path = Path(tmp.name)
            image_file.save(temp_path)

        # Acessa o model_manager através do 'current_app'
        result = run_analysis_pipeline(temp_path, current_app.model_manager)
        result["tempo_processamento"] = f"{time.time() - start_time:.2f} segundos"
        return jsonify(result), 200
        
    except Exception as e:
        current_app.logger.error(f"Erro no endpoint de predição: {e}", exc_info=True)
        return jsonify({"error": f"Erro interno no servidor: {e}"}), 500
        
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

@api_bp.route("/system", methods=["GET"])
def system_status():
    """Retorna o status do sistema e dos modelos de IA."""
    mm = current_app.model_manager
    status = {
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "models_loaded": {
            "blip2": mm.blip2_model is not None,
            "skin_classifier": mm.classifier_model is not None,
            "llm": mm.llm_model is not None
        }
    }
    return jsonify(status)

@api_bp.route("/results/<filename>")
def get_result_file(filename):
    """Serve os arquivos de resultado (JSON) salvos."""
    return send_from_directory(RESULTS_DIR, filename)
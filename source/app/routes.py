# app/routes.py
import time
from pathlib import Path
from flask import (
    Blueprint, request, jsonify, current_app, render_template, send_from_directory
)
import torch
# 1. Importar a classe de serviço, não mais a função individual
from app.services import DermatologyService
from config import ALLOWED_EXTENSIONS, RESULTS_DIR

main_bp = Blueprint('main', __name__)
api_bp = Blueprint('api', __name__)

def _is_allowed_file(filename: str) -> bool:
    """Verifica se a extensão do arquivo é permitida."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@main_bp.route("/")
def home():
    """Renderiza a página inicial de exemplo."""
    return render_template('index.html')

@api_bp.route('/predict', methods=['POST'])
def predict():
    """Endpoint para receber uma imagem e retornar a análise."""
    if "image" not in request.files:
        return jsonify({"error": "Nenhum arquivo de imagem enviado"}), 400
        
    image_file = request.files["image"]
    if not image_file.filename or not _is_allowed_file(image_file.filename):
        return jsonify({"error": "Tipo de arquivo inválido ou não permitido"}), 400
    
    start_time = time.time()
    try:
        # 2. Instanciar o serviço a partir do model_manager da aplicação
        model_manager = current_app.model_manager
        service = DermatologyService(model_manager)

        # 3. Chamar o método da instância de serviço, passando o stream de bytes
        #    Não é mais necessário salvar o arquivo temporário manualmente aqui.
        #    O serviço pode lidar com o objeto de imagem diretamente.
        result = service.run_analysis_pipeline(image_file)
        
        # Adicionar metadados da requisição ao resultado
        result["tempo_processamento"] = f"{time.time() - start_time:.2f} segundos"
        
        return jsonify(result), 200
        
    except Exception as e:
        current_app.logger.error(f"Erro no endpoint de predição: {e}", exc_info=True)
        return jsonify({"error": f"Erro interno no servidor: {str(e)}"}), 500
    # 4. Bloco 'finally' removido por ser redundante

# ... (outras rotas como system_status e get_result_file permanecem iguais)
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
            # "llm" foi removido pois é via Ollama, não um modelo carregado
        }
    }
    return jsonify(status)

@api_bp.route("/results/<filename>")
def get_result_file(filename):
    """Serve os arquivos de resultado (JSON) salvos."""
    return send_from_directory(RESULTS_DIR, filename)
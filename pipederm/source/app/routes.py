# app/routes.py
import time
import hashlib
import mimetypes
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
from flask import ( Blueprint, request, jsonify, current_app, render_template, send_from_directory  )
from app.services import DermatologyService
from app.utils import generate_request_id, validate_image, ValidationError
from config import RESULTS_DIR

# Importações para o endpoint de status, se mantido aqui
import torch
import ollama
from config import OLLAMA_CONFIG, MODEL_CONFIG

# Constantes
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}

def get_remote_address():
    """Obtém o endereço IP remoto da requisição."""
    if request.headers.get('X-Forwarded-For'):
        return request.headers.get('X-Forwarded-For').split(',')[0]
    return request.remote_addr


main_bp = Blueprint('main', __name__)
api_bp = Blueprint('api', __name__)

def _is_allowed_file(filename: str) -> bool:
    """Verifica se a extensão do arquivo é permitida."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def _validate_image_content(image_file) -> tuple[bool, str]:
    """Validação do conteúdo da imagem para segurança."""
    try:
        # Verificar se é uma imagem válida
        img = Image.open(image_file.stream)
        img.verify()  # Verifica se é uma imagem válida
        
        # Verificar dimensões mínimas e máximas
        img = Image.open(image_file.stream)  # Reabrir após verify()
        width, height = img.size
        
        if width < 50 or height < 50:
            return False, "Imagem muito pequena. Mínimo: 50x50 pixels"
        
        if width > 4096 or height > 4096:
            return False, "Imagem muito grande. Máximo: 4096x4096 pixels"
        
        # Verificar se não é uma imagem vazia ou muito simples
        img_array = np.array(img.convert('RGB'))
        if img_array.std() < 5:  # Muito pouca variação
            return False, "Imagem com muito pouca variação (possivelmente vazia)"
        
        return True, "Conteúdo da imagem válido"
        
    except Exception as e:
        return False, f"Erro na validação do conteúdo: {str(e)}"

def _validate_image_file(file) -> tuple[bool, str]:
    """Validação rigorosa do arquivo de imagem."""
    if not file or not file.filename:
        return False, "Nenhum arquivo enviado"
    
    # Verificar extensão
    if not _is_allowed_file(file.filename):
        return False, f"Tipo de arquivo não permitido. Tipos aceitos: {', '.join(ALLOWED_EXTENSIONS)}"
    
    # Verificar tamanho (20MB máximo)
    file.seek(0, 2)  # Ir para o final do arquivo
    size = file.tell()
    file.seek(0)  # Voltar ao início
    
    if size > 20 * 1024 * 1024:  # 20MB
        return False, "Arquivo muito grande. Máximo permitido: 20MB"
    
    # Verificar MIME type
    mime_type, _ = mimetypes.guess_type(file.filename)
    if mime_type and not mime_type.startswith('image/'):
        return False, "Arquivo não é uma imagem válida"
    
    # Verificar se é um arquivo real (não vazio)
    if size == 0:
        return False, "Arquivo vazio"
    
    # Validação do conteúdo da imagem
    is_valid_content, content_message = _validate_image_content(file)
    if not is_valid_content:
        return False, content_message
    
    return True, "Arquivo válido"

def _generate_request_id() -> str:
    """Gera um ID único para a requisição."""
    timestamp = str(time.time())
    remote_addr = get_remote_address()
    return hashlib.md5(f"{timestamp}{remote_addr}".encode()).hexdigest()[:8]

@main_bp.route("/")
def home():
    """Renderiza a página inicial."""
    # O conteúdo da documentação OpenAPI/Swagger foi mantido
    return render_template('index.html')

@api_bp.route('/predict', methods=['POST'])
def predict():
    """Endpoint para análise dermatológica de imagens."""
    request_id = generate_request_id()
    current_app.logger.info(f"[{request_id}] Iniciando análise de imagem")

    if "image" not in request.files:
        # A exceção será capturada pelo manipulador de erros global
        raise ValidationError("Nenhum arquivo de imagem enviado no campo 'image'.")
    
    image_file = request.files["image"]
    
    # O bloco try/except agora é muito mais limpo
    try:
        # Valida e retorna os bytes da imagem
        image_bytes = validate_image(image_file)
        
        start_time = time.time()
        service = DermatologyService(current_app.model_manager)
        
        # O serviço recebe os bytes para evitar reabrir o arquivo
        result = service.run_analysis_pipeline(image_bytes)
        
        if "error" in result:
            return jsonify({**result, "request_id": request_id}), 500

        result["tempo_processamento"] = f"{time.time() - start_time:.2f} segundos"
        result["request_id"] = request_id

        
        current_app.logger.info(f"[{request_id}] Análise concluída com sucesso.")
        return jsonify(result), 200

    except ValidationError as e:
        current_app.logger.warning(f"[{request_id}] Erro de validação: {e}")
        # Delega a formatação da resposta para um error handler global
        raise e
    except Exception as e:
        current_app.logger.error(f"[{request_id}] Erro inesperado no endpoint: {e}", exc_info=True)
        # Delega para o error handler de erro 500
        raise

@api_bp.route("/system", methods=["GET"])
def system_status():
    """
    Retorna o status do sistema e dos modelos de IA.
    
    Este endpoint verifica a disponibilidade de todos os componentes do sistema:
    - GPU e configurações CUDA
    - Modelos de IA carregados
    - Status do Ollama e modelos disponíveis
    - Informações do sistema
    
    ---
    tags:
      - Sistema
    summary: Status do sistema
    description: |
      Verifica disponibilidade de todos os componentes:
      - **GPU**: Detecção e configurações CUDA
      - **Modelos**: Status dos modelos de IA carregados
      - **Ollama**: Conectividade e modelos disponíveis
      - **Sistema**: Informações gerais do servidor
    responses:
      200:
        description: Status completo do sistema
        schema:
          type: object
          properties:
            gpu_available:
              type: boolean
              description: Se GPU CUDA está disponível
              example: true
            gpu_name:
              type: string
              description: Nome da GPU detectada
              example: "NVIDIA GeForce RTX 3080"
            models_loaded:
              type: object
              description: Status dos modelos carregados
              properties:
                skin_classifier:
                  type: boolean
                  description: Se o classificador de pele está carregado
                  example: true
            ollama_status:
              type: object
              description: Status do Ollama
              properties:
                service_online:
                  type: boolean
                  description: Se o Ollama está rodando
                  example: true
                models:
                  type: object
                  description: Modelos disponíveis no Ollama
                  properties:
                    llm:
                      type: boolean
                      description: Se o modelo LLM está disponível
                      example: true
                    description:
                      type: boolean
                      description: Se o modelo de descrição está disponível
                      example: true
            system_info:
              type: object
              description: Informações do sistema
              properties:
                memory_usage:
                  type: string
                  description: Uso de memória
                  example: "N/A"
                uptime:
                  type: string
                  description: Tempo de funcionamento
                  example: "N/A"
        examples:
          application/json:
            gpu_available: true
            gpu_name: "NVIDIA GeForce RTX 3080"
            models_loaded:
              skin_classifier: true
            ollama_status:
              service_online: true
              models:
                llm: true
                description: true
            system_info:
              memory_usage: "N/A"
              uptime: "N/A"
    """
    mm = current_app.model_manager
    status = {
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "models_loaded": {
            "skin_classifier": mm.classifier_model is not None,
        },
        "ollama_status": {},
        "system_info": {
            "memory_usage": "N/A",
            "uptime": "N/A"
        }
    }
    
    # Verificar status do Ollama
    try:
        client = ollama.Client(host=OLLAMA_CONFIG["base_url"])
        client.list() # Testa a conexão
        status["ollama_status"]["service_online"] = True
        
        available_models = [m["name"] for m in client.list().get("models", [])]
        status["ollama_status"]["models"] = {
            "llm": MODEL_CONFIG["llm"] in available_models,
            "description": MODEL_CONFIG["description_model"] in available_models
        }
    except Exception as e:
        status["ollama_status"]["service_online"] = False
        status["ollama_status"]["models"] = {}
        status["ollama_status"]["error"] = str(e)

    return jsonify(status)

@api_bp.route("/results/<string:filename>")
def get_result_file(filename):
    """Serve os arquivos de resultado (JSON) salvos de forma segura."""
    safe_filename = secure_filename(filename)
    if not safe_filename or safe_filename != filename:
        return jsonify({"error": "Nome de arquivo inválido."}), 400
    
    return send_from_directory(RESULTS_DIR, safe_filename)

@api_bp.route("/metrics", methods=["GET"])
def get_metrics():
    """
    Retorna métricas de performance do sistema.
    
    Este endpoint fornece estatísticas em tempo real sobre o uso do sistema:
    - Número total de requisições
    - Taxa de sucesso
    - Tempo médio de processamento
    - Tempo de funcionamento
    
    ---
    tags:
      - Sistema
    summary: Métricas do sistema
    description: |
      Retorna estatísticas de performance e uso do sistema:
      - **Requisições**: Total de requisições processadas
      - **Erros**: Total de erros ocorridos
      - **Sucesso**: Taxa de sucesso em porcentagem
      - **Performance**: Tempo médio de processamento
      - **Uptime**: Tempo de funcionamento do servidor
    responses:
      200:
        description: Métricas de performance do sistema
        schema:
          type: object
          properties:
            total_requests:
              type: integer
              description: Número total de requisições processadas
              example: 150
            total_errors:
              type: integer
              description: Número total de erros ocorridos
              example: 3
            success_rate:
              type: number
              description: Taxa de sucesso em porcentagem
              example: 98.0
            avg_processing_time:
              type: number
              description: Tempo médio de processamento em segundos
              example: 2.34
            uptime_seconds:
              type: number
              description: Tempo de funcionamento em segundos
              example: 86400
        examples:
          application/json:
            total_requests: 150
            total_errors: 3
            success_rate: 98.0
            avg_processing_time: 2.34
            uptime_seconds: 86400
    """
    return jsonify(current_app.metrics.get_stats())

@api_bp.route("/model-info", methods=["GET"])
def model_info():
    """
    Retorna informações sobre o modelo de classificação.
    
    ---
    tags:
      - Sistema
    summary: Informações do modelo
    description: Retorna detalhes sobre o modelo melaNet usado para classificação
    responses:
      200:
        description: Informações do modelo
        schema:
          type: object
          properties:
            model_name:
              type: string
              description: Nome do modelo
              example: "melaNet"
            version:
              type: string
              description: Versão do modelo
              example: "1.0"
            type:
              type: string
              description: Tipo de classificação
              example: "binary_classification"
            classes:
              type: array
              items:
                type: string
              description: Classes de classificação
              example: ["benign", "malignant"]
            input_shape:
              type: array
              items:
                type: integer
              description: Formato de entrada
              example: [224, 224, 3]
            is_simulated:
              type: boolean
              description: Se o modelo é simulado
              example: false
    """
    try:
        model_manager = current_app.model_manager
        if hasattr(model_manager.classifier_model, 'get_model_info'):
            info = model_manager.classifier_model.get_model_info()
        else:
            info = {
                "model_name": "unknown",
                "version": "unknown",
                "type": "unknown",
                "classes": list(model_manager.class_mapping.keys()),
                "input_shape": [224, 224, 3],
                "is_simulated": True
            }
        return jsonify(info)
    except Exception as e:
        return jsonify({
            "error": f"Erro ao obter informações do modelo: {str(e)}"
        }), 500

@api_bp.route("/health", methods=["GET"])
def health_check():
    """
    Endpoint de health check para monitoramento.
    
    ---
    tags:
      - Sistema
    summary: Health check
    description: Verifica se o serviço está funcionando
    responses:
      200:
        description: Serviço funcionando
        schema:
          type: object
          properties:
            status:
              type: string
            timestamp:
              type: number
            version:
              type: string
    """
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0"
    })
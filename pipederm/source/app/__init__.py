# app/__init__.py
import torch
import logging
import os
import time
from flask import Flask, request, g
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flasgger import Swagger
from config import RESULTS_DIR
from app.models import ModelManager
from app.utils import setup_periodic_cleanup

# Métricas básicas
class Metrics:
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.processing_times = []
        self.start_time = time.time()
    
    def record_request(self, processing_time=None):
        self.request_count += 1
        if processing_time:
            self.processing_times.append(processing_time)
    
    def record_error(self):
        self.error_count += 1
    
    def get_stats(self):
        avg_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        return {
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "success_rate": ((self.request_count - self.error_count) / self.request_count * 100) if self.request_count > 0 else 0,
            "avg_processing_time": avg_time,
            "uptime_seconds": time.time() - self.start_time
        }

def create_app():
    """Cria e configura a instância da aplicação Flask."""
    # Configurar logging primeiro
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # O diretório de templates agora é relativo à pasta 'app'
    app = Flask(__name__, template_folder='templates')
    
    # Configuração do Swagger
    swagger_config = {
        "headers": [],
        "specs": [
            {
                "endpoint": 'apispec_1',
                "route": '/apispec_1.json',
                "rule_filter": lambda rule: True,  # all in
                "model_filter": lambda tag: True,  # all in
            }
        ],
        "static_url_path": "/flasgger_static",
        "swagger_ui": True,
        "specs_route": "/apidocs/"
    }
    
    swagger_template = {
        "swagger": "2.0",
        "info": {
            "title": "PipeDerm API",
            "description": "API de análise dermatológica com Inteligência Artificial",
            "contact": {
                "name": "Suporte PipeDerm",
                "email": "suporte@pipederm.com"
            },
            "version": "1.0.0"
        },
        "host": "localhost:5000",
        "basePath": "/api",
        "schemes": ["http", "https"],
        "consumes": ["application/json", "multipart/form-data"],
        "produces": ["application/json"],
        "tags": [
            {
                "name": "Análise",
                "description": "Endpoints para análise de imagens dermatológicas"
            },
            {
                "name": "Sistema",
                "description": "Endpoints para monitoramento e status do sistema"
            },
            {
                "name": "Resultados",
                "description": "Endpoints para acesso aos resultados salvos"
            }
        ]
    }
    
    # Inicializar Swagger
    swagger = Swagger(app, config=swagger_config, template=swagger_template)
    
    # Configuração de CORS (libera para qualquer origem em dev ou via env)
    # Use a variável de ambiente CORS_ALLOWED_ORIGINS para restringir em produção
    allowed_origins_env = os.environ.get("CORS_ALLOWED_ORIGINS", "*")
    cors_origins = "*" if allowed_origins_env.strip() == "*" else [o.strip() for o in allowed_origins_env.split(",") if o.strip()]

    CORS(app, resources={
        r"/api/*": {
            "origins": cors_origins,
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type"],
            "supports_credentials": False,
        }
    })

    # Configuração de rate limiting
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=["200 per day", "50 per hour"],
        storage_uri="memory://"
    )
    
    # Rate limits específicos para endpoints críticos
    @limiter.limit("10 per minute")
    def api_limit():
        pass
    
    # Inicializar métricas
    app.metrics = Metrics()
    
    # Middleware para capturar métricas
    @app.before_request
    def before_request():
        g.start_time = time.time()
    
    @app.after_request
    def after_request(response):
        if hasattr(g, 'start_time'):
            processing_time = time.time() - g.start_time
            app.metrics.record_request(processing_time)
            
            if response.status_code >= 400:
                app.metrics.record_error()
        
        return response
    
    # Criar diretório de resultados
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Configurar dispositivo de forma otimizada
    if torch.cuda.is_available():
        app.device = torch.device("cuda")
        app.torch_dtype = torch.float16
        logging.info(f"GPU detectada: {torch.cuda.get_device_name(0)}")
    else:
        app.device = torch.device("cpu")
        app.torch_dtype = torch.float32
        logging.info("Usando CPU - Performance otimizada")

    # Carrega os modelos de IA dentro do contexto da aplicação
    with app.app_context():
        app.model_manager = ModelManager(app.device, app.torch_dtype)

    # Importa e registra os dois blueprints
    from app.routes import main_bp, api_bp
    app.register_blueprint(main_bp)  # Registra na raiz (/)
    app.register_blueprint(api_bp, url_prefix='/api') # Registra com prefixo /api

    # Inicia a limpeza periódica se não estiver em modo de depuração
    if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        setup_periodic_cleanup()

    return app
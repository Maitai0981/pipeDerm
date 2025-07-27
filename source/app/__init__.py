# app/__init__.py
import torch
import logging
import os
from flask import Flask
from flask_cors import CORS
from config import RESULTS_DIR
from app.models import ModelManager
from app.utils import setup_periodic_cleanup

def create_app():
    """Cria e configura a instância da aplicação Flask."""
    # O diretório de templates agora é relativo à pasta 'app'
    app = Flask(__name__, template_folder='templates')
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    RESULTS_DIR.mkdir(exist_ok=True)
    
    if torch.cuda.is_available():
        app.device = torch.device("cuda")
        app.torch_dtype = torch.float16
        logging.info(f"GPU detectada: {torch.cuda.get_device_name(0)}")
    else:
        app.device = torch.device("cpu")
        app.torch_dtype = torch.float32
        logging.warning("GPU não detectada, utilizando CPU.")

    # Carrega os modelos de IA dentro do contexto da aplicação
    with app.app_context():
        app.model_manager = ModelManager(app.device, app.torch_dtype)

    # Importa e registra os dois blueprints
    from app import routes
    app.register_blueprint(routes.main_bp)  # Registra na raiz (/)
    app.register_blueprint(routes.api_bp, url_prefix='/api') # Registra com prefixo /api

    # Inicia a limpeza periódica se não estiver em modo de depuração
    if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        setup_periodic_cleanup()

    return app
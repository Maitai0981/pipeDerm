# app/__init__.py
import torch
import logging
import os
from flask import Flask
from flask_cors import CORS
from config import RESULTS_DIR
from app.models import ModelManager
from app.app.utils import setup_periodic_cleanup

def create_app():
    """Cria e configura a instância da aplicação Flask."""
    app = Flask(__name__, template_folder='templates')
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Cria diretório de resultados se não existir
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Configuração de dispositivo e tipo de dado
    if torch.cuda.is_available():
        app.device = torch.device("cuda")
        app.torch_dtype = torch.float16
        logging.info(f"GPU detectada: {torch.cuda.get_device_name(0)}")
    else:
        app.device = torch.device("cpu")
        app.torch_dtype = torch.float32
        logging.warning("GPU não detectada, utilizando CPU.")

    # Carrega os modelos de IA e anexa à instância do app
    # Isso garante que os modelos sejam carregados apenas uma vez
    with app.app_context():
        app.model_manager = ModelManager(app.device, app.torch_dtype)

    # Registra as rotas da aplicação
    from app.app import routes
    app.register_blueprint(routes.api_bp, url_prefix='/api')

    # Configura a limpeza periódica
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
        setup_periodic_cleanup()

    return app
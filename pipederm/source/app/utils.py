# app/utils.py
import io
import time
import logging
import hashlib
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from flask import request
from werkzeug.datastructures import FileStorage
from apscheduler.schedulers.background import BackgroundScheduler
from config import RESULTS_DIR

# Constantes
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}

logger = logging.getLogger(__name__)


# ==============================================================================
# Exceção Customizada
# ==============================================================================

class ValidationError(ValueError):
    """Exceção customizada para erros de validação da aplicação."""
    pass


# ==============================================================================
# Seção 1: Validação de Arquivos
# ==============================================================================

def validate_image(file: FileStorage):
    """
    Executa uma validação completa e eficiente em um arquivo de imagem.
    Lê o arquivo em memória apenas uma vez e levanta ValidationError em caso de falha.
    
    Retorna os bytes da imagem se a validação for bem-sucedida.
    """
    if not file or not file.filename:
        raise ValidationError("Nenhum arquivo enviado.")

    # Verifica a extensão do nome do arquivo
    if '.' not in file.filename or \
       file.filename.rsplit('.', 1)[1].lower() not in ALLOWED_EXTENSIONS:
        raise ValidationError(f"Tipo de arquivo não permitido. Aceitos: {', '.join(ALLOWED_EXTENSIONS)}")

    # Lê o arquivo em memória para evitar múltiplas leituras do stream
    file_bytes = file.read()
    file_size = len(file_bytes)

    if file_size == 0:
        raise ValidationError("Arquivo vazio.")
    
    if file_size > 20 * 1024 * 1024:  # 20MB
        raise ValidationError("Arquivo muito grande. Máximo permitido: 20MB.")
    
    # Validação de conteúdo usando Pillow
    try:
        # Abre a imagem a partir dos bytes lidos
        with Image.open(io.BytesIO(file_bytes)) as img:
            img.verify()  # Verifica a integridade dos headers da imagem

        # Reabre para verificar metadados e conteúdo real
        with Image.open(io.BytesIO(file_bytes)) as img:
            if not img.format or img.format.lower() not in ALLOWED_EXTENSIONS:
                raise ValidationError("Conteúdo do arquivo não corresponde à extensão permitida.")

            width, height = img.size
            if width < 50 or height < 50:
                raise ValidationError("Imagem muito pequena. Mínimo: 50x50 pixels.")
            if width > 4096 or height > 4096:
                raise ValidationError("Imagem muito grande. Máximo: 4096x4096 pixels.")

            # Converte para array numpy para análise estatística
            img_array = np.array(img.convert('RGB'))
            if img_array.std() < 5:
                raise ValidationError("Imagem com pouca variação de cor (possivelmente inválida ou vazia).")

    except (UnidentifiedImageError, IOError, SyntaxError):
        raise ValidationError("O arquivo enviado não é uma imagem válida ou está corrompido.")
    except Exception as e:
        logger.error(f"Erro inesperado na validação da imagem: {e}", exc_info=True)
        raise ValidationError("Não foi possível processar a imagem.")

    return file_bytes


# ==============================================================================
# Seção 2: Processamento de Imagem
# ==============================================================================

def aplicar_clahe(image: Image.Image) -> Image.Image:
    """Aplica o Contraste Adaptativo de Histograma (CLAHE) à imagem."""
    img_np = np.array(image.convert('RGB'))
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    
    lab_clahe = cv2.merge((cl, a_channel, b_channel))
    rgb_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    
    return Image.fromarray(rgb_clahe)


# ==============================================================================
# Seção 3: Utilitários da Aplicação Web
# ==============================================================================

def generate_request_id() -> str:
    """Gera um ID único para a requisição baseado no tempo e IP."""
    timestamp = str(time.time())
    # Usa 'request.remote_addr' que é mais padrão que get_remote_address
    remote_addr = request.remote_addr or "local"
    return hashlib.md5(f"{timestamp}{remote_addr}".encode()).hexdigest()[:8]


def setup_periodic_cleanup():
    """Configura e inicia uma rotina para limpar arquivos de resultados antigos."""
    def cleanup_task():
        logger.info("Executando limpeza de resultados antigos...")
        cutoff = time.time() - 86400  # 24 horas atrás
        
        try:
            for f in RESULTS_DIR.glob("*.json"):
                if f.is_file() and f.stat().st_mtime < cutoff:
                    f.unlink()
                    logger.info(f"Removido arquivo antigo: {f.name}")
        except Exception as e:
            logger.error(f"Erro durante a tarefa de limpeza: {e}", exc_info=True)

    try:
        scheduler = BackgroundScheduler(daemon=True)
        scheduler.add_job(func=cleanup_task, trigger="interval", hours=6)
        scheduler.start()
        logger.info("Agendador de limpeza de arquivos configurado e iniciado.")
    except ImportError:
        logger.warning("APScheduler não instalado. Limpeza periódica de arquivos desativada.")
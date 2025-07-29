# app/utils.py
import cv2
import time
import numpy as np
from PIL import Image
from pathlib import Path
from config import RESULTS_DIR
from apscheduler.schedulers.background import BackgroundScheduler
import logging

logger = logging.getLogger(__name__)

def aplicar_clahe(image: Image.Image) -> Image.Image:
    """Aplica o Contraste Adaptativo de Histograma (CLAHE) à imagem."""
    img_np = np.array(image.convert('RGB'))
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab_clahe = cv2.merge((cl, a, b))
    return Image.fromarray(cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB))

def setup_periodic_cleanup():
    """Configura e inicia uma rotina para limpar arquivos antigos."""
    def cleanup_task():
        logger.info("Executando limpeza de resultados antigos...")
        for f in RESULTS_DIR.glob("*.json"):
            if f.is_file() and time.time() - f.stat().st_mtime > 86400: # 24h
                f.unlink()
                logger.info(f"Removido arquivo antigo: {f.name}")

    try:
        scheduler = BackgroundScheduler()
        scheduler.add_job(func=cleanup_task, trigger="interval", hours=6)
        scheduler.start()
        logger.info("Agendador de limpeza de arquivos iniciado.")
    except ImportError:
        logger.warning("APScheduler não instalado. Limpeza periódica desativada.")
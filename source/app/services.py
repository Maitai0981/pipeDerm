# app/services.py
import torch
import re
import json
import time
from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from app.utils import aplicar_clahe
from config import RESULTS_DIR
import logging

logger = logging.getLogger(__name__)
executor = ThreadPoolExecutor(max_workers=2)

def generate_llm_text(prompt: str, model_manager) -> str:
    """Gera texto usando o modelo LLM."""
    messages = [
        {"role": "system", "content": "Você é DermAI, um assistente IA para laudos dermatológicos."},
        {"role": "user", "content": prompt}
    ]
    inputs = model_manager.llm_tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model_manager.device)
    
    outputs = model_manager.llm_model.generate(
        input_ids=inputs, max_new_tokens=512, do_sample=True, temperature=0.6, top_p=0.9
    )
    response_ids = outputs[0][inputs.shape[-1]:]
    return model_manager.llm_tokenizer.decode(response_ids, skip_special_tokens=True).strip()

def gerar_descricao_imagem(image: Image.Image, model_manager) -> str:
    """Gera uma descrição textual da imagem da lesão usando BLIP-2."""
    prompt = ("Descreva objetivamente esta lesão cutânea. Foque em morfologia, cor, bordas e superfície. "
              "Não use HTML ou URLs.")
    inputs = model_manager.blip2_processor(image, text=prompt, return_tensors="pt").to(model_manager.device)
    output_tokens = model_manager.blip2_model.generate(**inputs, max_new_tokens=150)
    generated_ids = output_tokens[0][inputs.input_ids.shape[-1]:]
    decoded = model_manager.blip2_processor.decode(generated_ids, skip_special_tokens=True).strip()
    
    is_valid = decoded and not re.search(r'<.*?>|https?://', decoded) and len(decoded.split()) > 5
    return decoded if is_valid else "A descrição visual automática não pôde ser gerada com segurança."

def classificar_lesao(image: Image.Image, model_manager) -> list:
    """Classifica a lesão de pele, retornando os 3 principais diagnósticos."""
    processed_image = aplicar_clahe(image).resize((224, 224))
    model = model_manager.classifier_model
    inputs = model_manager.classifier_processor(images=processed_image, return_tensors="pt")
    
    with torch.no_grad():
        logits = model(**inputs).logits
        probas = torch.nn.functional.softmax(logits, dim=-1)[0]
        top3_conf, top3_idx = torch.topk(probas, 3)

    return [{
        "name": model_manager.class_mapping.get(model.config.id2label[idx.item()], "Desconhecido"),
        "confidence": conf.item()
    } for idx, conf in zip(top3_idx, top3_conf)]

def run_analysis_pipeline(image_path: Path, model_manager) -> dict:
    """Orquestra a pipeline de análise completa da imagem."""
    try:
        with Image.open(image_path).convert("RGB") as img:
            future_classificacao = executor.submit(classificar_lesao, img, model_manager)
            future_descricao = executor.submit(gerar_descricao_imagem, img, model_manager)
            
            diagnostico = future_classificacao.result()
            descricao = future_descricao.result()
    except Exception as e:
        logger.error(f"Erro ao processar imagem em paralelo: {e}")
        return {"error": "Falha na análise da imagem."}

    desc_laudo = descricao if "não pôde ser gerada" not in descricao else "Descrição visual indisponível."
    prompt_laudo = f"""
    Gere um laudo clínico preliminar com base nos dados:
    1. Descrição: "{desc_laudo}"
    2. Diagnósticos: {[f"{d['name']} ({d['confidence']*100:.1f}%)" for d in diagnostico]}
    O laudo deve ter seções de "Descrição Clínica", "Análise Diagnóstica", "Recomendações" e "Limitações".
    """
    laudo = generate_llm_text(prompt_laudo, model_manager)

    result = {
        "diagnostico_principal": f"{diagnostico[0]['name']} ({diagnostico[0]['confidence']*100:.1f}%)" if diagnostico else "N/A",
        "diagnosticos_alternativos": [{
            "nome": d['name'], "confianca": f"{d['confidence']*100:.1f}%"
        } for d in diagnostico[1:]],
        "descricao_lesao": descricao,
        "laudo_completo": laudo
    }

    # Salvar resultado em JSON
    report_path = RESULTS_DIR / f"report_{image_path.stem}_{int(time.time())}.json"
    report_path.write_text(json.dumps(result, indent=4, ensure_ascii=False))

    return result
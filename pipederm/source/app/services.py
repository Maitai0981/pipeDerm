# app/services.py
import torch
import re
import json
import time
import logging
import ollama
from PIL import Image
from pathlib import Path
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
from werkzeug.datastructures import FileStorage
from typing import Union
from app.utils import aplicar_clahe
from config import RESULTS_DIR, MODEL_CONFIG

logger = logging.getLogger(__name__)

def clear_cuda_cache(func):
    """Decorator para limpar o cache da VRAM da GPU após a execução da função."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return wrapper

class DermatologyService:
    """Encapsula toda a lógica de análise dermatológica."""

    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.llm_model_name = MODEL_CONFIG.get("llm", "llama3")

    def _generate_ollama_text(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = ollama.chat(
                model=self.llm_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options={"temperature": 0.5, "top_p": 0.9}
            )
            return response['message']['content'].strip()
        except Exception as e:
            logger.error(f"Falha ao se comunicar com a API do Ollama: {e}", exc_info=True)
            return "Erro: Falha ao se comunicar com o modelo de linguagem (Ollama)."

    @clear_cuda_cache
    def _gerar_descricao_imagem(self, image: Image.Image) -> str:
        """Gera uma descrição textual da imagem da lesão, com decodificação robusta."""
        prompt = "Descreva objetivamente as características visíveis desta lesão cutânea em um parágrafo conciso. Foque apenas em morfologia, cor, bordas e superfície. Não inclua URLs ou HTML."
        
        try:
            inputs = self.model_manager.blip2_processor(
                images=image, text=prompt, return_tensors="pt"
            ).to(self.model_manager.device)
            
            output_tokens = self.model_manager.blip2_model.generate(
                **inputs, max_new_tokens=150, do_sample=True, temperature=0.6, top_p=0.9
            )

            input_length = inputs.input_ids.shape[1]
            new_tokens = output_tokens[0, input_length:]
            decoded = self.model_manager.blip2_processor.decode(new_tokens, skip_special_tokens=True).strip()

            logger.info(f"Resposta bruta do BLIP-2: '{decoded}'")

            if not decoded or re.search(r'<.*?>|https?://', decoded) or len(decoded.split()) <= 5:
                logger.warning("Validação da descrição falhou. Resposta foi inválida ou muito curta.")
                return "A descrição visual automática não pôde ser gerada com segurança."
            
            return decoded

        except Exception as e:
            logger.error(f"Falha na geração de descrição (BLIP-2): {e}", exc_info=True)
            return "A descrição visual automática não pôde ser gerada com segurança."

    @clear_cuda_cache
    def _classificar_lesao(self, image: Image.Image) -> list:
        """Classifica a lesão, garantindo a compatibilidade de tipos de dados."""
        try:
            classifier_device = next(self.model_manager.classifier_model.parameters()).device
            # ✅ SOLUÇÃO PARA O ERRO DE TIPO: Obter o dtype (ex: float16) do modelo
            model_dtype = next(self.model_manager.classifier_model.parameters()).dtype

            processed_image = aplicar_clahe(image).resize((224, 224))
            inputs = self.model_manager.classifier_processor(images=processed_image, return_tensors="pt").to(classifier_device)
            
            # ✅ SOLUÇÃO PARA O ERRO DE TIPO: Converter o tensor de entrada para o mesmo dtype do modelo
            inputs['pixel_values'] = inputs['pixel_values'].to(model_dtype)
            
            with torch.no_grad():
                logits = self.model_manager.classifier_model(**inputs).logits
                probas = torch.nn.functional.softmax(logits, dim=-1)[0]
                top3_conf, top3_idx = torch.topk(probas, 3)

            return [
                {"name": self.model_manager.class_mapping.get(self.model_manager.classifier_model.config.id2label[idx.item()], "Desconhecido"),
                 "confidence": conf.item()}
                for idx, conf in zip(top3_idx, top3_conf)
            ]
        except Exception as e:
            logger.error(f"Erro na classificação da lesão: {e}", exc_info=True)
            return []

    def _build_report_prompt(self, desc_laudo: str, diagnostico: list) -> tuple[str, str]:
        diag_principal_str = f"{diagnostico[0]['name']} (Confiança: {diagnostico[0]['confidence']*100:.1f}%)"
        diag_alt_str = ", ".join([f"{d['name']} ({d['confidence']*100:.1f}%)" for d in diagnostico[1:]]) if len(diagnostico) > 1 else "Nenhum"
        
        system_prompt = "Você é DermAI, um assistente de IA que gera laudos dermatológicos preliminares, seguindo estritamente a estrutura solicitada."
        user_prompt = f"""
        Com base nos dados a seguir, gere um laudo dermatológico preliminar conciso e profissional. Preencha cada seção de forma clara.

        **Dados Fornecidos:**
        - Descrição da Lesão: "{desc_laudo}"
        - Diagnóstico Principal por IA: {diag_principal_str}
        - Diagnósticos Alternativos por IA: {diag_alt_str}

        **Estrutura do Laudo (Preencha as seções a seguir):**
        - **Descrição Clínica:** (Redija um parágrafo baseado na descrição da lesão fornecida.)
        - **Análise Diagnóstica Automatizada:** (Apresente os resultados da IA. Ex: "A análise por IA sugere como principal hipótese diagnóstica um {diagnostico[0]['name']} com alta confiança. Lesões como {diag_alt_str} foram consideradas com menor probabilidade.")
        - **Recomendações:** (Escreva um texto padrão recomendando fortemente a consulta com um dermatologista para avaliação clínica e confirmação diagnóstica.)
        - **Limitações:** (Escreva um texto padrão informando que este é um exame de triagem por IA, não substitui a avaliação médica e o diagnóstico definitivo depende de um profissional qualificado.)
        """
        return system_prompt, user_prompt.strip()

    def _save_report(self, result: dict, image_name: str):
        report_path = RESULTS_DIR / f"report_{image_name}_{int(time.time())}.json"
        try:
            report_path.write_text(json.dumps(result, indent=4, ensure_ascii=False))
            logger.info(f"Relatório salvo em: {report_path}")
        except Exception as e:
            logger.error(f"Falha ao salvar o relatório em {report_path}: {e}", exc_info=True)

    def run_analysis_pipeline(self, image_source: Union[FileStorage, Path]) -> dict:
        image_name = "unknown"
        try:
            if isinstance(image_source, Path):
                image_name = image_source.stem
                img = Image.open(image_source).convert("RGB")
            else:
                image_name = Path(image_source.filename).stem
                img = Image.open(image_source.stream).convert("RGB")

            with img:
                future_classificacao = self.executor.submit(self._classificar_lesao, img)
                future_descricao = self.executor.submit(self._gerar_descricao_imagem, img)
                diagnostico = future_classificacao.result()
                descricao = future_descricao.result()
        except Exception as e:
            logger.error(f"Erro ao processar imagem: {e}", exc_info=True)
            return {"error": "Falha na análise inicial da imagem."}

        if not diagnostico:
            return {"error": "A classificação da lesão falhou."}

        desc_laudo = descricao if "não pôde ser gerada" not in descricao else "A descrição visual não pôde ser gerada. A análise prossegue com base apenas na classificação."
        
        system_prompt, user_prompt = self._build_report_prompt(desc_laudo, diagnostico)
        laudo = self._generate_ollama_text(system_prompt, user_prompt)

        result = {
            "diagnostico_principal": f"{diagnostico[0]['name']} (Confiança: {diagnostico[0]['confidence']*100:.1f}%)",
            "diagnosticos_alternativos": [{"nome": d['name'], "confianca": f"{d['confidence']*100:.1f}%"} for d in diagnostico[1:]],
            "descricao_lesao": descricao,
            "laudo_completo": laudo
        }
        
        self._save_report(result, image_name)
        return result
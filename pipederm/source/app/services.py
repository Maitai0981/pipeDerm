# app/services.py
import torch
import json
import time
import logging
import ollama
import io
import base64
from PIL import Image
from pathlib import Path
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
from werkzeug.datastructures import FileStorage
from typing import Union, List
from app.utils import aplicar_clahe
from config import RESULTS_DIR, MODEL_CONFIG, OLLAMA_CONFIG

logger = logging.getLogger(__name__)

def clear_cuda_cache(func):
    """Decorator para limpar o cache da VRAM após execução."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return wrapper

class DermatologyService:
    """Serviço de análise dermatológica com tratamento robusto de erros."""

    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.executor = ThreadPoolExecutor(max_workers=2)
        # ADAPTADO: Nomes dos modelos buscados da config
        self.llm_model_name = MODEL_CONFIG.get("llm", "llama3.1:8b")
        self.description_model_name = MODEL_CONFIG.get("description_model", "llava:7b")
        
        self.ollama_available = self._verify_ollama_connection()

    def _verify_ollama_connection(self) -> bool:
        """Verifica se o Ollama está acessível e os modelos necessários estão disponíveis."""
        try:
            client = ollama.Client(host=OLLAMA_CONFIG["base_url"])
            models_info = client.list()
            
            # Tratar diferentes formatos de resposta da API do Ollama
            available_models = []
            if hasattr(models_info, 'models') and models_info.models:
                # Formato: ListResponse com atributo models
                available_models = [m.model for m in models_info.models if hasattr(m, 'model')]
            elif isinstance(models_info, dict) and "models" in models_info:
                # Formato: {"models": [{"name": "model1"}, {"name": "model2"}]}
                available_models = [m.get("name", "") for m in models_info.get("models", []) if m.get("name")]
            elif isinstance(models_info, list):
                # Formato: [{"name": "model1"}, {"name": "model2"}]
                available_models = [m.get("name", "") for m in models_info if m.get("name")]
            else:
                logger.error(f"Formato inesperado do Ollama: {type(models_info)}")
                logger.error(f"Conteúdo: {models_info}")
                return False
            
            logger.info(f"Modelos disponíveis no Ollama: {available_models}")
            
            # ADAPTADO: Verifica todos os modelos Ollama necessários
            required_models = {self.llm_model_name, self.description_model_name}
            missing_models = []
            
            for model_name in required_models:
                if model_name not in available_models:
                    missing_models.append(model_name)
                    logger.warning(f"Modelo {model_name} não encontrado no Ollama. Tentando baixar...")
                    try:
                        client.pull(model_name)
                        logger.info(f"Modelo {model_name} baixado com sucesso.")
                    except Exception as e:
                        logger.error(f"Falha ao baixar modelo {model_name}: {e}")
                else:
                    logger.info(f"Modelo {model_name} disponível no Ollama.")
            
            if missing_models:
                logger.warning(f"Modelos não disponíveis: {missing_models}")
                return False
                
            return True
                    
        except Exception as e:
            logger.error(f"Erro na verificação do Ollama: {e}")
            logger.error("Certifique-se de que o Ollama está rodando: ollama serve")
            return False

    def _create_optimized_image(self, image: Image.Image) -> str:
        """Cria versão otimizada da imagem para LLaVA e converte para base64."""
        try:
            # Otimizar tamanho da imagem se muito grande
            max_size = 800
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                logger.info(f"Imagem redimensionada para: {image.size}")
            
            # Converter para base64 com qualidade otimizada
            with io.BytesIO() as buffer:
                # Usar JPEG com qualidade alta para imagens médicas
                image.save(buffer, format="JPEG", quality=95, optimize=True)
                image_bytes = buffer.getvalue()
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                
            logger.info(f"Imagem otimizada: {len(image_base64)} caracteres base64")
            return image_base64
            
        except Exception as e:
            logger.error(f"Erro ao otimizar imagem: {e}")
            raise

    def _gerar_descricao_imagem(self, image: Image.Image) -> str:
        """Gera uma descrição textual da imagem usando LLaVA via Ollama."""
        logger.info(f"Iniciando geração de descrição com modelo: {self.description_model_name}")
        
        # Verificar se Ollama está disponível
        if not self.ollama_available:
            logger.warning("Ollama não disponível")
            return "Erro: Sistema de descrição de imagem não disponível. Ollama não está conectado."
               
        # PROMPT MELHORADO E MAIS ESPECÍFICO PARA LLaVA
        prompt = """INSTRUÇÃO: Você é um dermatologista especialista. Analise esta imagem dermatológica e forneça uma descrição técnica OBJETIVA seguindo EXATAMENTE esta estrutura:

**MORFOLOGIA:** [Descreva a forma geométrica: circular, oval, irregular, lobulada, linear]
**DIMENSÕES:** [Estime o tamanho relativo: pequena (<5mm), média (5-15mm), grande (>15mm)]
**COLORAÇÃO:** [Cor predominante e variações: homogênea, heterogênea, tons específicos]
**BORDAS:** [Definição: bem delimitadas, mal delimitadas, regulares, irregulares]
**SUPERFÍCIE:** [Textura: lisa, rugosa, escamosa, crostosa, ulcerada, verrucosa]
**ELEVAÇÃO:** [Altura: plana, elevada, deprimida, nodular]
**DISTRIBUIÇÃO:** [Padrão: única, múltipla, agrupada, linear, anular]
**CARACTERÍSTICAS ESPECIAIS:** [Assimetria, variação cromática, evolução aparente]

IMPORTANTE: 
- Use APENAS termos dermatológicos precisos
- Seja OBJETIVO, sem interpretações
- Limite cada item a 1-2 palavras técnicas
- NÃO sugira diagnósticos
- Descreva SOMENTE o que vê visualmente"""

        try:
            # Otimizar imagem para LLaVA
            image_base64 = self._create_optimized_image(image)
            
            # Configurar cliente Ollama
            client = ollama.Client(
                host=OLLAMA_CONFIG["base_url"],
                timeout=OLLAMA_CONFIG["timeout"]
            )
            
            # Verificar se modelo está disponível
            models = client.list()
            available_models = []
            if hasattr(models, 'models') and models.models:
                # Formato: ListResponse com atributo models
                available_models = [m.model for m in models.models if hasattr(m, 'model')]
            elif isinstance(models, dict) and "models" in models:
                available_models = [m.get("name", "") for m in models.get("models", []) if m.get("name")]
            elif isinstance(models, list):
                available_models = [m.get("name", "") for m in models if m.get("name")]
            
            if self.description_model_name not in available_models:
                logger.warning(f"Modelo {self.description_model_name} não disponível")
                logger.info("Tentando baixar modelo...")
                try:
                    client.pull(self.description_model_name)
                    logger.info(f"Modelo {self.description_model_name} baixado com sucesso")
                except Exception as pull_error:
                    logger.error(f"Falha ao baixar modelo: {pull_error}")
                    return "Erro: Modelo LLaVA não disponível e não foi possível baixá-lo."
            
            # Fazer múltiplas tentativas com configurações otimizadas
            configs = [
                {"temperature": 0.1, "top_p": 0.7, "num_predict": 350, "repeat_penalty": 1.1},
                {"temperature": 0.2, "top_p": 0.8, "num_predict": 300, "repeat_penalty": 1.2},
                {"temperature": 0.0, "top_p": 0.6, "num_predict": 250, "repeat_penalty": 1.15}
            ]
            
            for attempt, config in enumerate(configs):
                try:
                    logger.info(f"Tentativa {attempt + 1} com config: {config}")
                    
                    response = client.chat(
                        model=self.description_model_name,
                        messages=[{
                            'role': 'user',
                            'content': prompt,
                            'images': [image_base64]
                        }],
                        options=config
                    )
                    
                    decoded = response['message']['content'].strip()
                    
                    # Validação melhorada da resposta
                    if self._validate_llava_response(decoded):
                        logger.info("✅ Descrição LLaVA válida gerada com sucesso")
                        return decoded
                    
                    # Se veio algum conteúdo aproveitável, retorne como não estruturado em vez de falhar
                    if decoded and len(decoded.strip()) >= 30:
                        logger.warning("Descrição LLaVA não estruturada, retornando conteúdo bruto")
                        return f"Descrição (não estruturada):\n{decoded}"
                    
                    logger.warning(f"Resposta inválida na tentativa {attempt + 1}")
                    
                except Exception as e:
                    logger.warning(f"Tentativa {attempt + 1} falhou: {e}")
                    if attempt < len(configs) - 1:
                        time.sleep(1)
                        continue
            
            # Se todas as tentativas falharam
            logger.error("Todas as tentativas com LLaVA falharam")
            # Fallback: descrição mínima padronizada
            return (
                "Descrição (fallback): Não foi possível estruturar a descrição visual automaticamente. "
                "A análise segue baseada na classificação por IA e avaliação clínica é recomendada."
            )
            
        except Exception as e:
            logger.error(f"Erro geral na geração de descrição: {e}")
            return "Erro: Falha geral no sistema de descrição de imagem."

    def _validate_llava_response(self, response: str) -> bool:
        """Valida se a resposta do LLaVA é adequada e segue a estrutura esperada."""
        if not response or len(response.strip()) < 50:
            return False
        
        # Verificar se contém a estrutura esperada
        required_sections = [
            'MORFOLOGIA:', 'DIMENSÕES:', 'COLORAÇÃO:', 'BORDAS:', 
            'SUPERFÍCIE:', 'ELEVAÇÃO:', 'DISTRIBUIÇÃO:'
        ]
        
        response_upper = response.upper()
        found_sections = sum(1 for section in required_sections if section in response_upper)
        
        if found_sections < 5:  # Pelo menos 5 das 7 seções obrigatórias
            logger.warning(f"Estrutura incompleta: {found_sections}/7 seções encontradas")
            return False
        
        # Verificar palavras-chave dermatológicas específicas
        keywords = [
            'irregular', 'regular', 'homogênea', 'heterogênea', 'delimitada',
            'elevada', 'plana', 'lisa', 'rugosa', 'escamosa', 'nodular',
            'circular', 'oval', 'lobulada', 'crostosa', 'ulcerada'
        ]
        
        response_lower = response.lower()
        found_keywords = sum(1 for kw in keywords if kw in response_lower)
        
        if found_keywords < 3:
            logger.warning(f"Poucos termos técnicos encontrados: {found_keywords}")
            return False
        
        # Verificar se não contém frases de erro
        error_phrases = [
            "não posso ver", "não consigo ver", "unable to see",
            "cannot see", "error", "erro", "falha", "impossível"
        ]
        
        for phrase in error_phrases:
            if phrase in response_lower:
                logger.warning(f"Frase de erro detectada: {phrase}")
                return False
        
        logger.info(f"Resposta validada: {found_sections}/7 seções, {found_keywords} termos técnicos")
        return True

    def _generate_fallback_report(self, diagnostico: list, descricao: str) -> str:
        """Gera um laudo básico quando o LLM não está disponível."""
        if not diagnostico:
            return "Erro: Não foi possível realizar a classificação da lesão."
        
        principal = diagnostico[0]
        confianca = principal['confidence'] * 100
        
        # ADAPTADO: Para classificação binária, destacamos se é benigno ou maligno
        resultado_binario = "BENIGNO" if principal['name'] == "Benigno" else "MALIGNO"
        urgencia = "URGENTE" if principal['name'] == "Maligno" else "ROTINA"
        
        laudo = f"""**LAUDO DERMATOLÓGICO PRELIMINAR (MODO BÁSICO)**

**DESCRIÇÃO CLÍNICA:**
{descricao}

**ANÁLISE COMPUTACIONAL:**
• Classificação: {resultado_binario}
• Nível de Confiança: {confianca:.1f}%
• Método: Classificação Binária por Inteligência Artificial (melaNet)
• Prioridade: {urgencia}

**RECOMENDAÇÕES MÉDICAS:**
• Consulta dermatológica presencial obrigatória para confirmação diagnóstica
• {'Avaliação URGENTE recomendada' if principal['name'] == 'Maligno' else 'Avaliação de rotina recomendada'}
• Seguimento médico conforme orientação especializada

**LIMITAÇÕES TÉCNICAS:**
• Análise automatizada em modo básico (sistema avançado indisponível)
• Classificação binária simplificada (benigno/maligno)
• Resultado preliminar, não substitui avaliação médica
• Precisão diagnóstica limitada sem correlação clínica

**DISCLAIMER:**
Este laudo é gerado por sistema de IA para fins de triagem inicial. A confirmação diagnóstica requer avaliação médica presencial por dermatologista qualificado."""

        return laudo

    def _generate_ollama_text(self, system_prompt: str, user_prompt: str) -> str:
        """Gera texto (laudo) usando Ollama com retry e timeout."""
        if not self.ollama_available:
            logger.warning("Ollama não disponível, usando laudo básico")
            return "Erro: Sistema de geração de laudos não disponível. Verifique o serviço Ollama."
        
        for attempt in range(OLLAMA_CONFIG["max_retries"]):
            try:
                client = ollama.Client(
                    host=OLLAMA_CONFIG["base_url"],
                    timeout=OLLAMA_CONFIG["timeout"]
                )
                response = client.chat(
                    model=self.llm_model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    options={
                        "temperature": 0.3, 
                        "top_p": 0.85, 
                        "num_predict": 600,
                        "repeat_penalty": 1.1,
                        "top_k": 40
                    }
                )
                return response['message']['content'].strip()
            except Exception as e:
                logger.warning(f"Tentativa {attempt + 1} de gerar laudo falhou: {e}")
                if attempt == OLLAMA_CONFIG["max_retries"] - 1:
                    logger.error("Todas as tentativas de comunicação com Ollama falharam.")
                    return "Erro: Não foi possível gerar o laudo. Verifique o serviço Ollama."
                time.sleep(2)

    @clear_cuda_cache  
    def _classificar_lesao(self, image: Image.Image) -> list:
        """Classifica a lesão com o modelo binário melaNet."""
        try:
            if not hasattr(self.model_manager, 'classifier_model') or self.model_manager.classifier_model is None:
                raise RuntimeError("Modelo classificador não está carregado")
            
            model = self.model_manager.classifier_model
            
            # ADAPTADO: Usar a implementação real do melaNet
            if hasattr(model, 'predict'):
                # Modelo melaNet real
                result = model.predict(image)
                
                # Converter resultado para formato esperado
                results = []
                for class_name, confidence in result['probabilities'].items():
                    mapped_name = self.model_manager.class_mapping.get(class_name, class_name)
                    results.append({"name": mapped_name, "confidence": confidence})
                
                # Ordenar por confiança (maior primeiro)
                results.sort(key=lambda x: x['confidence'], reverse=True)
                
            else:
                # Fallback para modelo Hugging Face
                processed_image = aplicar_clahe(image).resize((224, 224))
                inputs = self.model_manager.classifier_processor(images=processed_image, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
                    top2_scores, top2_indices = torch.topk(probabilities, 2)

                results = []
                for idx, score in zip(top2_indices, top2_scores):
                    label = model.config.id2label[idx.item()]
                    class_name = self.model_manager.class_mapping.get(label, label)
                    results.append({"name": class_name, "confidence": score.item()})
            
            logger.info(f"Classificação binária concluída: {results[0]['name']} ({results[0]['confidence']:.2%})")
            return results
            
        except Exception as e:
            logger.error(f"Erro na classificação: {e}", exc_info=True)
            return []

    def _build_report_prompt(self, desc_laudo: str, diagnostico: list) -> tuple[str, str]:
        """Constrói prompts otimizados para geração do laudo binário."""
        if not diagnostico: 
            return "", ""
            
        diag_principal = f"{diagnostico[0]['name']} (Confiança: {diagnostico[0]['confidence']*100:.1f}%)"
        # ADAPTADO: Para classificação binária, sempre temos apenas uma alternativa
        diag_alt = f"{diagnostico[1]['name']} ({diagnostico[1]['confidence']*100:.1f}%)" if len(diagnostico) > 1 else "Nenhuma hipótese alternativa identificada"
        
        # SYSTEM PROMPT MAIS ESPECÍFICO E DETALHADO
        system_prompt = """IDENTIDADE: Você é DermAI, um sistema especializado em gerar laudos dermatológicos preliminares padronizados.

DIRETRIZES OBRIGATÓRIAS:
1. Use EXCLUSIVAMENTE terminologia médica dermatológica precisa
2. Mantenha estrutura EXATA conforme especificado
3. Seja CONCISO: máximo 2-3 frases por seção
4. SEMPRE enfatize natureza preliminar e necessidade de avaliação médica
5. NÃO faça diagnósticos definitivos
6. Use linguagem técnica mas acessível
7. NUNCA omita seções obrigatórias

FORMATO DE SAÍDA OBRIGATÓRIO:
- Usar exatamente os cabeçalhos especificados
- Cada seção deve ter conteúdo substantivo
- Manter consistência terminológica
- Incluir todas as limitações e disclaimers"""

        # USER PROMPT MAIS ESTRUTURADO E ESPECÍFICO
        user_prompt = f"""DADOS PARA ANÁLISE:

**DESCRIÇÃO MORFOLÓGICA DA LESÃO:**
{desc_laudo}

**RESULTADO DA CLASSIFICAÇÃO IA:**
• Hipótese Principal: {diag_principal}
• Hipóteses Alternativas: {diag_alt}

GERE UM LAUDO SEGUINDO EXATAMENTE ESTA ESTRUTURA:

**DESCRIÇÃO CLÍNICA:**
[Reformule a descrição morfológica em linguagem médica padronizada, destacando características relevantes para diagnóstico diferencial. Máximo 3 frases.]

**ANÁLISE COMPUTACIONAL:**
[Apresente os resultados da IA de forma técnica, incluindo níveis de confiança e metodologia utilizada. Mencione que é análise preliminar. Máximo 3 frases.]

**CORRELAÇÃO CLÍNICO-PATOLÓGICA:**
[Relacione achados morfológicos com hipóteses diagnósticas, destacando características sugestivas e diagnósticos diferenciais relevantes. Máximo 3 frases.]

**RECOMENDAÇÕES MÉDICAS:**
[Especifique encaminhamentos necessários: dermatologista, exames complementares se indicados, seguimento. Seja específico sobre urgência se aplicável. Máximo 3 frases.]

**LIMITAÇÕES E DISCLAIMER:**
[Enfatize que é análise de IA para triagem, não substitui avaliação médica, necessidade de confirmação diagnóstica presencial. Máximo 2 frases.]

INSTRUÇÕES ESPECÍFICAS:
- Use terminologia dermatológica precisa (ex: mácula, pápula, nódulo, etc.)
- Inclua sempre a expressão "análise preliminar por IA"
- Mencione "confirmação diagnóstica requer avaliação médica presencial"
- Mantenha tom profissional e técnico
- NÃO use bullet points dentro das seções"""

        return system_prompt, user_prompt.strip()

    def _save_report(self, result: dict, image_name: str):
        """Salva o relatório em arquivo JSON."""
        timestamp = int(time.time())
        report_path = RESULTS_DIR / f"report_{image_name}_{timestamp}.json"
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"Relatório salvo: {report_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar relatório: {e}")

    def run_analysis_pipeline(self, image_source: Union[FileStorage, Path]) -> dict:
        """Pipeline principal de análise."""
        image_name = "unknown"
        try:
            if isinstance(image_source, Path):
                image_name = image_source.stem
                img = Image.open(image_source).convert("RGB")
            
            elif isinstance(image_source, bytes):
                image_name = "upload"
                img = Image.open(io.BytesIO(image_source)).convert("RGB")

            elif isinstance(image_source, FileStorage):
                image_name = Path(image_source.filename).stem if image_source.filename else "upload"
                img = Image.open(image_source.stream).convert("RGB")

            else:
                raise TypeError(f"Tipo de fonte de imagem não suportado: {type(image_source)}")

            logger.info(f"Iniciando análise da imagem: {image_name}")

            with img:
                future_class = self.executor.submit(self._classificar_lesao, img)
                future_desc = self.executor.submit(self._gerar_descricao_imagem, img)
                # classificação deve ser rápida; descrição pode demorar
                diagnostico = future_class.result(timeout=1000)
                try:
                    descricao = future_desc.result(timeout=1000)
                except Exception as e:
                    logger.warning(f"Timeout/erro ao gerar descrição: {e}")
                    descricao = "Erro: Não foi possível gerar descrição da imagem no tempo limite."

        except Exception as e:
            logger.error(f"Erro no processamento da imagem: {e}", exc_info=True)
            return {"error": f"Falha no processamento: {str(e)}"}

        if not diagnostico:
            return {"error": "Falha na classificação da lesão."}

        desc_laudo = descricao if "Erro" not in descricao else "Descrição visual não disponível - análise baseada apenas em classificação automatizada."
        
        # Tentar gerar laudo completo se Ollama estiver disponível
        if self.ollama_available:
            system_prompt, user_prompt = self._build_report_prompt(desc_laudo, diagnostico)
            laudo = self._generate_ollama_text(system_prompt, user_prompt) if system_prompt and user_prompt else "Erro na geração do laudo."
            if laudo.startswith("Erro"):
                # Fallback automático para laudo básico
                laudo = self._generate_fallback_report(diagnostico, desc_laudo)
        else:
            # Usar laudo básico se Ollama não estiver disponível
            laudo = self._generate_fallback_report(diagnostico, desc_laudo)

        # ADAPTADO: Para classificação binária, simplificamos o resultado
        resultado_binario = "BENIGNO" if diagnostico[0]['name'] == "Benigno" else "MALIGNO"
        urgencia = "URGENTE" if diagnostico[0]['name'] == "Maligno" else "ROTINA"
        
        result = {
            "classificacao": resultado_binario,
            "confianca": f"{diagnostico[0]['confidence']*100:.1f}%",
            "prioridade": urgencia,
            "diagnostico_principal": f"{diagnostico[0]['name']} (Confiança: {diagnostico[0]['confidence']*100:.1f}%)",
            "diagnostico_alternativo": f"{diagnostico[1]['name']} (Confiança: {diagnostico[1]['confidence']*100:.1f}%)" if len(diagnostico) > 1 else "N/A",
            "descricao_lesao": descricao,
            "laudo_completo": laudo,
            "ollama_available": self.ollama_available,
            "modelo_utilizado": "hasibzunair/melanet (Classificação Binária)"
        }
        
        self.model_manager.clear_cache()
        self._save_report(result, image_name)
        
        logger.info(f"Análise concluída para: {image_name}")
        return result
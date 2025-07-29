# describe_image.py (versão final robusta)
import torch
import argparse
import logging
from PIL import Image
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    BitsAndBytesConfig
)

# Configuração básica de logging para acompanhar o processo
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def describe_with_blip2(image_path: str):
    """
    Carrega o modelo BLIP-2, processa uma imagem local e retorna sua descrição.

    Args:
        image_path (str): O caminho para o arquivo de imagem.

    Returns:
        str: A descrição gerada para a imagem, ou None em caso de erro.
    """
    try:
        # 1. Definição do dispositivo e do modelo
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "Salesforce/blip2-opt-2.7b"
        
        logging.info(f"Usando dispositivo: {device}")
        
        # 2. Configuração de Quantização
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )

        # 3. Carregamento do Modelo e Processador
        logging.info(f"Carregando o modelo '{model_name}'...")
        processor = Blip2Processor.from_pretrained(model_name)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto"
        )
        logging.info("Modelo carregado com sucesso.")

        # 4. Preparação da Imagem e do Prompt
        image = Image.open(image_path).convert("RGB")
        prompt = "Descreva objetivamente as características visíveis desta lesão cutânea em um parágrafo conciso. Foque apenas em morfologia, cor, bordas e superfície."

        logging.info("Processando a imagem e gerando a descrição...")
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

        # 5. Geração da Descrição
        output_tokens = model.generate(
            **inputs, 
            max_new_tokens=150,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )

        # 6. Decodificação do Resultado
        input_token_len = inputs.input_ids.shape[1]
        generated_tokens = output_tokens[0][input_token_len:]
        description = processor.decode(generated_tokens, skip_special_tokens=True).strip()
        
        return description

    except FileNotFoundError:
        logging.error(f"Erro: O arquivo de imagem não foi encontrado em '{image_path}'")
        return None
    except Exception as e:
        logging.error(f"Ocorreu um erro inesperado: {e}", exc_info=True)
        return None
    finally:
        # É uma boa prática limpar a memória da GPU ao final de um script independente
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info("Cache da GPU limpo.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera a descrição de uma imagem de pele usando o modelo BLIP-2.")
    parser.add_argument("image_path", type=str, help="O caminho completo para a imagem a ser analisada.")
    
    args = parser.parse_args()
    
    final_description = describe_with_blip2(args.image_path)
    
    print("\n" + "="*50)
    # ================== #
    #  INÍCIO DA MELHORIA #
    # ================== #

    # 7. Fornecer feedback claro se a descrição estiver vazia
    if final_description:
        print("Descrição da Imagem Gerada pelo BLIP-2:")
        print("="*50)
        print(final_description)
    else:
        print("Falha na Geração da Descrição.")
        print("="*50)
        print("O modelo não retornou uma descrição para a imagem fornecida.")
        print("Isso pode ocorrer por problemas de memória ou uma falha silenciosa no modelo.")

    # ================== #
    #    FIM DA MELHORIA   #
    # ================== #
    print("="*50)
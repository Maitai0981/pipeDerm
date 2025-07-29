# pipeDerm

**pipeDerm** é um sistema de análise de lesões cutâneas assistida por Inteligência Artificial. A aplicação utiliza uma API robusta para processar imagens de lesões de pele, fornecendo uma descrição clínica, classificação de possíveis condições e um laudo preliminar estruturado.

## Visão Geral

O projeto consiste em uma API construída com Flask que orquestra um pipeline de múltiplos modelos de IA para realizar uma análise dermatológica completa a partir de uma única imagem. O sistema foi projetado para ser uma ferramenta de auxílio diagnóstico, otimizada para performance e com uma arquitetura modular.

## Recursos

-   **Pipeline Multi-Modelo:** Integra modelos de IA especializados para classificação, descrição visual e geração de laudos.
-   **API RESTful:** Oferece endpoints para predição, consulta de status do sistema e acesso aos resultados.
-   **Otimização de Hardware:** Detecta e utiliza GPU (CUDA) para aceleração, com fallback para CPU. Aplica quantização de modelos (`bitsandbytes`) para reduzir o consumo de memória VRAM.
-   **Processamento Assíncrono:** Utiliza `ThreadPoolExecutor` para executar tarefas de análise em paralelo, melhorando o tempo de resposta.
-   **Interface de Teste:** Inclui uma página web simples para upload de imagens e visualização dos resultados em tempo real.
-   **Limpeza Automática:** Um agendador (`apscheduler`) realiza a limpeza periódica de arquivos de resultados antigos para gerenciar o espaço de armazenamento.

## Como Funciona

O fluxo de análise da aplicação segue os seguintes passos:

1.  **Upload da Imagem:** O usuário envia uma imagem de uma lesão cutânea através da interface web ou diretamente para o endpoint `/api/predict`.
2.  **Pré-processamento:** A imagem passa por um tratamento de otimização de contraste (CLAHE) para realçar características importantes.
3.  **Análise Paralela:**
    * **Classificação:** Um modelo de classificação de imagens (`NeuronZero/SkinCancerClassifier`) analisa a imagem e retorna as hipóteses diagnósticas mais prováveis com seus respectivos scores de confiança.
    * **Descrição Visual:** Um modelo de visão-linguagem (`Salesforce/blip2-opt-2.7b`) gera uma descrição textual detalhada da morfologia, cor, bordas e superfície da lesão.
4.  **Geração do Laudo:** Os resultados da classificação e a descrição visual são enviados para um Modelo de Linguagem Grande (LLM) via Ollama, que gera um laudo médico preliminar coeso e estruturado.
5.  **Retorno do Resultado:** A API retorna um JSON completo contendo o diagnóstico principal, diagnósticos alternativos, a descrição da lesão e o laudo gerado.

## Modelos Utilizados

-   **Descrição Visual:** `Salesforce/blip2-opt-2.7b`
-   **Classificação de Lesões:** `NeuronZero/SkinCancerClassifier`
-   **Geração de Laudos (LLM):** `llama3:8b` (ou outro modelo configurado via Ollama)

## Instalação e Configuração

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/maitai0981/pipederm.git](https://github.com/maitai0981/pipederm.git)
    cd pipederm/source
    ```

2.  **Crie um ambiente virtual e instale as dependências:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Instale e execute o Ollama:**
    É necessário ter o [Ollama](https://ollama.com/) instalado e em execução para a geração dos laudos. Após instalar, puxe o modelo LLM:
    ```bash
    ollama pull llama3:8b
    ```

## Como Executar

Com o ambiente configurado e o Ollama em execução, inicie a aplicação Flask a partir do diretório `source`:

```bash
python run.py --host 0.0.0.0 --port 5000

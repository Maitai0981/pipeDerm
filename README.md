# PipeDerm - Sistema de Análise Dermatológica com IA

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1.1-green.svg)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Descrição

O **PipeDerm** é um sistema de análise dermatológica que combina modelos de Inteligência Artificial para classificar lesões cutâneas e gerar laudos médicos preliminares. O sistema utiliza:

- **Classificação de Imagens**: Modelo especializado em dermatologia para identificar tipos de lesões
- **Descrição Visual**: Geração automática de descrições detalhadas das lesões
- **Laudos Médicos**: Geração de relatórios médicos preliminares
- **API REST**: Interface completa para integração com outros sistemas
- **Documentação Swagger**: Documentação automática da API
- **Monitoramento**: Sistema de métricas e health checks

## 🚀 Funcionalidades

### ✨ Principais Recursos
- **Análise de Imagens**: Upload e processamento de imagens dermatológicas
- **Classificação Automática**: Identificação de 9 tipos diferentes de lesões cutâneas
- **Descrição Visual**: Geração automática de descrições detalhadas
- **Laudos Médicos**: Criação de relatórios médicos preliminares
- **Interface Web**: Interface amigável para upload e visualização de resultados
- **API REST**: Endpoints para integração com outros sistemas
- **Documentação Swagger**: Interface interativa para testar a API
- **Métricas em Tempo Real**: Monitoramento de performance e uso
- **Validação Avançada**: Verificação rigorosa de arquivos de entrada
- **Rate Limiting**: Proteção contra abuso da API

### 🎯 Tipos de Lesões Suportadas
- **AK**: Ceratose Actínica
- **BCC**: Carcinoma Basocelular
- **BKL**: Ceratose Benigna
- **DF**: Dermatofibroma
- **MEL**: Melanoma
- **NV**: Nevo Melanocítico
- **SCC**: Carcinoma Espinocelular
- **VASC**: Lesão Vascular
- **SEB**: Queratose Seborreica

## 🛠️ Tecnologias Utilizadas

### Backend
- **Flask**: Framework web para API
- **PyTorch**: Framework de deep learning
- **Transformers**: Biblioteca para modelos de IA
- **Pillow**: Processamento de imagens
- **OpenCV**: Processamento avançado de imagens
- **Gunicorn**: Servidor WSGI para produção

### IA e Modelos
- **Skin Cancer Classifier**: Modelo especializado em dermatologia
- **LLaVA**: Modelo multimodal para descrição de imagens
- **Llama 3**: Modelo de linguagem para geração de laudos

### Infraestrutura
- **Ollama**: Servidor local para modelos de linguagem
- **CUDA**: Aceleração GPU (opcional)
- **APScheduler**: Agendamento de tarefas
- **Flasgger**: Documentação automática da API

## 📦 Instalação

### Pré-requisitos

1. **Python 3.8+**
2. **Git**
3. **Ollama** (para modelos de linguagem)

Para mais detalhes, consulte: [WSL_SETUP.md](WSL_SETUP.md)

### 1. Clone o Repositório
```bash
git clone https://github.com/seu-usuario/pipederm.git
cd pipederm
```

### 2. Instalação do Ollama

#### Windows
```bash
# Baixe e instale do site oficial
# https://ollama.ai/download
```

#### Linux/macOS/WSL
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 3. Configuração Manual (Alternativa)

#### Criar Ambiente Virtual
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

#### Instalar Dependências
```bash
pip install -r requirements.txt
```

#### Baixar Modelos Ollama
```bash
# Iniciar servidor Ollama
ollama serve

# Em outro terminal, baixar os modelos necessários
ollama pull llama3:8b
ollama pull llava
```


### 📋 **Passo a Passo Detalhado**

#### 1. Pré-requisitos
```bash
# Verificar Python
python --version  # Deve ser 3.8+

# Verificar pip
pip --version

# Instalar dependências
pip install -r requirements.txt
```

#### 2. Configurar Ollama
```bash
# Iniciar Ollama
ollama serve

# Em outro terminal, baixar modelos
ollama pull llama3:8b
ollama pull llava
```

#### 3. Executar Aplicação
```bash
# Desenvolvimento
python run.py
```

### 🌐 **Acessos Disponíveis**

| Funcionalidade | URL | Descrição |
|----------------|-----|-----------|
| **Interface Web** | `http://localhost:5000` | Upload de imagens e visualização |
| **Documentação API** | `http://localhost:5000/apidocs/` | Swagger UI interativo |
| **Health Check** | `http://localhost:5000/api/health` | Status do sistema |
| **Métricas** | `http://localhost:5000/api/metrics` | Performance e uso |
| **Status Sistema** | `http://localhost:5000/api/system` | GPU, modelos, Ollama |

### 🐛 **Troubleshooting Rápido**

#### Erro: "ModuleNotFoundError"
```bash
# Reinstalar dependências
pip install -r requirements.txt
```

#### Erro: "Ollama não encontrado"
```bash
# Verificar se está rodando
curl http://localhost:11434/api/tags

# Se não estiver, inicie:
ollama serve
```


#### 🔧 **Problema: LLaVA retorna descrições insatisfatórias**

Se o LLaVA está retornando descrições como "Lesão cutânea com dimensões 600x450 pixels, apresentando coloração complexa...", execute:

```bash
# 1. Verificar modelos disponíveis
ollama list

# 2. Reinstalar LLaVA se necessário
ollama rm llava
ollama pull llava
```

**Soluções específicas:**

1. **Verificar versão do LLaVA**:
   ```bash
   ollama show llava
   ```

2. **Usar versão específica**:
   ```bash
   ollama pull llava:latest
   ```

3. **Verificar recursos do sistema**:
   ```bash
   # Memória disponível
   free -h
   
   # GPU (se disponível)
   nvidia-smi
   ```

4. **Ajustar configurações**:
   ```bash
   # Editar config.py para aumentar timeout
   OLLAMA_CONFIG = {
       "base_url": "http://localhost:11434",
       "timeout": 1000,  # Aumentar para 1000s
       "max_retries": 5   # Aumentar tentativas
   }
   ```

5. **Testar com imagem simples**:
   ```bash
   # Criar imagem de teste
   python -c "
   from PIL import Image, ImageDraw
   img = Image.new('RGB', (100, 100), 'red')
   draw = ImageDraw.Draw(img)
   draw.ellipse([20, 20, 80, 80], fill='brown')
   img.save('test.jpg')
   "
   
   # Testar LLaVA
   ollama run llava "Descreva esta imagem" test.jpg
   ```

### 📊 **Monitoramento**

#### Verificar Status
```bash
# Health check
curl http://localhost:5000/api/health

# Métricas
curl http://localhost:5000/api/metrics

# Status do sistema
curl http://localhost:5000/api/system
```

## 📖 Uso

### Interface Web
1. Acesse `http://localhost:5000`
2. Faça upload de uma imagem dermatológica
3. Aguarde o processamento
4. Visualize os resultados

### Documentação da API
1. Acesse `http://localhost:5000/apidocs/`
2. Explore os endpoints disponíveis
3. Teste as funcionalidades diretamente na interface

### API REST

#### Endpoint de Análise
```bash
curl -X POST http://localhost:5000/api/predict \
  -F "image=@sua_imagem.jpg"
```

#### Resposta da API
```json
{
  "diagnostico_principal": "Nevo Melanocítico (Confiança: 99.6%)",
  "diagnosticos_alternativos": [
    {
      "nome": "Melanoma",
      "confianca": "0.4%"
    }
  ],
  "descricao_lesao": "Lesão pigmentada com bordas regulares...",
  "laudo_completo": "**Laudo Dermatológico Preliminar**\n\n**Descrição Clínica:**...",
  "tempo_processamento": "2.34 segundos",
  "request_id": "a1b2c3d4"
}
```

#### Status do Sistema
```bash
curl http://localhost:5000/api/system
```

#### Métricas
```bash
curl http://localhost:5000/api/metrics
```

## ⚙️ Configuração

### Arquivo de Configuração (`config.py`)

```python
# Configurações de Modelos
MODEL_CONFIG = {
    "description_model": "llava:7b", 
    "skin_classifier": "NeuronZero/SkinCancerClassifier",
    "llm": "llama3.1:8b" 
}
# Configurações Ollama
OLLAMA_CONFIG = {
    "base_url": "http://localhost:11434",
    "timeout": 1000,
    "max_retries": 3
}
```

### Variáveis de Ambiente
```bash
# Configurações opcionais
export FLASK_ENV=production
export CUDA_VISIBLE_DEVICES=0  # Para GPU específica
export SECRET_KEY=your-secret-key
```

## 🧪 Testes

### Executar Testes Abrangentes
```bash
python test_comprehensive.py
```

### Testes Manuais
```bash
# Testar conectividade Ollama
curl http://localhost:11434/api/tags

# Testar API
curl http://localhost:5000/api/health
curl http://localhost:5000/api/metrics
```

## 🔧 Troubleshooting

### Problemas Comuns

#### 1. Ollama não está rodando
```bash
# Verificar se o serviço está ativo
ollama serve

# Verificar modelos disponíveis
ollama list
```

#### 2. Erro de GPU
```bash
# Forçar uso de CPU
export CUDA_VISIBLE_DEVICES=""

# Ou modificar config.py
device = torch.device("cpu")
```

#### 3. Erro de Memória
```bash
# Limpar cache CUDA
python -c "import torch; torch.cuda.empty_cache()"

# Reduzir batch size em config.py
```

#### 4. Dependências não encontradas
```bash
# Reinstalar dependências
pip install --upgrade -r requirements.txt

# Verificar versão Python
python --version
```

#### 5. Erro de Swagger
```bash
# Verificar se flasgger está instalado
pip install flasgger

# Verificar logs
tail -f logs/gunicorn_error.log
```

## 📊 Monitoramento

### Logs
Os logs são salvos automaticamente com informações sobre:
- Carregamento de modelos
- Processamento de imagens
- Erros e exceções
- Performance

### Métricas Disponíveis
- Tempo de processamento
- Uso de GPU/CPU
- Taxa de sucesso
- Erros por endpoint
- Número total de requisições

### Endpoints de Monitoramento
- `/api/health` - Health check
- `/api/system` - Status do sistema
- `/api/metrics` - Métricas de performance


### Padrões de Código
- Seguir PEP 8
- Adicionar docstrings
- Incluir testes
- Atualizar documentação

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🙏 Agradecimentos

- **Hugging Face** pelos modelos de IA
- **Ollama** pelo servidor de modelos locais
- **Comunidade Open Source** pelas bibliotecas utilizadas

## 📈 Roadmap

### Versão 1.1
- [x] Interface de documentação Swagger
- [x] Sistema de métricas
- [x] Validação avançada de arquivos
- [x] Scripts de deploy automatizados
- [ ] Sistema de cache Redis
- [ ] Autenticação JWT

### Versão 1.2
- [ ] Processamento em lote
- [ ] API GraphQL
- [ ] Suporte a múltiplos idiomas
- [ ] Dashboard de administração

### Versão 2.0
- [ ] Modelos customizados
- [ ] Integração com PACS
- [ ] Análise temporal
- [ ] Machine Learning contínuo

---

**⚠️ Aviso Legal**: Este sistema é destinado apenas para triagem e não substitui a avaliação médica profissional. Sempre consulte um dermatologista para diagnóstico definitivo. 

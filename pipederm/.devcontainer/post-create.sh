#!/bin/bash
set -e # Faz o script parar imediatamente se houver um erro

echo "Iniciando script de pós-criação..."

# 1. Atualiza os pacotes e instala o 'curl' com permissão de admin
sudo apt-get update
sudo apt-get install -y curl

# 2. Instala o Ollama
echo "Instalando Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

# 3. Instala as dependências do Python
echo "Instalando dependências do Python..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Baixa o modelo de linguagem em segundo plano
echo "Baixando o modelo LLM..."
ollama pull llama3:8b &

echo "Script de pós-criação concluído com sucesso."
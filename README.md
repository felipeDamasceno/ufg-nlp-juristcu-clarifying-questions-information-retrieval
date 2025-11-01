# Sistema de Busca Híbrida - JurisTCU

Este sistema implementa uma busca híbrida combinando **BM25** e **embeddings Gemini** usando LlamaIndex para o dataset jurisTCU.

## 📁 Arquivos do Sistema

- `busca_hibrida_llamaindex.py` - Módulo principal com as classes de busca
- `teste_busca_hibrida.py` - Teste com amostra de dados do jurisTCU
- `requirements.txt` - Dependências necessárias

## 🚀 Instalação

1. **Instalar dependências:**
```bash
pip install -r requirements.txt
```

2. **Configurar API Gemini:**
```bash
# Windows PowerShell
$env:GOOGLE_API_KEY='sua_chave_aqui'

# Windows CMD
set GOOGLE_API_KEY=sua_chave_aqui

# Linux/Mac
export GOOGLE_API_KEY='sua_chave_aqui'
```
## Download dos dados

Para baixar os dados do jurisTCU, execute o script `utils/download_juris_tcu.py`:
```bash
python utils/download_juris_tcu.py
```
Isso baixará os arquivos necessários para o dataset jurisTCU na pasta `dados/`.

## 🧪 Executando Testes

### Teste com amostra de dados do jurisTCU
```bash
python teste_busca_hibrida.py
```
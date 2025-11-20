# Sistema de Busca Híbrida e Perguntas Clarificadoras — JurisTCU

Sistema de recuperação e reranking para o dataset JurisTCU combinando **BM25**, **embeddings jurídicos PT-BR**, **reranking Jina** e **perguntas clarificadoras via Gemini**. Inclui dois modos de geração de perguntas: com pares de documentos e sem pares.

## Visão Geral do Fluxo
- Carrega documentos (`doc.csv`) e queries (`query.csv`).
- Gera candidatos Top‑20 por query com busca híbrida e reranking.
- Opcional: gera a intenção de busca para cada query (`query_intencao.csv`).
- Executa pipeline de chat + perguntas clarificadoras em dois modos:
  - `pares`: pergunta clarificadora baseada em diferenças entre dois documentos similares.
  - `sem_pares`: perguntas diretamente a partir da pergunta original (sem documentos).
- Aplica novo reranking usando a conversa acumulada e salva CSVs e métricas.

## Estrutura Principal
- `src/buscador_hibrido.py`: busca híbrida (BM25 + embeddings LlamaIndex + reranker Jina).
- `src/candidatos.py` e `src/run_candidatos.py`: geração de candidatos Top‑20.
- `src/clarifying_questions.py`: geração de perguntas clarificadoras (`pares` e `sem_pares`).
- `src/resposta_clarificadora.py`: resposta automática para cada pergunta clarificadora, usando a intenção da query.
- `src/run_chat_rerank_candidatos.py`: pipeline de chat + perguntas + rerank; salva resultados e métricas por modo.
- `src/run_metricas_candidatos.py`: métricas para o CSV de candidatos base.
- `src/gerar_intencoes_dataset.py`: gera `dados/query_intencao.csv` com coluna `INTENCAO`.
- `src/utils/*`: utilitários de dados, métricas, preprocessamento e Gemini.

## Instalação
1. Instalar dependências
```bash
pip install -r requirements.txt
```

2. Configurar `.env` (recomendado)
```bash
cp .env.example .env
# Edite .env com sua chave: GOOGLE_API_KEY="sua_chave"
```
Ou definir a variável diretamente no ambiente:
```bash
# Windows PowerShell
$env:GOOGLE_API_KEY='sua_chave_aqui'
# Windows CMD
set GOOGLE_API_KEY=sua_chave_aqui
# Linux/Mac
export GOOGLE_API_KEY='sua_chave_aqui'
```

## Dados
- Download automático do dataset (doc.csv, query.csv, qrel.csv):
```bash
python utils/download_jurisTCU.py
```
- Arquivos esperados: `dados/juris_tcu/doc.csv`, `dados/juris_tcu/query.csv`, `dados/juris_tcu/qrel.csv`.

## Geração de Candidatos (Top‑20)
```bash
python -m src.run_candidatos
```
- Saída: `dados/candidatos_top20_full.csv`

## Intenção de Busca (opcional)
Gera `query_intencao.csv` com a coluna `INTENCAO` para ser usada no pipeline de chat.
```bash
python -m src.gerar_intencoes_dataset
```
- Saída: `dados/query_intencao.csv`

## Chat + Perguntas Clarificadoras + Rerank
- Modo com pares (usa diferenças entre documentos similares):
```bash
python -m src.run_chat_rerank_candidatos --modo pares --n 3
```
- Modo sem pares (gera perguntas apenas da pergunta original):
```bash
python -m src.run_chat_rerank_candidatos --modo sem_pares --n 3
```
- `--n` controla quantas queries serão processadas (use `--n 0` para todas).
- Saídas por modo:
  - `pares`: `dados/candidatos_chat_top20.csv` e `dados/metricas_candidatos_chat_top10.csv`
  - `sem_pares`: `dados/candidatos_chat_nodocs_top20.csv` e `dados/metricas_candidatos_chat_nodocs_top10.csv`

## Métricas do CSV de Candidatos
```bash
python -m src.run_metricas_candidatos
```
- Saída: `dados/metricas_candidatos_top10.csv`

## Testes úteis
```bash
# Busca híbrida + rerank sample
python -m tests.teste_busca_hibrida_rerank_sample

# Perguntas clarificadoras (com pares)
python -m tests.teste_perguntas_clarificadoras_fake

# Perguntas clarificadoras (sem pares)
python -m tests.teste_perguntas_sem_pares_fake

# Fluxo real com intenção clarificadora
python -m tests.teste_fluxo_real_intencao_clarificadora
```

## Modelos e Notas
- Embeddings: `stjiris/bert-large-portuguese-cased-legal-mlm-sts-v1.0` (PT‑BR jurídico).
- Reranker: `jinaai/jina-reranker-v2-base-multilingual` (CPU/GPU automático).
- Gemini: configurar `GOOGLE_API_KEY`; opcional `GEMINI_MODEL_NAME` (`.env.example`).

## Pastas de Saída
- `dados/` contém todos os CSVs gerados: candidatos, candidatos_chat (por modo) e métricas correspondentes.
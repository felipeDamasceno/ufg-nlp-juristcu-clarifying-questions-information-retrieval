"""
Gera a intenção de busca para TODAS as queries do dataset jurisTCU,
usando o ENUNCIADO completo dos documentos com SCORE 3 (qrel.csv),
removendo HTML via utilitário de preprocessamento, e salva um novo CSV
em `outputs/query_intencao.csv` com a nova coluna `INTENCAO`.

Execução:
    python src/gerar_intencoes_dataset.py

Requisitos:
    - `.env` com `GOOGLE_API_KEY` configurado
    - Dependências em `requirements.txt` instaladas
"""

import os
import sys
from typing import List

import pandas as pd
from dotenv import load_dotenv

# Carregar variáveis de ambiente sempre no topo (fora de funções)
load_dotenv()

# Adicionar raiz do projeto ao path para imports relativos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, "dados", "juris_tcu")
QUERY_CSV = os.path.join(DATA_DIR, "query.csv")
QREL_CSV = os.path.join(DATA_DIR, "qrel.csv")
DOC_CSV = os.path.join(DATA_DIR, "doc.csv")

OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
OUTPUT_CSV = os.path.join(OUTPUTS_DIR, "query_intencao.csv")

import time
from src.intencao_busca import gerar_intencao_busca
from src.utils.dados import load_queries_df, load_qrels_df, load_docs_enunciado_map_clean


def gerar_para_todas_as_queries() -> None:
    if not (os.path.exists(QUERY_CSV) and os.path.exists(QREL_CSV) and os.path.exists(DOC_CSV)):
        print("❌ Arquivos necessários não encontrados em dados/juris_tcu")
        print(f"   Esperados: {QUERY_CSV}, {QREL_CSV}, {DOC_CSV}")
        return

    queries_df = load_queries_df(QUERY_CSV)
    # Suporte a retomada: se já existir saída com INTENCAO preenchida, reaproveitar
    if os.path.exists(OUTPUT_CSV):
        try:
            existing_out = pd.read_csv(OUTPUT_CSV)
            if "INTENCAO" in existing_out.columns:
                queries_df = queries_df.merge(existing_out[["ID", "INTENCAO"]], on="ID", how="left")
        except Exception:
            pass
    qrels_df = load_qrels_df(QREL_CSV)
    docs_map = load_docs_enunciado_map_clean(DOC_CSV)

    # Sempre usar TODOS os documentos com SCORE 3, ordenados por RANK
    score3_df = qrels_df[qrels_df["SCORE"] == 3].sort_values("RANK")

    intents: List[str] = []
    print(f"--- Gerando intenções para {len(queries_df)} queries ---")
    for _, qrow in queries_df.iterrows():
        qid = int(qrow["ID"])
        qtext = str(qrow.get("TEXT", ""))

        # Documentos ideais para essa query (todos SCORE 3)
        docs_rows = score3_df[score3_df["QUERY_ID"] == qid]
        docs_ideais: List[str] = []
        for _, drow in docs_rows.iterrows():
            doc_id = int(drow["DOC_ID"])
            enun = docs_map.get(doc_id)
            if enun:
                docs_ideais.append(enun)

        # Pular se já há INTENCAO previamente salva (retomada) — apenas se não for vazio/NaN
        if "INTENCAO" in queries_df.columns:
            val = qrow.get("INTENCAO", None)
            if val is not None and not pd.isna(val):
                sval = str(val).strip()
                if sval and sval.lower() != "nan":
                    intents.append(sval)
                    continue

        # Tentar com backoff exponencial simples em caso de 429 (quota)
        max_retries = 5
        base_delay = 5.0  # segundos
        for attempt in range(1, max_retries + 1):
            try:
                resultado = gerar_intencao_busca(qtext, docs_ideais)
                intents.append(str(resultado.get("intent", "")))
                break
            except RuntimeError as e:
                msg = str(e)
                if "429" in msg:
                    # Se for limite diário, não adianta continuar: salvar parcial e sair
                    if "PerDay" in msg or "per day" in msg.lower():
                        print("  ⚠ Limite diário atingido. Salvando progresso e encerrando.")
                        intents.append("")
                        queries_df["INTENCAO"] = intents
                        os.makedirs(OUTPUTS_DIR, exist_ok=True)
                        queries_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
                        return
                    if attempt < max_retries:
                        delay = base_delay * (2 ** (attempt - 1))
                        print(f"  ↻ Query ID {qid}: 429 (quota). Aguardando {delay:.0f}s e tentando novamente...")
                        time.sleep(delay)
                        continue
                print(f"  ✗ Query ID {qid}: falha ao gerar intenção: {e}")
                intents.append("")
                break

        # Ritmo constante para respeitar limite por minuto
        time.sleep(5)

    # Salvar CSV com coluna INTENCAO
    queries_df["INTENCAO"] = intents
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    queries_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"✓ CSV gerado em: {OUTPUT_CSV}")


if __name__ == "__main__":
    gerar_para_todas_as_queries()
import sys
import os
import random
import re
import pandas as pd

# Adicionar o diretório raiz do projeto ao sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from src.intencao_busca import gerar_intencao_busca


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dados", "juris_tcu")

QUERY_CSV = os.path.join(DATA_DIR, "query.csv")
QREL_CSV = os.path.join(DATA_DIR, "qrel.csv")
DOC_CSV = os.path.join(DATA_DIR, "doc.csv")


def _load_queries(path: str):
    df = pd.read_csv(path, dtype=str, encoding="utf-8").fillna("")
    return df[["ID", "TEXT", "SOURCE"]].to_dict(orient="records")


def _load_qrels(path: str):
    df = pd.read_csv(path, encoding="utf-8")
    df["QUERY_ID"] = pd.to_numeric(df["QUERY_ID"], errors="coerce")
    df["DOC_ID"] = pd.to_numeric(df["DOC_ID"], errors="coerce")
    df["SCORE"] = pd.to_numeric(df["SCORE"], errors="coerce").fillna(0)
    df["RANK"] = pd.to_numeric(df["RANK"], errors="coerce").fillna(999999)
    df = df.dropna(subset=["QUERY_ID", "DOC_ID"]).astype({"QUERY_ID": int, "DOC_ID": int, "RANK": int})
    df = df.sort_values("RANK")
    by_query = {}
    for qid, group in df.groupby("QUERY_ID"):
        by_query[int(qid)] = group[["DOC_ID", "SCORE", "RANK"]].to_dict(orient="records")
    return by_query


def _load_docs_enunciado_by_numeric_key(path: str):
    df = pd.read_csv(path, dtype=str, encoding="utf-8").fillna("")
    df["NUM"] = df["KEY"].astype(str).str.extract(r"(\d+)$")
    df["NUM"] = pd.to_numeric(df["NUM"], errors="coerce")
    df = df.dropna(subset=["NUM"]).astype({"NUM": int})
    return df.set_index("NUM")["ENUNCIADO"].to_dict()


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]*>", "", text or "").strip()


def teste_intencao_busca():
    """
    Teste que:
    - Seleciona 5 queries aleatórias com resultados
    - Lista documentos ideais (score 3) por query
    - Gera a intenção de busca via Gemini e imprime saída JSON e campo 'intent'
    """
    load_dotenv()

    print("--- Iniciando Teste de Intenção de Busca ---")

    if not (os.path.exists(QUERY_CSV) and os.path.exists(QREL_CSV) and os.path.exists(DOC_CSV)):
        print("✗ Teste abortado: arquivos necessários não encontrados em dados/juris_tcu")
        print(f"  Esperados: {QUERY_CSV}, {QREL_CSV}, {DOC_CSV}")
        return

    queries = _load_queries(QUERY_CSV)
    qrels_by_query = _load_qrels(QREL_CSV)
    docs_map = _load_docs_enunciado_by_numeric_key(DOC_CSV)

    query_ids_with_results = [int(q["ID"]) for q in queries if q.get("ID") and int(q["ID"]) in qrels_by_query]
    if not query_ids_with_results:
        print("✗ Nenhuma query com resultados encontrada no qrels.")
        return

    sample_ids = random.sample(query_ids_with_results, min(5, len(query_ids_with_results)))
    top_n = int(os.getenv("PREVIEW_TOP", "10"))

    query_by_id = {int(q["ID"]): q for q in queries if q.get("ID")}

    for idx, qid in enumerate(sample_ids, 1):
        q = query_by_id.get(qid, {"TEXT": "", "SOURCE": ""})
        print(f"\n--- Query #{idx} (ID: {qid}) ---")
        print(f"  Texto: {q.get('TEXT', '')}")
        print(f"  Source: {q.get('SOURCE', '')}")

        resultados = [r for r in qrels_by_query.get(qid, []) if int(r.get("SCORE", 0)) == 3]
        if not resultados:
            print("  (Sem documentos de score 3)")
            continue

        docs_enunciados = []
        print("  Documentos ideais (score 3):")
        for item in resultados[:top_n]:
            doc_id = item["DOC_ID"]
            enunciado = docs_map.get(doc_id)
            enun_clean = _strip_html(enunciado) if enunciado else "(enunciado não encontrado)"
            docs_enunciados.append(enun_clean)
            print(f"    - Doc {doc_id} | Rank {item['RANK']}: {enun_clean}")

        try:
            resultado = gerar_intencao_busca(q.get("TEXT", ""), docs_enunciados)
            print(f"  Resposta completa (Gemini): {resultado.get('full_text')}")
            print(f"  Intenção (extraída): {resultado.get('intent')}\n")
        except RuntimeError as e:
            print("  ✗ Erro ao gerar intenção via Gemini:")
            print(f"    Detalhes: {e}")
            print("    Dicas: verifique se 'google-generativeai' está instalado e se 'GOOGLE_API_KEY' está configurada.")

    print("\n--- Teste de Intenção de Busca Concluído ---")


if __name__ == "__main__":
    teste_intencao_busca()
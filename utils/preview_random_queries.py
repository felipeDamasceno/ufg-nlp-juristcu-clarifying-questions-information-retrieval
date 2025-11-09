"""
Utilitário: imprime 5 queries aleatórias do dataset jurisTCU
e lista os documentos retornados com seus scores, juntando:
- query.csv (ID, TEXT, SOURCE)
- qrel.csv (QUERY_ID, DOC_ID, SCORE, ENGINE, RANK)
- doc.csv (usa o ENUNCIADO do documento via sufixo numérico da coluna KEY)

Execução:
    python utils/preview_random_queries.py

Opcional:
    Defina a variável de ambiente PREVIEW_TOP para ajustar quantos docs por query mostrar (padrão 10).
"""

import os
import csv
import random
import re
from typing import Dict, List, Tuple


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dados", "juris_tcu")

QUERY_CSV = os.path.join(DATA_DIR, "query.csv")
QREL_CSV = os.path.join(DATA_DIR, "qrel.csv")
DOC_CSV = os.path.join(DATA_DIR, "doc.csv")


def _load_queries(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [
            {"ID": row.get("ID", ""), "TEXT": row.get("TEXT", ""), "SOURCE": row.get("SOURCE", "")}
            for row in reader
        ]
    return rows


def _load_qrels(path: str) -> Dict[int, List[Dict[str, str]]]:
    """Carrega qrels e agrupa por QUERY_ID, ordenando por RANK."""
    by_query: Dict[int, List[Dict[str, str]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                qid = int(row.get("QUERY_ID", ""))
                doc_id = int(row.get("DOC_ID", ""))
            except ValueError:
                continue
            score_str = row.get("SCORE", "0")
            rank_str = row.get("RANK", "999999")
            try:
                score = int(score_str)
            except ValueError:
                try:
                    score = float(score_str)
                except ValueError:
                    score = 0
            try:
                rank = int(rank_str)
            except ValueError:
                rank = 999999
            by_query.setdefault(qid, []).append({
                "DOC_ID": doc_id,
                "SCORE": score,
                "RANK": rank,
            })

    # Ordena cada lista por rank crescente
    for qid, items in by_query.items():
        items.sort(key=lambda x: x["RANK"])
    return by_query


def _load_docs_enunciado_by_numeric_key(path: str) -> Dict[int, str]:
    """Cria um mapa DOC_ID (número) -> ENUNCIADO, extraindo o sufixo numérico da coluna KEY."""
    mapping: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row.get("KEY", "")
            enunciado = row.get("ENUNCIADO", "") or ""
            m = re.search(r"(\d+)$", str(key))
            if not m:
                continue
            try:
                numeric_id = int(m.group(1))
            except ValueError:
                continue
            mapping[numeric_id] = enunciado
    return mapping


def _strip_html(text: str) -> str:
    # Remoção simples de tags HTML
    return re.sub(r"<[^>]*>", "", text or "").strip()


def main():
    if not (os.path.exists(QUERY_CSV) and os.path.exists(QREL_CSV) and os.path.exists(DOC_CSV)):
        print("❌ Arquivos necessários não encontrados em dados/juris_tcu")
        print(f"   Esperados: {QUERY_CSV}, {QREL_CSV}, {DOC_CSV}")
        return

    queries = _load_queries(QUERY_CSV)
    qrels_by_query = _load_qrels(QREL_CSV)
    docs_map = _load_docs_enunciado_by_numeric_key(DOC_CSV)

    # Seleciona IDs de queries que têm resultados no qrels
    query_ids_with_results = [int(q["ID"]) for q in queries if q.get("ID") and int(q["ID"]) in qrels_by_query]
    if not query_ids_with_results:
        print("❌ Nenhuma query com resultados encontrada no qrels.")
        return

    k = 5
    sample_ids = random.sample(query_ids_with_results, min(k, len(query_ids_with_results)))
    top_n = int(os.getenv("PREVIEW_TOP", "10"))

    # Índice de queries por ID para acesso rápido
    query_by_id: Dict[int, Dict[str, str]] = {int(q["ID"]): q for q in queries if q.get("ID")}

    print("\n=== Pré-visualização de 5 queries aleatórias (jurisTCU) ===\n")
    for idx, qid in enumerate(sample_ids, 1):
        q = query_by_id.get(qid, {"TEXT": "", "SOURCE": ""})
        print(f"--- Query #{idx} (ID: {qid}) ---")
        print(f"  Texto: {q.get('TEXT', '')}")
        print(f"  Source: {q.get('SOURCE', '')}")

        resultados = qrels_by_query.get(qid, [])
        if not resultados:
            print("  (Sem resultados no qrels)\n")
            continue

        print("  Documentos retornados:")
        for item in resultados[:top_n]:
            doc_id = item["DOC_ID"]
            score = item["SCORE"]
            rank = item["RANK"]
            enunciado = docs_map.get(doc_id)
            enun_clean = _strip_html(enunciado) if enunciado else "(enunciado não encontrado)"
            # Pequena truncagem para caber no terminal
            if len(enun_clean) > 300:
                enun_clean = enun_clean[:300].rstrip() + "..."
            print(f"    - Rank {rank} | Doc {doc_id} | Score {score}: {enun_clean}")
        print("")

    print("--- Fim da pré-visualização ---")


if __name__ == "__main__":
    main()
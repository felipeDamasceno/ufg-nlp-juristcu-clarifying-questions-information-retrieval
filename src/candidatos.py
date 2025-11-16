import os
import csv
from typing import List, Dict
from src.buscador_hibrido import BuscadorHibridoLlamaIndex

def executar_busca_candidatos(
    queries: List[Dict],
    documentos,
    output_csv_path: str,
    persist_dir: str,
    bm25_top_k: int = 50,
    embeddings_top_k: int = 50,
    hybrid_top_k: int = 50,
    rerank_top_n: int = 20,
):
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    os.makedirs(persist_dir, exist_ok=True)

    buscador = BuscadorHibridoLlamaIndex()
    buscador.carregar_documentos(documentos)
    buscador.set_bm25_top_k(bm25_top_k)
    buscador.set_embeddings_top_k(embeddings_top_k)
    buscador.set_hibrido_top_k(hybrid_top_k)

    try:
        buscador.vector_index.storage_context.persist(persist_dir=persist_dir)
    except Exception:
        pass

    rows = []
    for q in queries:
        qid = int(q.get("ID")) if q.get("ID") is not None else None
        text = str(q.get("TEXT", ""))
        resultados = buscador.buscar_hibrido(text, top_k=rerank_top_n, use_reranker=True) or []
        for rank, item in enumerate(resultados, start=1):
            rows.append({
                "QUERY_ID": qid,
                "DOC_ID": item.get("id"),
                "RERANK_SCORE": item.get("score"),
                "RANK": rank,
            })

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["QUERY_ID", "DOC_ID", "RERANK_SCORE", "RANK"])
        writer.writeheader()
        writer.writerows(rows)

    return rows
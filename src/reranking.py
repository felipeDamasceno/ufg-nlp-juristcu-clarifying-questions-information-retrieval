from typing import List, Any
import torch


def rerank_nodes(reranker_model, query: str, nodes: List[Any], top_n: int = 5) -> List[Any]:
    """
    Aplica o reranking nos nós usando o modelo Jina Reranker (ou compatível).

    Mantém o padrão de logs para não quebrar a expectativa dos testes existentes.
    """
    if not reranker_model or not nodes:
        return nodes

    print(f"--- Aplicando Reranking em {len(nodes)} nós ---")

    # Criar pares de [query, texto_do_nó]
    pairs = [[query, node.get_content()] for node in nodes]

    with torch.no_grad():
        scores = reranker_model.compute_score(pairs, batch_size=4)

    # Atribuir novos scores
    for node, score in zip(nodes, scores):
        node.score = float(score)

    # Ordenar por score desc
    reranked_nodes = sorted(nodes, key=lambda x: x.score, reverse=True)

    print(f"✓ Reranking concluído. Retornando os {top_n} melhores resultados.")
    return reranked_nodes[:top_n]
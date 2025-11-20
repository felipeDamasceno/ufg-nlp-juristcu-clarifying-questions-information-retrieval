from typing import List, Any
import torch
from llama_index.core.schema import NodeWithScore


def rerank_nodes(reranker_model, query: str, nodes: List[Any], top_n: int = 5) -> List[Any]:
    """
    Aplica o reranking nos nós usando o modelo Jina Reranker (ou compatível).

    Mantém o padrão de logs para não quebrar a expectativa dos testes existentes.
    """
    if not reranker_model or not nodes:
        return nodes

    print(f"--- Aplicando Reranking em {len(nodes)} nós ---")

    # Criar pares de [query, texto_do_nó] suportando NodeWithScore de entrada
    pairs = []
    base_nodes = []
    for node in nodes:
        base = getattr(node, 'node', node)
        try:
            content = base.get_content()
        except Exception:
            content = getattr(base, 'text', '')
        pairs.append([query, content])
        base_nodes.append(base)

    with torch.no_grad():
        scores = reranker_model.compute_score(pairs, batch_size=4)

    # Atribuir novos scores ao nó base (não embrulhar NodeWithScore dentro de outro)
    scored = [NodeWithScore(node=base, score=float(score)) for base, score in zip(base_nodes, scores)]

    # Ordenar por score desc
    reranked_nodes = sorted(scored, key=lambda x: x.score, reverse=True)

    print(f"✓ Reranking concluído. Retornando os {top_n} melhores resultados.")
    return reranked_nodes[:top_n]
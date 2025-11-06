from typing import List, Dict, Optional
import itertools
import numpy as np


def _texto_do_resultado(item: Dict) -> str:
    """
    Retorna o texto do item de resultado de busca, com fallback para chaves comuns.
    """
    if 'conteudo' in item and item['conteudo']:
        return item['conteudo']
    if 'texto_completo' in item and item['texto_completo']:
        return item['texto_completo']
    # Fallback mínimo: tenta enunciado
    return item.get('enunciado', '')


def calcular_similaridade_entre_pares(
    resultados_busca: List[Dict],
    embeddings_model,
    limite_similaridade: float = 0.8,
    top_k: int = 3
) -> Optional[List[Dict]]:
    """
    Calcula a similaridade de cosseno entre todos os pares de documentos de um resultado de busca.

    Args:
        resultados_busca: Lista de dicionários, cada um representa um documento retornado.
        embeddings_model: Modelo de embeddings a ser usado (objeto com método `get_text_embedding_batch`).
        limite_similaridade: Limite mínimo de similaridade para considerar um par.
        top_k: Número máximo de pares mais similares a retornar.

    Returns:
        Uma lista com os top_k pares mais similares que atendem ao limite, ou None se nenhum par for encontrado.
    """
    if not embeddings_model or not resultados_busca or len(resultados_busca) < 2:
        return None

    # 1. Gerar embeddings para todos os documentos nos resultados
    textos = [_texto_do_resultado(resultado) for resultado in resultados_busca]
    try:
        embeddings = embeddings_model.get_text_embedding_batch(textos, show_progress=False)
    except Exception as e:
        print(f"Erro ao gerar embeddings em lote: {e}")
        return None

    # 2. Criar todos os pares possíveis de documentos
    pares_indices = list(itertools.combinations(range(len(resultados_busca)), 2))

    pares_similares = []

    # 3. Calcular a similaridade de cosseno para cada par
    for i, j in pares_indices:
        embedding_i = np.array(embeddings[i]).reshape(1, -1)
        embedding_j = np.array(embeddings[j]).reshape(1, -1)

        # Normalizar os vetores para cálculo de similaridade de cosseno
        embedding_i_norm = embedding_i / np.linalg.norm(embedding_i)
        embedding_j_norm = embedding_j / np.linalg.norm(embedding_j)

        similaridade = np.dot(embedding_i_norm, embedding_j_norm.T).item()

        if similaridade > limite_similaridade:
            par = {
                "documento_1": resultados_busca[i],
                "documento_2": resultados_busca[j],
                "similaridade": similaridade
            }
            pares_similares.append(par)

    # 4. Se nenhum par atendeu ao limite, retornar None
    if not pares_similares:
        return None

    # 5. Ordenar os pares pela similaridade em ordem decrescente
    pares_similares.sort(key=lambda x: x['similaridade'], reverse=True)

    # 6. Retornar o top_k
    return pares_similares[:top_k]
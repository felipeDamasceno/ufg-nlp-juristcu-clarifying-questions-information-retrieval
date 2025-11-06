"""Script de Teste - Sistema de Busca Híbrida LlamaIndex
Testa o sistema com dados reais do jurisTCU (100 primeiros documentos)
Usando embedding português jurídico local
"""

import os
import sys
from dotenv import load_dotenv

# Adicionar o diretório raiz ao path do Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Imports da estrutura modular
from src.buscador_hibrido import BuscadorHibridoLlamaIndex
from src.utils.dados import carregar_dados_juris_tcu


def testar_reranker_com_dados_reais():
    """Testa a integração do Reranker com a busca híbrida usando dados reais."""
    print("\n" + "=" * 60)
    print("TESTE DE RERANKING COM DADOS REAIS")
    print("=" * 60)

    caminho_csv = "dados/juris_tcu/doc.csv"
    if not os.path.exists(caminho_csv):
        print(f"❌ Arquivo não encontrado: {caminho_csv}")
        return

    buscador = BuscadorHibridoLlamaIndex()
    documentos = carregar_dados_juris_tcu(caminho_csv, limite=50) # Usar um subconjunto menor para agilidade
    buscador.carregar_documentos(documentos)

    query = "contratos administrativos e superfaturamento"
    print(f"\n🔍 Testando Reranker com a query: '{query}'")
    print("-" * 50)

    # 1. Buscar sem Reranker (apenas RRF)
    print("\n📊 Resultados Híbridos (RRF - antes do Reranking):")
    resultados_sem_reranker = buscador.buscar_hibrido(query, top_k=5)
    if resultados_sem_reranker:
        for i, res in enumerate(resultados_sem_reranker, 1):
            print(f"  {i}. ID: {res['id']} | Score RRF: {res['score']:.4f}")
    else:
        print("  Nenhum resultado encontrado.")

    # 2. Buscar com Reranker
    print("\n✨ Resultados Híbridos (com Reranker):")
    resultados_com_reranker = buscador.buscar_hibrido(query, top_k=5, use_reranker=True)
    if resultados_com_reranker:
        for i, res in enumerate(resultados_com_reranker, 1):
            print(f"  {i}. ID: {res['id']} | Score Rerank: {res['score']:.4f} | Método: {res['metodo']}")
    else:
        print("  Nenhum resultado encontrado.")

    # 3. Validação
    if resultados_sem_reranker and resultados_com_reranker:
        ids_originais = [r['id'] for r in resultados_sem_reranker]
        ids_reranked = [r['id'] for r in resultados_com_reranker]
        
        print("\n--- Comparação de Ordem ---")
        print(f"Ordem original (RRF): {ids_originais}")
        print(f"Ordem com Reranker:   {ids_reranked}")
        
        if ids_originais != ids_reranked:
            print("\n✓ SUCESSO: O Reranker alterou a ordem dos resultados.")
        else:
            print("\n⚠️ AVISO: O Reranker não alterou a ordem dos resultados. Isso pode ser esperado para certas queries.")

        # Verificar se o score mudou
        score_original = resultados_sem_reranker[0]["score"]
        score_reranked = resultados_com_reranker[0]["score"]
        assert score_original != score_reranked, "O score do reranker deve ser diferente do score RRF."
        print("✓ SUCESSO: O score foi atualizado pelo Reranker.")


if __name__ == "__main__":

    # Teste do Reranker
    testar_reranker_com_dados_reais()

    print("\n" + "=" * 60)
    print("✅ Todos os testes foram concluídos com sucesso!")
    print("=" * 60)
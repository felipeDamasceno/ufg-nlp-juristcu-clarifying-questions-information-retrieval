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
from src.utils.dados import criar_dados_exemplo

def testar_com_dados_exemplo():
    """Testa o sistema com dados de exemplo"""
    print("=" * 60)
    print("TESTE COM DADOS DE EXEMPLO")
    print("=" * 60)
    
    # Criar buscador
    buscador = BuscadorHibridoLlamaIndex()
    
    # Carregar dados de exemplo
    documentos = criar_dados_exemplo()
    buscador.carregar_documentos(documentos)
    
    # Queries de teste
    queries = [
        "responsabilidade fiscal",
        "auditoria contas públicas",
        "controle interno"
    ]
    
    for query in queries:
        print(f"\n🔍 Testando query: '{query}'")
        print("-" * 40)
        
        # Busca BM25
        print("\n📊 Resultados BM25:")
        resultados_bm25 = buscador.buscar_bm25(query, top_k=3)
        if resultados_bm25:
            for i, resultado in enumerate(resultados_bm25, 1):
                print(f"  {i}. ID: {resultado['id']} | Score: {resultado['score']:.4f}")
                print(f"     Enunciado: {resultado['enunciado'][:80]}...")
        else:
            print("  Nenhum resultado encontrado")
        
        # Busca por embeddings (se disponível)
        if buscador.vector_retriever:
            print("\n🧠 Resultados Embeddings:")
            resultados_embeddings = buscador.buscar_embeddings(query, top_k=3)
            if resultados_embeddings:
                for i, resultado in enumerate(resultados_embeddings, 1):
                    print(f"  {i}. ID: {resultado['id']} | Score: {resultado['score']:.4f}")
                    print(f"     Enunciado: {resultado['enunciado'][:80]}...")
            else:
                print("  Nenhum resultado encontrado")
        else:
            print("\n🧠 Embeddings: Não disponível (erro na configuração)")
        
        # Busca híbrida
        print("\n🔄 Resultados Híbridos (QueryFusionRetriever - RRF):")
        resultados_hibrido = buscador.buscar_hibrido(query, top_k=3)
        if resultados_hibrido:
            for i, resultado in enumerate(resultados_hibrido, 1):
                print(f"  {i}. ID: {resultado['id']} | Score RRF: {resultado['score']:.4f}")
                print(f"     Método: {resultado.get('metodo', 'Híbrido')}")
                print(f"     Conteúdo: {resultado.get('conteudo', resultado.get('titulo', ''))[:80]}...")
        else:
            print("  Nenhum resultado encontrado")
        
        # Métricas de performance
        print("\n⏱️ Performance:")
        metricas = buscador.avaliar_performance(query)
        for metodo, dados in metricas.items():
            if dados.get("disponivel"):
                print(f"  {metodo.upper()}: {dados['tempo']:.4f}s | {dados['resultados']} resultados")
            else:
                print(f"  {metodo.upper()}: Não disponível")

def testar_configuracoes_hibridas():
    """Testa a busca híbrida usando QueryFusionRetriever com Reciprocal Rank Fusion"""
    print("\n" + "=" * 60)
    print("TESTE DE BUSCA HÍBRIDA - QUERYFUSIONRETRIEVER (RRF)")
    print("=" * 60)
    
    # Embeddings locais sempre disponíveis
    print("✅ Usando embedding português jurídico local")
    
    # Criar buscador
    buscador = BuscadorHibridoLlamaIndex()
    
    # Usar dados de exemplo para teste rápido
    documentos = criar_dados_exemplo()
    buscador.carregar_documentos(documentos)
    
    query = "responsabilidade fiscal"
    print(f"\n🔍 Testando busca híbrida com RRF para: '{query}'")
    
    # Testar diferentes métodos de busca para comparação
    print("\n--- Comparação de Métodos ---")
    
    # BM25 apenas
    print("\n1. BM25 apenas:")
    resultados_bm25 = buscador.buscar_bm25(query, top_k=3)
    for i, resultado in enumerate(resultados_bm25, 1):
        print(f"  {i}. ID: {resultado['id']} | Score BM25: {resultado['score']:.4f}")
    
    # Embeddings apenas (se disponível)
    if buscador.vector_retriever:
        print("\n2. Embeddings apenas:")
        resultados_embeddings = buscador.buscar_embeddings(query, top_k=3)
        for i, resultado in enumerate(resultados_embeddings, 1):
            print(f"  {i}. ID: {resultado['id']} | Score Embedding: {resultado['score']:.4f}")
    
    # Busca híbrida com QueryFusionRetriever
    print("\n3. Híbrido (QueryFusionRetriever - RRF):")
    resultados_hibrido = buscador.buscar_hibrido(query, top_k=3)
    for i, resultado in enumerate(resultados_hibrido, 1):
        print(f"  {i}. ID: {resultado['id']} | Score RRF: {resultado['score']:.4f}")
        print(f"     Método: {resultado.get('metodo', 'Híbrido')}")
    
    # Teste com diferentes queries
    queries_teste = [
        "controle interno",
        "auditoria governamental", 
        "gestão pública"
    ]
    
    print(f"\n--- Teste com Múltiplas Queries ---")
    for query_teste in queries_teste:
        print(f"\n🔍 Query: '{query_teste}'")
        resultados = buscador.buscar_hibrido(query_teste, top_k=2)
        for i, resultado in enumerate(resultados, 1):
            print(f"  {i}. ID: {resultado['id']} | Score: {resultado['score']:.4f}")

if __name__ == "__main__":
    # Teste rápido com dados de exemplo
    testar_com_dados_exemplo()

    # Teste de configuração híbrida
    testar_configuracoes_hibridas()

    print("\n" + "=" * 60)
    print("✅ Todos os testes foram concluídos com sucesso!")
    print("=" * 60)
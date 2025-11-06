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
from src.utils.dados import carregar_dados_juris_tcu, criar_dados_exemplo

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

def testar_com_dados_reais():
    """Testa o sistema com dados reais do jurisTCU"""
    print("\n" + "=" * 60)
    print("TESTE COM DADOS REAIS - JURISTCU (100 DOCUMENTOS)")
    print("=" * 60)
    
    # Verificar se o arquivo existe
    caminho_csv = "dados/juris_tcu/doc.csv"
    if not os.path.exists(caminho_csv):
        print(f"❌ Arquivo não encontrado: {caminho_csv}")
        print("   Certifique-se de que o dataset jurisTCU está disponível")
        return
    
    # Criar buscador
    buscador = BuscadorHibridoLlamaIndex()
    
    # Carregar dados reais (100 primeiros)
    print(f"\n📂 Carregando dados de: {caminho_csv}")
    documentos = carregar_dados_juris_tcu(caminho_csv, limite=100)
    
    if not documentos:
        print("❌ Não foi possível carregar os documentos")
        return
    
    buscador.carregar_documentos(documentos)
    
    # Mostrar estatísticas do dataset
    print(f"\n📊 Estatísticas do Dataset:")
    print(f"  - Total de documentos: {len(documentos)}")
    
    # Calcular estatísticas de texto
    tamanhos_enunciado = [len(doc.enunciado) for doc in documentos]
    tamanhos_excerto = [len(doc.excerto) for doc in documentos]
    
    print(f"  - Tamanho médio do enunciado: {sum(tamanhos_enunciado)/len(tamanhos_enunciado):.1f} caracteres")
    print(f"  - Tamanho médio do excerto: {sum(tamanhos_excerto)/len(tamanhos_excerto):.1f} caracteres")
    print(f"  - Maior enunciado: {max(tamanhos_enunciado)} caracteres")
    print(f"  - Maior excerto: {max(tamanhos_excerto)} caracteres")
    
    # Queries de teste específicas para dados jurídicos
    queries_juridicas = [
        "responsabilidade fiscal",
        "auditoria contas públicas",
        "licitação pública",
        "controle interno",
        "prestação de contas"
    ]
    
    for query in queries_juridicas:
        print(f"\n🔍 Testando query: '{query}'")
        print("-" * 50)
        
        # Busca BM25
        print("\n📊 Top 5 Resultados BM25:")
        resultados_bm25 = buscador.buscar_bm25(query, top_k=5)
        if resultados_bm25:
            for i, resultado in enumerate(resultados_bm25, 1):
                print(f"  {i}. ID: {resultado['id']} | Score: {resultado['score']:.4f}")
                enunciado_limpo = resultado['enunciado'].replace('<p>', '').replace('</p>', '')
                print(f"     Enunciado: {enunciado_limpo[:100]}...")
        else:
            print("  Nenhum resultado encontrado")
        
        # Busca por embeddings (se disponível)
        if buscador.vector_retriever:
            print("\n🧠 Top 5 Resultados Embeddings:")
            resultados_embeddings = buscador.buscar_embeddings(query, top_k=5)
            if resultados_embeddings:
                for i, resultado in enumerate(resultados_embeddings, 1):
                    print(f"  {i}. ID: {resultado['id']} | Score: {resultado['score']:.4f}")
                    enunciado_limpo = resultado['enunciado'].replace('<p>', '').replace('</p>', '')
                    print(f"     Enunciado: {enunciado_limpo[:100]}...")
            else:
                print("  Nenhum resultado encontrado")
        else:
            print("\n🧠 Embeddings: Não disponível (erro na configuração)")
        
        # Busca híbrida
        print("\n🔄 Top 5 Resultados Híbridos:")
        resultados_hibrido = buscador.buscar_hibrido(query, top_k=5)
        if resultados_hibrido:
            for i, resultado in enumerate(resultados_hibrido, 1):
                print(f"  {i}. ID: {resultado['id']} | Score Final: {resultado['score']:.4f}")
                print(f"     Método: {resultado.get('metodo', 'Híbrido')}")
                conteudo_limpo = resultado.get('conteudo', resultado.get('titulo', '')).replace('<p>', '').replace('</p>', '')
                print(f"     Conteúdo: {conteudo_limpo[:100]}...")
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
    # Teste rápido com dados de exemplo
    testar_com_dados_exemplo()
    
    # Teste de configuração híbrida
    testar_configuracoes_hibridas()

    # Teste com dados reais
    testar_com_dados_reais()

    # Teste do Reranker
    testar_reranker_com_dados_reais()

    print("\n" + "=" * 60)
    print("✅ Todos os testes foram concluídos com sucesso!")
    print("=" * 60)
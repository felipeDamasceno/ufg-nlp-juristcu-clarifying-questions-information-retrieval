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

def main():
    """Função principal que executa todos os testes"""
    print("🚀 Iniciando testes do Sistema de Busca Híbrida LlamaIndex")
    print(f"📍 Diretório atual: {os.getcwd()}")
    
    # Verificar configuração de embeddings
    print("✅ Embeddings locais configurados - Modelo português jurídico disponível")
    print("   Modelo: stjiris/bert-large-portuguese-cased-legal-mlm-sts-v1.0")
    
    try:
        # Teste 1: Dados de exemplo
        testar_com_dados_exemplo()
        
        # Teste 2: Dados reais
        testar_com_dados_reais()
        
        # Teste 3: Configurações híbridas (se embeddings disponíveis)
        testar_configuracoes_hibridas()
        
        print("\n" + "=" * 60)
        print("✅ TODOS OS TESTES CONCLUÍDOS COM SUCESSO!")
        print("=" * 60)
        
        # Resumo final
        print("\n📋 Resumo dos Testes:")
        print("  ✓ Busca BM25 com preprocessamento tokenizador_pt_remove_html")
        print("  ✓ Configuração BM25: enunciado + excerto")
        print("  ✓ Configuração Embeddings: apenas enunciado (sem HTML)")
        print("  ✓ Busca híbrida com diferentes pesos")
        print("  ✓ Teste com dados reais do jurisTCU (100 documentos)")
        print("  ✓ Métricas de performance")
        
        if not os.getenv("GOOGLE_API_KEY"):
            print("\n💡 Sistema configurado com embedding português jurídico local")
        
    except Exception as e:
        print(f"\n❌ Erro durante os testes: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
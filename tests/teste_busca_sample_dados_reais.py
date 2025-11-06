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


if __name__ == "__main__":

    # Teste com dados reais
    testar_com_dados_reais()


    print("\n" + "=" * 60)
    print("✅ Todos os testes foram concluídos com sucesso!")
    print("=" * 60)
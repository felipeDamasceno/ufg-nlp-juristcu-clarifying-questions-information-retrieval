import sys
import os

# Adicionar o diretório src ao sys.path para encontrar os módulos
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from buscador_hibrido import BuscadorHibridoLlamaIndex

def teste_calculo_similaridade():
    """
    Testa a função calcular_similaridade_entre_pares com dados simulados.
    """
    print("--- Iniciando Teste de Similaridade entre Pares ---")

    # 1. Inicializar o buscador para ter acesso ao modelo de embedding
    try:
        buscador = BuscadorHibridoLlamaIndex()
        if not buscador.embeddings_model:
            print("✗ Teste abortado: Modelo de embedding não foi carregado.")
            return
    except Exception as e:
        print(f"✗ Falha ao inicializar o BuscadorHibridoLlamaIndex: {e}")
        return

    # 2. Criar um resultado de busca simulado (fake)
    resultados_fake = [
        {
            "id": "doc1",
            "conteudo": "O tribunal de contas analisou o processo de licitação.",
            "score": 0.9,
            "metodo": "Híbrido"
        },
        {
            "id": "doc2",
            "conteudo": "A corte de contas examinou o procedimento licitatório.",
            "score": 0.88,
            "metodo": "Híbrido"
        },
        {
            "id": "doc3",
            "conteudo": "O relator apresentou seu voto na sessão plenária.",
            "score": 0.85,
            "metodo": "Híbrido"
        },
        {
            "id": "doc4",
            "conteudo": "O processo de licitação foi cuidadosamente analisado pelo tribunal de contas.",
            "score": 0.91,
            "metodo": "Híbrido"
        },
        {
            "id": "doc5",
            "conteudo": "A auditoria interna revelou diversas irregularidades nos contratos.",
            "score": 0.82,
            "metodo": "Híbrido"
        }
    ]
    print(f"✓ {len(resultados_fake)} resultados de busca simulados foram criados.")

    # 3. Chamar o método para calcular a similaridade
    print("\nCalculando a similaridade entre os pares...")
    pares_similares = buscador.calcular_similaridade_entre_pares(
        resultados_busca=resultados_fake,
        limite_similaridade=0.8,
        top_k=3
    )

    # 4. Verificar e exibir os resultados
    if pares_similares is None:
        print("\nResultado: Nenhum par com similaridade > 0.8 foi encontrado.")
    else:
        print(f"\n✓ Foram encontrados {len(pares_similares)} pares com similaridade > 0.8:")
        for i, par in enumerate(pares_similares):
            print(f"\n--- Par Similar #{i+1} ---")
            print(f"  Similaridade: {par['similaridade']:.4f}")
            print(f"  Documento 1 (ID: {par['documento_1']['id']}): \"{par['documento_1']['conteudo']}\"")
            print(f"  Documento 2 (ID: {par['documento_2']['id']}): \"{par['documento_2']['conteudo']}\"")

    print("\n--- Teste de Similaridade Concluído ---")

if __name__ == "__main__":
    teste_calculo_similaridade()
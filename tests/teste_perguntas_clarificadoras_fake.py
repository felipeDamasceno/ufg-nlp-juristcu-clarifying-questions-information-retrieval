import sys
import os

# Adicionar o diretório raiz do projeto ao sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.buscador_hibrido import BuscadorHibridoLlamaIndex
from src.clarifying_questions import gerar_perguntas_clarificadoras_para_pares


def teste_perguntas_clarificadoras_fake():
    """
    Teste com dados fake que:
    1) Calcula similaridade entre pares
    2) Gera perguntas clarificadoras (uma por par) usando Gemini ou fallback
    3) Imprime as perguntas para inspeção
    """
    print("--- Iniciando Teste de Perguntas Clarificadoras com Dados Fake ---")

    # 1. Inicializar o buscador para ter acesso ao modelo de embedding
    try:
        buscador = BuscadorHibridoLlamaIndex()
        if not buscador.embeddings_model:
            print("✗ Teste abortado: Modelo de embedding não foi carregado.")
            return
    except Exception as e:
        print(f"✗ Falha ao inicializar o BuscadorHibridoLlamaIndex: {e}")
        return

    # 2. Criar dados fake (mesma estrutura do teste de similaridade)
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

    # 3. Calcular similaridade entre pares e pegar top-3
    print("\nCalculando a similaridade entre os pares...")
    pares_similares = buscador.calcular_similaridade_entre_pares(
        resultados_busca=resultados_fake,
        limite_similaridade=0.8,
        top_k=3
    )

    if not pares_similares:
        print("\nResultado: Nenhum par com similaridade > 0.8 foi encontrado.")
        print("--- Teste encerrado sem geração de perguntas ---")
        return

    print(f"\n✓ Foram encontrados {len(pares_similares)} pares com similaridade > 0.8.")

    # 4. Gerar perguntas clarificadoras para os top-3 pares
    conversa_exemplo = (
        "Usuário: Estou analisando casos de licitação e auditoria no TCU. "
    )

    print("\nGerando perguntas clarificadoras (uma por par)...")
    try:
        perguntas = gerar_perguntas_clarificadoras_para_pares(
            pares_similares=pares_similares,
            conversa=conversa_exemplo,
            max_perguntas=3,
        )

        # Exibir par de documentos e pergunta
        for item in perguntas:
            idx = item.get("par_index")
            pergunta = item.get("pergunta")
            resposta_completa = item.get("resposta_completa")
            par = pares_similares[idx]
            doc1 = par.get("documento_1", {})
            doc2 = par.get("documento_2", {})
            print(f"\n--- Par Similar #{idx+1} ---")
            print(f"  Documento 1 (ID: {doc1.get('id')}): \"{doc1.get('conteudo', doc1.get('enunciado', ''))}\"")
            print(f"  Documento 2 (ID: {doc2.get('id')}): \"{doc2.get('conteudo', doc2.get('enunciado', ''))}\"")
            print(f"  Resposta completa (Gemini): {resposta_completa}")
            print(f"  Pergunta (extraída): {pergunta}")
    except RuntimeError as e:
        print("\n✗ Erro ao gerar perguntas via Gemini:")
        print(f"  Detalhes: {e}")
        print("  Dicas: verifique se 'google-generativeai' está instalado e se 'GOOGLE_API_KEY' está configurada.")

    print("\n--- Teste de Perguntas Clarificadoras Concluído ---")


if __name__ == "__main__":
    teste_perguntas_clarificadoras_fake()
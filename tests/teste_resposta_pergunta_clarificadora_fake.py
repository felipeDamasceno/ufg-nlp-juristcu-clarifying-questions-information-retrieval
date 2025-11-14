import os
import sys

from dotenv import load_dotenv

# Carregar .env para chave de API
load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.resposta_clarificadora import responder_pergunta_clarificadora


def main():
    print("=== Teste de resposta a perguntas clarificadoras (fake) ===")

    # Caso 1: BACKGROUND com informação suficiente
    intent_com_info = (
        "Necessidade de informação: entender o alcance da sanção de impedimento de licitar e contratar. "
        "Pontos que devem constar nos resultados: aplica-se em toda a Administração Pública (direta e indireta), "
        "abrange União, Estados, DF e Municípios, e inclui empresas estatais (empresas públicas e sociedades de economia mista)."
    )
    perguntas_info = [
        "Essa sanção vale apenas no órgão que aplicou ou em toda a administração?",
        "Empresas estatais também são abrangidas por essa sanção?",
        "Se aplicada por um estado, ela se estende à União?",
    ]

    print("\n--- Caso com informação no BACKGROUND ---")
    print(f"BACKGROUND: {intent_com_info}\n")
    for i, p in enumerate(perguntas_info, start=1):
        try:
            resposta = responder_pergunta_clarificadora(intent_com_info, p)
            print(f"[{i}] Pergunta: {p}")
            print(f"    Resposta: {resposta}\n")
        except Exception as e:
            print(f"[{i}] Pergunta: {p}")
            print(f"    Falha ao responder: {e}\n")

    # Caso 2: BACKGROUND sem informação suficiente
    intent_sem_info = (
        "Resumo desejado: entendimento geral sobre a sanção de impedimento de licitar e contratar. "
        "Não contém detalhes sobre abrangência entre órgãos, esferas federativas ou empresas estatais."
    )
    perguntas_sem_info = [
        "Essa sanção vale apenas no órgão que aplicou ou em toda a administração?",
    ]

    print("\n--- Caso sem informação no BACKGROUND ---")
    print(f"BACKGROUND: {intent_sem_info}\n")
    for i, p in enumerate(perguntas_sem_info, start=1):
        try:
            resposta = responder_pergunta_clarificadora(intent_sem_info, p)
            print(f"[{i}] Pergunta: {p}")
            print(f"    Resposta: {resposta}\n")
        except Exception as e:
            print(f"[{i}] Pergunta: {p}")
            print(f"    Falha ao responder: {e}\n")


if __name__ == "__main__":
    main()
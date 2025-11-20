import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clarifying_questions import gerar_perguntas_sem_pares


def teste_perguntas_sem_pares_fake():
    print("--- Teste de Perguntas Clarificadoras SEM pares ---")
    pergunta = "Usuário: Pregão eletrônico para serviços comuns de engenharia."
    try:
        perguntas = gerar_perguntas_sem_pares(pergunta=pergunta, max_perguntas=3)
        if not perguntas:
            print("(Nenhuma pergunta gerada)")
            return
        for item in perguntas:
            print(f"Pergunta: {item.get('pergunta')}")
    except Exception as e:
        print(f"✗ Erro ao gerar perguntas sem pares: {e}")


if __name__ == "__main__":
    teste_perguntas_sem_pares_fake()
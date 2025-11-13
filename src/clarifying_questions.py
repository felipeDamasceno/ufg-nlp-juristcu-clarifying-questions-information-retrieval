from typing import List, Dict, Optional
import os
import json

try:
    import google.generativeai as genai  # type: ignore
    _HAS_GEMINI = True
except Exception:
    _HAS_GEMINI = False

from src.similaridade import _texto_do_resultado
from src.utils.gemini import configurar_gemini, strip_code_fences, extrair_texto_resposta

from dotenv import load_dotenv

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()


def _formatar_prompt(conversa: str, caso1: str, caso2: str) -> str:
    """
    Formata o prompt conforme o template solicitado pelo usuário.
    """
    return (
        "You are now a knowledgeable judge in law. The current conversation between you and the user\n"
        f"is as follows: [{conversa}]. Based on the above conversation, what clarifying question can\n"
        "you ask to further understand the background information of the case?\n"
        "Specifically, there are similar cases with the following circumstances: Case 1: "
        f"[{caso1}]. Case 2: [{caso2}].\n"
        "Identify the differences between Case 1 and Case 2, and generate the clarifying question\n"
        "based on the differences in Portuguese Brazil.\n\n"
        "Return ONLY a single-line JSON object with the following structure:\n"
        "{\"question\": \"<uma pergunta clara em PT-BR, uma frase>\", \"rationale\": \"<diferenciação entre o caso 1 e o caso 2 e racional por traz da pergunta gerada>\"}.\n"
        "Do not include anything else besides the JSON."
    )


def _gerar_via_gemini(prompt: str) -> Dict[str, str]:
    """
    Gera uma pergunta clarificadora via Gemini e retorna um dict:
    { 'full_text': <resposta bruta do modelo>, 'question': <pergunta extraída> }.
    Levanta RuntimeError em caso de falha.
    """
    configurar_gemini()

    # Permite customizar o nome do modelo via env; define um padrão se não houver.
    model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash-lite")
    try:
        # Configuração para forçar saída JSON, e tentar schema estruturado
        schema = {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "rationale": {"type": "string"},
            },
            "required": ["question", "rationale"],
        }

        gen_config = {"response_mime_type": "application/json", "response_schema": schema}
        model = genai.GenerativeModel(model_name, generation_config=gen_config)
        try:
            resp = model.generate_content(prompt, generation_config=gen_config)
        except Exception as e:
            raise RuntimeError(f"Falha ao gerar conteúdo com Gemini: {e}")

        # Extrai texto da resposta usando utilitário compartilhado
        full_text = extrair_texto_resposta(resp)
        if full_text is None:
            raise RuntimeError("Resposta do Gemini não contém texto gerado.")

        # Normaliza: remove cercas de código se houver
        full_text = strip_code_fences(full_text)

        # Tenta fazer parsing do JSON e extrair 'question'
        try:
            data = json.loads(full_text)
        except Exception as e:
            raise RuntimeError(f"Resposta do Gemini não está em JSON válido: {e}. Conteúdo: {full_text}")

        question = data.get("question")
        if not question or not isinstance(question, str):
            raise RuntimeError(f"Campo 'question' ausente ou inválido no JSON. Conteúdo: {full_text}")

        return {"full_text": full_text, "question": question.strip()}
    except Exception as e:
        raise RuntimeError(f"Falha ao gerar conteúdo com Gemini: {e}")




def gerar_perguntas_clarificadoras_para_pares(
    pares_similares: List[Dict],
    conversa: str,
    max_perguntas: int = 3,
) -> List[Dict]:
    """
    Gera perguntas clarificadoras para os top-N pares similares.

    Args:
        pares_similares: Lista de pares retornados por calcular_similaridade_entre_pares.
        conversa: Texto da conversa atual.
        max_perguntas: Quantidade máxima de perguntas a gerar (uma por par).

    Returns:
        Lista com dicts: { 'par_index': int, 'pergunta': str, 'origem': 'gemini' }
    """
    resultados: List[Dict] = []
    if not pares_similares:
        return resultados

    limite = min(max_perguntas, len(pares_similares))
    for i in range(limite):
        par = pares_similares[i]
        doc1 = par.get("documento_1", {})
        doc2 = par.get("documento_2", {})
        caso1 = _texto_do_resultado(doc1)
        caso2 = _texto_do_resultado(doc2)

        prompt = _formatar_prompt(conversa, caso1, caso2)
        resultado = _gerar_via_gemini(prompt)
        origem = "gemini"

        resultados.append({
            "par_index": i,
            "pergunta": resultado["question"],
            "resposta_completa": resultado["full_text"],
            "origem": origem,
        })

    return resultados
"""
Responde perguntas clarificadoras com base em uma intenção de busca
utilizando o utilitário centralizado de Gemini.

Função principal:
    responder_pergunta_clarificadora(query_intencao: str, pergunta: str) -> dict

Retorno:
    { "answer": <texto da resposta> }

Uso de prompt solicitado:
    "Please read the following background information: [the query intention].
     And answer the following question: [clarifying question]."
"""

from typing import Dict
import os
import json

# Tentativa de importar o SDK do Gemini
try:
    import google.generativeai as genai  # type: ignore
    _HAS_GEMINI = True
except Exception:
    _HAS_GEMINI = False

from src.utils.gemini import configurar_gemini, extrair_texto_resposta, strip_code_fences


def responder_pergunta_clarificadora(query_intencao: str, pergunta: str) -> str:
    if not query_intencao or not pergunta:
        raise ValueError("query_intencao e pergunta são obrigatórias")

    configurar_gemini()

    prompt = (
        "Você receberá um BACKGROUND (descrição da necessidade de informação da busca) e uma PERGUNTA.\n"
        "Responda EXCLUSIVAMENTE com base no BACKGROUND: cite apenas afirmações explícitas nele.\n"
        "Se a resposta NÃO estiver coberta pelo BACKGROUND, responda exatamente 'não sei'.\n"
        "Não invente, não use conhecimento externo, não reinterprete.\n\n"
        "BACKGROUND:\n" + str(query_intencao) + "\n\n"
        "PERGUNTA:\n" + str(pergunta) + "\n\n"
        "Retorne APENAS um JSON de uma linha com o seguinte formato: "
        '{"answer": "<resposta direta em PT-BR baseada no BACKGROUND ou \"não sei\">"}.\n'
        "Não inclua nada além do JSON."
    )

    model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash-lite")
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
        },
        "required": ["answer"],
    }

    gen_config = {
        "response_mime_type": "application/json",
        "response_schema": schema,
        "temperature": 0.2,
        "max_output_tokens": 1024,
    }
    model = genai.GenerativeModel(model_name, generation_config=gen_config)

    response = model.generate_content(prompt, generation_config=gen_config)
    texto = extrair_texto_resposta(response)
    texto = strip_code_fences(texto)

    # Parsear JSON e extrair somente o campo 'answer'
    try:
        data = json.loads(texto)
    except Exception as e:
        raise RuntimeError(f"Resposta do Gemini não está em JSON válido: {e}. Conteúdo: {texto}")

    answer = data.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError(f"Campo 'answer' ausente ou inválido no JSON. Conteúdo: {texto}")

    return answer.strip()
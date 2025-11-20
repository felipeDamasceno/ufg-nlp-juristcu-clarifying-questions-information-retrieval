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
import time
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
        "Você atua como um USUÁRIO realizando uma busca. Você receberá:\n"
        "1. INTENÇÃO DE BUSCA (o que você quer descobrir/pesquisar).\n"
        "2. PERGUNTA DO SISTEMA (uma pergunta feita para refinar sua busca).\n\n"
        
        "SUA TAREFA: Responder à PERGUNTA DO SISTEMA usando a INTENÇÃO DE BUSCA e apenas ela como guia.\n\n"
        
        "REGRAS DE RESPOSTA:\n"
        "- O texto da INTENÇÃO DE BUSCA contém os tópicos que te interessam. NÃO trate esse texto como uma resposta pronta que você já sabe. Trate como o TEMA que você quer confirmar.\n"
        "- Se a PERGUNTA tocar em um ponto que está explícito na INTENÇÃO DE BUSCA, responda confirmando o interesse. (Ex: 'Gostaria de saber sobre [tópico citado no texto]' ou 'Procuro informações sobre [trecho do texto]').\n"
        "- Se a PERGUNTA tocar em um ponto não mencionado explicitamente na INTENÇÃO DE BUSCA, responda não confirmando o interesse. (Ex: 'Não gostaria de saber sobre [tópico citado no texto]' ou 'Não procuro informações sobre [trecho do texto]').\n"
        "- Mantenha a resposta curta e natural, não responda nada além do que foi perguntado.\n\n"
        
        "INTENÇÃO DE BUSCA:\n" + str(query_intencao) + "\n\n"
        "PERGUNTA DO SISTEMA:\n" + str(pergunta) + "\n\n"
        
        "Retorne APENAS um JSON de uma linha com o seguinte formato: "
        '{"answer": "<sua resposta>"}.\n'
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
        "temperature": 0,
        "max_output_tokens": 1024,
    }
    model = genai.GenerativeModel(model_name, generation_config=gen_config)

    response = model.generate_content(prompt, generation_config=gen_config)
    time.sleep(1)
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
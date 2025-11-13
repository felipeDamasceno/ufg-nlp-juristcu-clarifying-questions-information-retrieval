"""
Módulo para geração de intenção de busca usando Gemini.

Função principal:
    gerar_intencao_busca(query_text: str, docs_ideais: List[str]) -> Dict[str, str]

Retorna dicionário com:
    - full_text: resposta completa do modelo (JSON em string)
    - intent: intenção de busca extraída
    - rationale: (opcional) justificativa se presente
"""

from typing import List, Dict
import os
import json

# Tentativa de importar o SDK do Gemini; mantém flag de disponibilidade
try:
    import google.generativeai as genai  # type: ignore
    _HAS_GEMINI = True
except Exception:
    _HAS_GEMINI = False

from dotenv import load_dotenv
from src.utils.gemini import configurar_gemini, strip_code_fences, extrair_texto_resposta

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()


# Removido: funções duplicadas de configuração e strip; usar utilitário compartilhado


def _formatar_prompt_intencao(query_text: str, docs_ideais: List[str]) -> str:
    bullets = "\n".join([f"- {d}" for d in docs_ideais])
    return (
        "Você é um assistente jurídico em PT-BR.\n"
        "A seguir está a pergunta original do usuário e os documentos julgados ideais para a busca da pergunta (score 3).\n"
        f"Pergunta: {query_text}\n"
        "Documentos ideais:\n"
        f"{bullets}\n\n"
        "Com base nisso, escreva a intenção de pergunta/busca do usuário de forma detalhada sem informar a exata lei ou artigo:\n"
        "{\"intent\": \"<Intensão de busca em detalhes em PT-BR>\", \"rationale\": \"<por que esta é a intenção com base nos docs>\"}"
    )


def gerar_intencao_busca(query_text: str, docs_ideais: List[str]) -> Dict[str, str]:
    """
    Gera a intenção de busca via Gemini com base na pergunta original e nos documentos ideais.

    Retorna dict com 'full_text', 'intent' e (se presente) 'rationale'.
    """
    configurar_gemini()

    model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash-lite")

    # Schema de saída JSON estruturado
    schema = {
        "type": "object",
        "properties": {
            "intent": {"type": "string"},
            "rationale": {"type": "string"},
        },
        "required": ["intent"],
    }

    gen_config = {"response_mime_type": "application/json", "response_schema": schema}
    model = genai.GenerativeModel(model_name, generation_config=gen_config)

    prompt = _formatar_prompt_intencao(query_text, docs_ideais)
    try:
        resp = model.generate_content(prompt, generation_config=gen_config)
    except Exception as e:
        raise RuntimeError(f"Falha ao gerar conteúdo com Gemini: {e}")

    # Extrair texto da resposta com utilitário
    full_text = extrair_texto_resposta(resp)
    if full_text is None:
        raise RuntimeError("Resposta do Gemini não contém texto gerado.")

    full_text = strip_code_fences(full_text)
    try:
        data = json.loads(full_text)
    except Exception as e:
        raise RuntimeError(f"Resposta do Gemini não está em JSON válido: {e}. Conteúdo: {full_text}")

    intent = data.get("intent")
    rationale = data.get("rationale")
    if not intent or not isinstance(intent, str):
        raise RuntimeError(f"Campo 'intent' ausente ou inválido no JSON. Conteúdo: {full_text}")

    result: Dict[str, str] = {"full_text": full_text, "intent": intent.strip()}
    if isinstance(rationale, str):
        result["rationale"] = rationale.strip()
    return result
from typing import List, Dict, Optional
import os
import json
import time

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


def _formatar_prompt(conversa, caso1, caso2):
    return (
        f"""
Você agora é um juiz especializado em Direito. A conversa atual é: [{conversa}].

Com base na consulta do usuário, uma busca inicial retornou dois documentos semelhantes, porém distintos:
Documento 1: [{caso1}]
Documento 2: [{caso2}]

Sua tarefa é:
1. **Analisar** e identificar a **única diferença mais importante** (fática ou jurídica) entre o Documento 1 e o Documento 2.
2. Com base **exclusivamente** nessa diferença-chave, gerar uma pergunta única e clara de esclarecimento **em português do Brasil (PT-BR)**.

O objetivo dessa pergunta é ajudar o usuário a especificar sua intenção, permitindo entender qual contexto documental é mais relevante para a situação dele.

Retorne APENAS um objeto JSON de uma única linha com a seguinte estrutura:
{{"question": "<uma pergunta clara em PT-BR, uma frase>", "rationale": "<A diferença principal entre o caso 1 e o caso 2 e o racional por trás da pergunta gerada>"}}

Não inclua nada além do JSON.
"""
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

        gen_config = {"response_mime_type": "application/json", "response_schema": schema, "temperature": 0}
        model = genai.GenerativeModel(model_name, generation_config=gen_config)
        try:
            resp = model.generate_content(prompt, generation_config=gen_config)
        except Exception as e:
            raise RuntimeError(f"Falha ao gerar conteúdo com Gemini: {e}")
        time.sleep(1)

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




def _formatar_prompt_sem_pares(pergunta: str, n: int = 3) -> str:
    return (
        f"""
Você é um assistente de IA especialista em Direito Brasileiro. A consulta original do usuário é: [{pergunta}].

Com base nessa consulta, gere {n} perguntas clarificadoras em Português do Brasil que ajudem a identificar a necessidade específica do usuário.

Cada pergunta deve explorar uma diferença chave de interpretação ou de situação prática relevante à consulta, sem mencionar documentos específicos.

Retorne APENAS um array JSON com {n} objetos no formato:
[{{"question": "<pergunta em PT-BR>", "rationale": "<racional sucinto>"}}, ...]

Não inclua nada além do JSON.
"""
    )


def gerar_perguntas_sem_pares(pergunta: str, max_perguntas: int = 3) -> List[Dict]:
    configurar_gemini()
    model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash-lite")
    n = max(1, max_perguntas)
    prompt = _formatar_prompt_sem_pares(pergunta, n)
    try:
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["question", "rationale"],
            },
        }
        gen_config = {"response_mime_type": "application/json", "response_schema": schema, "temperature": 0}
        model = genai.GenerativeModel(model_name, generation_config=gen_config)
        resp = model.generate_content(prompt, generation_config=gen_config)
        time.sleep(1)
        full_text = extrair_texto_resposta(resp)
        if full_text is None:
            raise RuntimeError("Resposta do Gemini não contém texto gerado.")
        full_text = strip_code_fences(full_text)
        data = json.loads(full_text)
        if not isinstance(data, list):
            raise RuntimeError("Resposta não é um array JSON.")
        resultados: List[Dict] = []
        for i, item in enumerate(data[:n]):
            q = item.get("question")
            if not q or not isinstance(q, str):
                continue
            resultados.append({
                "par_index": i,
                "pergunta": q.strip(),
                "resposta_completa": full_text,
                "origem": "gemini",
            })
        return resultados
    except Exception as e:
        raise RuntimeError(f"Falha ao gerar perguntas sem pares: {e}")

def gerar_perguntas_clarificadoras_para_pares(
    pares_similares: List[Dict],
    conversa: str,
    max_perguntas: int = 3,
) -> List[Dict]:
    """
    Gera perguntas clarificadoras para os top-N pares similares.

    Args:
        pares_similares: Lista de pares retornados por calcular_similaridade_entre_pares.
        conversa: Texto da conversa atual com o usuário.
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
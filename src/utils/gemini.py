"""
Helpers compartilhados para integração com Gemini.

Inclui:
- configurar_gemini(): carrega .env e configura cliente com GOOGLE_API_KEY
- strip_code_fences(text): remove cercas de código Markdown
- extrair_texto_resposta(resp): extrai texto dos candidatos ou de resp.text
"""

import os
from typing import Optional

try:
    import google.generativeai as genai  # type: ignore
    _HAS_GEMINI = True
except Exception:
    _HAS_GEMINI = False

from dotenv import load_dotenv

# Sempre carregar variáveis de ambiente
load_dotenv()


def configurar_gemini() -> None:
    """Configura o cliente Gemini usando GOOGLE_API_KEY."""
    if not _HAS_GEMINI:
        raise RuntimeError("Pacote 'google-generativeai' não está instalado. Execute 'pip install -r requirements.txt'.")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Variável de ambiente 'GOOGLE_API_KEY' não definida. Configure sua chave da API Gemini.")
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        raise RuntimeError(f"Falha ao configurar o cliente Gemini: {e}")


def strip_code_fences(text: str) -> str:
    """Remove cercas de código Markdown (``` e ```json) do texto, se presentes."""
    s = (text or "").strip()
    if s.startswith("```"):
        first_newline = s.find("\n")
        if first_newline != -1:
            s = s[first_newline + 1:]
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


def extrair_texto_resposta(resp) -> Optional[str]:
    """Extrai texto de uma resposta do SDK do Gemini.

    Procura em `resp.candidates[*].content.parts[*].text` e faz fallback para `resp.text`.
    Retorna None se não encontrar.
    """
    full_text = None
    parts = getattr(resp, "candidates", [])
    if parts:
        for cand in parts:
            content = getattr(cand, "content", None)
            if content and getattr(content, "parts", None):
                for p in content.parts:
                    text = getattr(p, "text", None)
                    if text:
                        full_text = text.strip()
                        break
            if full_text:
                break
    if full_text is None:
        full_text = getattr(resp, "text", None)
        if isinstance(full_text, str):
            full_text = full_text.strip()
    return full_text
"""
Módulo para representação de documentos jurídicos do TCU
"""

from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class DocumentoJuris:
    """Estrutura para representar um documento jurídico do TCU"""
    id: str
    enunciado: str
    excerto: str
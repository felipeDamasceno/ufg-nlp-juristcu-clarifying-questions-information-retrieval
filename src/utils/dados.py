"""
Utilitários para carregamento e manipulação de dados do jurisTCU
"""

import pandas as pd
from typing import List

from src.documento import DocumentoJuris


def carregar_dados_juris_tcu(caminho_csv: str, limite: int = None) -> List[DocumentoJuris]:
    """
    Carrega dados do dataset jurisTCU
    
    Args:
        caminho_csv: Caminho para o arquivo CSV
        limite: Número máximo de documentos a carregar (None para todos)
        
    Returns:
        Lista de documentos jurídicos
    """
    try:
        df = pd.read_csv(caminho_csv)
        
        if limite:
            df = df.head(limite)
        
        documentos = []
        for idx, row in df.iterrows():
            
            enunciado = str(row.get('ENUNCIADO', ''))
            excerto = str(row.get('EXCERTO', ''))
            
            doc = DocumentoJuris(
                id=str(row.get('KEY', idx)),
                enunciado=enunciado,
                excerto=excerto  
            )
            documentos.append(doc)
        
        print(f"✓ {len(documentos)} documentos carregados do arquivo {caminho_csv}")
        return documentos
        
    except Exception as e:
        print(f"✗ Erro ao carregar dados: {e}")
        return []


def criar_dados_exemplo() -> List[DocumentoJuris]:
    """Cria dados de exemplo para teste"""
    return [
        DocumentoJuris(
            id="1",
            enunciado="<p>Responsabilidade fiscal na administração pública</p>",
            excerto="A responsabilidade fiscal é fundamental para a gestão pública eficiente e transparente."
        ),
        DocumentoJuris(
            id="2", 
            enunciado="<p>Auditoria de contas públicas</p>",
            excerto="As auditorias devem seguir normas técnicas específicas para garantir a qualidade dos trabalhos."
        ),
        DocumentoJuris(
            id="3",
            enunciado="<p>Controle interno e externo</p>",
            excerto="O sistema de controle deve ser integrado e efetivo para prevenir irregularidades."
        )
    ]
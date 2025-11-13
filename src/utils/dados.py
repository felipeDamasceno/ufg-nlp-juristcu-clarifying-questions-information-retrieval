"""
Utilitários para carregamento e manipulação de dados do jurisTCU.

Inclui loaders reutilizáveis para `query.csv`, `qrel.csv` e `doc.csv`.
"""

import pandas as pd
from typing import List, Dict

from src.documento import DocumentoJuris
from src.utils.preprocessamento import PreprocessadorTexto


def carregar_dados_juris_tcu(caminho_csv: str, limite: int = None) -> List[DocumentoJuris]:
    """Carrega documentos do CSV (KEY, ENUNCIADO, EXCERTO) em objetos DocumentoJuris."""
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
        return documentos
    except Exception:
        return []


def load_queries_df(path: str) -> pd.DataFrame:
    """Carrega `query.csv` como DataFrame com ID numérico e TEXT/SOURCE como string."""
    df = pd.read_csv(path, dtype=str, encoding="utf-8").fillna("")
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce")
    df = df.dropna(subset=["ID"]).astype({"ID": int})
    return df


def load_qrels_df(path: str) -> pd.DataFrame:
    """Carrega `qrel.csv` como DataFrame tipado e pronto para filtros e ordenações."""
    df = pd.read_csv(path, encoding="utf-8")
    df["QUERY_ID"] = pd.to_numeric(df["QUERY_ID"], errors="coerce")
    df["DOC_ID"] = pd.to_numeric(df["DOC_ID"], errors="coerce")
    df["SCORE"] = pd.to_numeric(df["SCORE"], errors="coerce").fillna(0)
    df["RANK"] = pd.to_numeric(df["RANK"], errors="coerce").fillna(999999)
    df = df.dropna(subset=["QUERY_ID", "DOC_ID"]).astype({"QUERY_ID": int, "DOC_ID": int, "RANK": int})
    return df


def load_docs_enunciado_map_clean(path: str) -> Dict[int, str]:
    """Cria um mapa DOC_ID (numérico extraído de KEY) -> ENUNCIADO limpo (HTML removido)."""
    preproc = PreprocessadorTexto()
    df = pd.read_csv(path, dtype=str, encoding="utf-8").fillna("")
    df["NUM"] = df["KEY"].astype(str).str.extract(r"(\d+)$")
    df["NUM"] = pd.to_numeric(df["NUM"], errors="coerce")
    df = df.dropna(subset=["NUM"]).astype({"NUM": int})
    df["ENUNCIADO_CLEAN"] = df["ENUNCIADO"].apply(lambda x: preproc.remove_html(x))
    return df.set_index("NUM")["ENUNCIADO_CLEAN"].to_dict()


def criar_dados_exemplo() -> List[DocumentoJuris]:
    """Cria dados de exemplo para teste."""
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
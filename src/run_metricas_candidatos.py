import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.utils.metricas import metricas
from src.utils.dados import load_qrels_df
DATA_DIR = os.path.join(BASE_DIR, "dados", "juris_tcu")
RESULT_CSV = os.path.join(BASE_DIR, "dados", "candidatos_top20_full.csv")
QRELS_CSV = os.path.join(DATA_DIR, "qrel.csv")
OUT_CSV = os.path.join(BASE_DIR, "dados", "metricas_candidatos_top10.csv")

def main():
    if not (os.path.exists(RESULT_CSV) and os.path.exists(QRELS_CSV)):
        print("Arquivos necessários não encontrados.")
        print(f"Resultado: {RESULT_CSV}")
        print(f"Qrels: {QRELS_CSV}")
        return

    resultado_df = pd.read_csv(RESULT_CSV, dtype={"QUERY_ID": int, "DOC_ID": str, "RERANK_SCORE": float, "RANK": int})

    # Extrai o número ao final do DOC_ID original (ex.: "JURISPRUDENCIA-SELECIONADA-85434" -> 85434)
    def _extract_numeric_doc_id(value: str):
        import re
        if pd.isna(value):
            return None
        m = re.search(r"(\d+)$", str(value))
        return int(m.group(1)) if m else None

    resultado_df["DOC_ID_NUM"] = resultado_df["DOC_ID"].apply(_extract_numeric_doc_id)
    resultado_df = resultado_df.dropna(subset=["DOC_ID_NUM"]).astype({"DOC_ID_NUM": int})
    qrels_df = load_qrels_df(QRELS_CSV)

    pd_metricas = metricas(
        resultado_pesquisa=resultado_df,
        qrels=qrels_df,
        col_resultado_query_key="QUERY_ID",
        col_resultado_doc_key="DOC_ID_NUM",
        col_resultado_rank="RANK",
        col_qrels_query_key="QUERY_ID",
        col_qrels_doc_key="DOC_ID",
        col_qrels_score="SCORE",
        k=[10],
        debug=False,
        aproximacao_trec_eval=False,
    )

    means = pd_metricas.drop(columns=["QUERY_KEY"]).mean(numeric_only=True)
    means_df = pd.DataFrame({"QUERY_KEY": ["MEAN"], **{col: [means[col]] for col in means.index}})
    pd_metricas_out = pd.concat([pd_metricas, means_df], ignore_index=True)
    pd_metricas_out.to_csv(OUT_CSV, index=False, encoding="utf-8")

    print(pd_metricas_out.to_string(index=False))
    print(f"Metricas salvas em: {OUT_CSV}")

if __name__ == "__main__":
    main()
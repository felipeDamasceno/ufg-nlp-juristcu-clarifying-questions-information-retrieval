import os
from src.utils.dados import carregar_dados_juris_tcu, load_queries_df
from src.candidatos import executar_busca_candidatos

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dados", "juris_tcu")
DOC_CSV = os.path.join(DATA_DIR, "doc.csv")
QUERY_CSV = os.path.join(DATA_DIR, "query.csv")
OUT_CSV = os.path.join(BASE_DIR, "dados", "candidatos_top20_full.csv")
PERSIST_DIR = os.path.join(BASE_DIR, "storage", "vector_index")

def main():
    if not (os.path.exists(DOC_CSV) and os.path.exists(QUERY_CSV)):
        print("Arquivos necessários não encontrados.")
        return

    documentos = carregar_dados_juris_tcu(DOC_CSV)
    queries_df = load_queries_df(QUERY_CSV)
    if queries_df.empty:
        print("Sem queries válidas.")
        return

    queries = queries_df.to_dict(orient="records")

    rows = executar_busca_candidatos(
        queries=queries,
        documentos=documentos,
        output_csv_path=OUT_CSV,
        persist_dir=PERSIST_DIR,
        bm25_top_k=50,
        embeddings_top_k=50,
        hybrid_top_k=50,
        rerank_top_n=20,
    )

    print(f"Total linhas salvas: {len(rows)}")
    print(f"Arquivo: {OUT_CSV}")

if __name__ == "__main__":
    main()
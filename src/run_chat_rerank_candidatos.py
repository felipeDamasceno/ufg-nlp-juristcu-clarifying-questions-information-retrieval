import os
import sys
import argparse
import pandas as pd
from typing import List, Dict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from src.utils.dados import load_queries_df, load_docs_enunciado_map_clean, load_qrels_df
from src.buscador_hibrido import BuscadorHibridoLlamaIndex
from src.reranking import rerank_nodes
from src.clarifying_questions import gerar_perguntas_clarificadoras_para_pares, gerar_perguntas_sem_pares
from src.resposta_clarificadora import responder_pergunta_clarificadora
from llama_index.core.schema import TextNode
from src.utils.metricas import metricas

DATA_DIR = os.path.join(BASE_DIR, "dados", "juris_tcu")
DOC_CSV = os.path.join(DATA_DIR, "doc.csv")
QUERY_CSV = os.path.join(DATA_DIR, "query.csv")
CANDIDATOS_CSV = os.path.join(BASE_DIR, "dados", "candidatos_top20_full.csv")
OUT_CSV_PAIRS = os.path.join(BASE_DIR, "dados", "candidatos_chat_top20.csv")
OUT_METRICAS_PAIRS = os.path.join(BASE_DIR, "dados", "metricas_candidatos_chat_top10.csv")
OUT_CSV_NO_PAIRS = os.path.join(BASE_DIR, "dados", "candidatos_chat_nodocs_top20.csv")
OUT_METRICAS_NO_PAIRS = os.path.join(BASE_DIR, "dados", "metricas_candidatos_chat_nodocs_top10.csv")
QUERY_INTENCAO_CSV = os.path.join(BASE_DIR, "dados", "query_intencao.csv")


def _extract_numeric_doc_id(value: str):
    import re
    if pd.isna(value):
        return None
    m = re.search(r"(\d+)$", str(value))
    return int(m.group(1)) if m else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modo", choices=["pares", "sem_pares"], default="pares")
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args()
    if not (os.path.exists(DOC_CSV) and os.path.exists(QUERY_CSV) and os.path.exists(CANDIDATOS_CSV) and os.path.exists(QUERY_INTENCAO_CSV)):
        print("Arquivos necessários não encontrados.")
        return

    queries_df = load_queries_df(QUERY_CSV)
    inten_df = pd.read_csv(QUERY_INTENCAO_CSV, encoding="utf-8")
    inten_df["INTENCAO"] = inten_df["INTENCAO"].astype(str).fillna("").str.strip()
    inten_df = inten_df[(inten_df["INTENCAO"] != "") & (inten_df["INTENCAO"].str.lower() != "nan")]
    docs_map = load_docs_enunciado_map_clean(DOC_CSV)
    candidatos_df = pd.read_csv(CANDIDATOS_CSV, dtype={"QUERY_ID": int, "DOC_ID": str, "RERANK_SCORE": float, "RANK": int})
    candidatos_df["DOC_ID_NUM"] = candidatos_df["DOC_ID"].apply(_extract_numeric_doc_id)
    candidatos_df = candidatos_df.dropna(subset=["DOC_ID_NUM"]).astype({"DOC_ID_NUM": int})

    buscador = BuscadorHibridoLlamaIndex()
    if not buscador.embeddings_model:
        print("✗ Modelo de embeddings não carregado.")
        return

    # Seleciona as 3 primeiras queries presentes no arquivo de candidatos
    all_ids = sorted(candidatos_df["QUERY_ID"].unique())
    if args.n and args.n > 0:
        query_ids = all_ids[:args.n]
    else:
        query_ids = all_ids

    all_rows: List[Dict] = []

    for idx, qid in enumerate(query_ids, start=1):
        qrow = queries_df[queries_df["ID"] == qid]
        if qrow.empty:
            continue
        qtext = str(qrow.iloc[0]["TEXT"])
        irow = inten_df[inten_df["ID"] == qid]
        intent_text = str(irow.iloc[0]["INTENCAO"]) if not irow.empty else ""

        print(f"\n=== Query #{idx} (ID: {qid}) ===")
        print(f"Texto: {qtext}")

        # Top 20 candidatos para a query
        cand_rows = candidatos_df[candidatos_df["QUERY_ID"] == qid].sort_values("RANK").head(20)
        if cand_rows.empty:
            print("(Sem candidatos)")
            continue

        # Monta resultados_busca para similaridade
        resultados_busca = []
        for drow in cand_rows.itertuples(index=False):
            doc_id = int(drow.DOC_ID_NUM)
            enun = docs_map.get(doc_id, "")
            resultados_busca.append({"id": str(doc_id), "conteudo": enun, "score": 1.0, "metodo": "Candidato"})

        conversa = qtext
        if args.modo == "pares":
            print("Gerando pares similares (top 3, min sim 0.8)...")
            try:
                pares_similares = buscador.calcular_similaridade_entre_pares(
                    resultados_busca=resultados_busca,
                    limite_similaridade=0.8,
                    top_k=3,
                )
            except Exception as e:
                print(f"✗ Falha ao calcular similaridade entre pares: {e}")
                pares_similares = []
            if pares_similares:
                for pidx, par in enumerate(pares_similares[:3], start=1):
                    try:
                        perguntas = gerar_perguntas_clarificadoras_para_pares(
                            pares_similares=[par],
                            conversa=conversa,
                            max_perguntas=1,
                        )
                    except Exception as e:
                        print(f"✗ Erro ao gerar perguntas (passo {pidx}): {e}")
                        continue
                    pergunta = (perguntas[0].get("pergunta") if perguntas else "")
                    print(f"\n[Passo {pidx}] Pergunta clarificadora: {pergunta}")
                    try:
                        resposta = responder_pergunta_clarificadora(intent_text, pergunta)
                    except Exception as e:
                        resposta = f"(Falha ao responder: {e})"
                    print(f"Resposta: {resposta}")
                    conversa = conversa + "\n\nPergunta clarificadora: " + pergunta + "\nResposta: " + resposta
            else:
                print("(Nenhum par com similaridade suficiente)")
        else:
            try:
                perguntas = gerar_perguntas_sem_pares(pergunta=qtext, max_perguntas=3)
            except Exception as e:
                print(f"✗ Erro ao gerar perguntas sem pares: {e}")
                perguntas = []
            for pidx, item in enumerate(perguntas[:3], start=1):
                pergunta = item.get("pergunta") or ""
                print(f"\n[Passo {pidx}] Pergunta clarificadora: {pergunta}")
                try:
                    resposta = responder_pergunta_clarificadora(intent_text, pergunta)
                except Exception as e:
                    resposta = f"(Falha ao responder: {e})"
                print(f"Resposta: {resposta}")
                conversa = conversa + "\n\nPergunta clarificadora: " + pergunta + "\nResposta: " + resposta

        # Rerank dos 20 candidatos usando a conversa completa
        nodes = []
        for drow in cand_rows.itertuples(index=False):
            doc_id = int(drow.DOC_ID_NUM)
            enun = docs_map.get(doc_id, "")
            node = TextNode(
                text=enun,
                id_=str(doc_id),
                metadata={"id": doc_id, "enunciado": enun, "titulo": enun[:100]},
            )
            nodes.append(node)

        try:
            reranked = rerank_nodes(buscador.reranker_model, conversa, nodes, top_n=20)
        except Exception as e:
            print(f"✗ Erro no rerank: {e}")
            reranked = []

        for rank, item in enumerate(reranked, start=1):
            did = getattr(getattr(item, 'node', item), 'metadata', {}).get('id')
            all_rows.append({
                "QUERY_ID": qid,
                "DOC_ID": did,
                "RERANK_SCORE": getattr(item, 'score', None),
                "RANK": rank,
            })

    # Salva CSV consolidado
    os.makedirs(os.path.join(BASE_DIR, "dados"), exist_ok=True)
    out_df = pd.DataFrame(all_rows)
    if args.modo == "pares":
        out_df.to_csv(OUT_CSV_PAIRS, index=False, encoding="utf-8")
        print(f"\nArquivo salvo: {OUT_CSV_PAIRS} (linhas: {len(out_df)})")
    else:
        out_df.to_csv(OUT_CSV_NO_PAIRS, index=False, encoding="utf-8")
        print(f"\nArquivo salvo: {OUT_CSV_NO_PAIRS} (linhas: {len(out_df)})")

    # Calcula métricas top-10
    if out_df.empty:
        print("\n(⚠ Sem linhas para métricas; rerank retornou vazio)")
    else:
        qrels_df = load_qrels_df(os.path.join(DATA_DIR, "qrel.csv"))
        pd_metricas = metricas(
            resultado_pesquisa=out_df,
            qrels=qrels_df,
            col_resultado_query_key="QUERY_ID",
            col_resultado_doc_key="DOC_ID",
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
        if args.modo == "pares":
            pd_metricas_out.to_csv(OUT_METRICAS_PAIRS, index=False, encoding="utf-8")
            print(pd_metricas_out.to_string(index=False))
            print(f"Metricas salvas em: {OUT_METRICAS_PAIRS}")
        else:
            pd_metricas_out.to_csv(OUT_METRICAS_NO_PAIRS, index=False, encoding="utf-8")
            print(pd_metricas_out.to_string(index=False))
            print(f"Metricas salvas em: {OUT_METRICAS_NO_PAIRS}")


if __name__ == "__main__":
    main()
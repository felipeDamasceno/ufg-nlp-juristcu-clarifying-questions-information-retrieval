import os
import sys
import random
import pandas as pd
from dotenv import load_dotenv
import time
# Carregar .env para chave de API
load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.buscador_hibrido import BuscadorHibridoLlamaIndex
from src.clarifying_questions import gerar_perguntas_clarificadoras_para_pares
from src.resposta_clarificadora import responder_pergunta_clarificadora
from src.utils.dados import load_qrels_df, load_docs_enunciado_map_clean


OUTPUTS_CSV = os.path.join(BASE_DIR, "dados", "query_intencao.csv")
DATA_DIR = os.path.join(BASE_DIR, "dados", "juris_tcu")
QREL_CSV = os.path.join(DATA_DIR, "qrel.csv")
DOC_CSV = os.path.join(DATA_DIR, "doc.csv")


def _carregar_intencoes(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    # Normalizar valores vazios
    df["INTENCAO"] = df["INTENCAO"].astype(str).fillna("").str.strip()
    df = df[(df["INTENCAO"] != "") & (df["INTENCAO"].str.lower() != "nan")]
    return df


def main():
    print("=== Teste Realista: Perguntas Clarificadoras + Resposta Clarificadora ===")

    # Verificações de arquivos
    if not os.path.exists(OUTPUTS_CSV):
        print(f"✗ Saída não encontrada: {OUTPUTS_CSV}")
        return
    if not (os.path.exists(QREL_CSV) and os.path.exists(DOC_CSV)):
        print(f"✗ Arquivos necessários não encontrados: {QREL_CSV}, {DOC_CSV}")
        return

    # Carregar intenções do dataset gerado
    inten_df = _carregar_intencoes(OUTPUTS_CSV)
    if inten_df.empty:
        print("✗ Nenhuma intenção válida encontrada em dados/query_intencao.csv")
        return

    # Carregar qrels e mapa de enunciados limpos
    qrels_df = load_qrels_df(QREL_CSV)
    docs_map = load_docs_enunciado_map_clean(DOC_CSV)

    # Inicializar buscador para cálculo de similaridade entre documentos ideais
    try:
        buscador = BuscadorHibridoLlamaIndex()
        if not buscador.embeddings_model:
            print("✗ Teste abortado: Modelo de embedding não foi carregado.")
            return
    except Exception as e:
        print(f"✗ Falha ao inicializar o BuscadorHibridoLlamaIndex: {e}")
        return

    # Selecionar 5 queries aleatórias com INTENCAO
    amostra = inten_df.sample(n=min(5, len(inten_df)), random_state=random.randint(0, 10_000))

    # Usar TODOS os documentos do qrels (sem filtrar por SCORE)
    qrels_sorted = qrels_df.sort_values("RANK")

    for idx, row in enumerate(amostra.itertuples(index=False), start=1):
        time.sleep(10)
        qid = int(row.ID)
        qtext = str(row.TEXT)
        intent_text = str(row.INTENCAO)

        print(f"\n=== Query #{idx} (ID: {qid}) ===")
        print(f"Texto: {qtext}")
        print(f"Intenção/Necessidade de informação: {intent_text}\n")

        # Documentos da busca (todos os qrels para a query)
        docs_rows = qrels_sorted[qrels_sorted["QUERY_ID"] == qid]
        if docs_rows.empty:
            print("(Sem documentos ideais de score 3)")
            continue

        resultados_busca = []
        for drow in docs_rows.itertuples(index=False):
            doc_id = int(drow.DOC_ID)
            enun = docs_map.get(doc_id) or ""
            resultados_busca.append({"id": str(doc_id), "conteudo": enun, "score": 1.0, "metodo": "Qrels"})

        # Calcular similaridade e gerar perguntas (uma por par)
        print("\nGerando perguntas clarificadoras para pares de docs ideais...")
        try:
            pares_similares = buscador.calcular_similaridade_entre_pares(
                resultados_busca=resultados_busca,
                limite_similaridade=0.8,
                top_k=3,
            )
        except Exception as e:
            print(f"✗ Falha ao calcular similaridade entre pares: {e}")
            continue

        if not pares_similares:
            print("(Nenhum par com similaridade suficiente)")
            continue

        conversa_ctx = intent_text
        

        try:
            perguntas = gerar_perguntas_clarificadoras_para_pares(
                pares_similares=pares_similares,
                conversa=conversa_ctx,
                max_perguntas=min(3, len(pares_similares)),
            )
        except RuntimeError as e:
            print(f"✗ Erro ao gerar perguntas via Gemini: {e}")
            continue

        if not perguntas:
            print("(Nenhuma pergunta gerada)")
            continue

        # Responder perguntas com base EXCLUSIVA na intenção
        print("\nRespondendo perguntas apenas com a INTENCAO (BACKGROUND)...")
        for pidx, item in enumerate(perguntas, start=1):
            pergunta = item.get("pergunta") or ""
            #resposta_completa = item.get("resposta_completa")
            par_index = item.get("par_index", pidx)
            # Mostrar apenas o par usado para gerar a pergunta
            try:
                par = pares_similares[par_index]
            except Exception:
                par = {}
            doc1 = par.get("documento_1", {})
            doc2 = par.get("documento_2", {})
            d1_text = doc1.get("conteudo", doc1.get("enunciado", ""))
            d2_text = doc2.get("conteudo", doc2.get("enunciado", ""))

            print(f"\n[Par #{par_index + 1}] Documentos usados para a pergunta:")
            print(f"  Documento 1 (ID: {doc1.get('id')}): {d1_text}")
            print(f"  Documento 2 (ID: {doc2.get('id')}): {d2_text}")
            print(f"  Pergunta gerada: {pergunta}")
            #print(f"  Resposta completa (Gemini - geração da pergunta): {resposta_completa}")
            try:
                resposta = responder_pergunta_clarificadora(intent_text, pergunta)
                print(f"  Resposta clarificadora (apenas BACKGROUND): {resposta}")
            except Exception as e:
                print(f"  ✗ Falha ao responder clarificadora: {e}")


if __name__ == "__main__":
    main()
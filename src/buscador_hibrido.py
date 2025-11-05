"""
Sistema de Busca Híbrida usando LlamaIndex
Combina BM25 e embeddings do Gemini para busca em documentos jurídicos do TCU
"""

import os
import time
from typing import List, Dict, Any, Optional

# Imports do LlamaIndex
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.schema import TextNode
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import QueryBundle

# Imports locais
from src.documento import DocumentoJuris
from src.utils.preprocessamento import PreprocessadorTexto
from src.bm25 import BM25RetrieverCustom


class BuscadorHibridoLlamaIndex:
    """
    Sistema de busca híbrida usando LlamaIndex
    Combina BM25 e embeddings do Gemini
    """
    
    def __init__(self):
        """
        Inicializa o buscador híbrido com embedding português jurídico
        """
        self.preprocessador = PreprocessadorTexto()
        self.documentos = []
        self.bm25_retriever = None
        self.vector_retriever = None
        self.embeddings_model = None
        
        # Retrievers do LlamaIndex
        self.llama_bm25_retriever = None
        self.llama_vector_retriever = None
        self.hybrid_retriever = None
        
        # Configurar modelo de embeddings português jurídico
        try:
            self.embeddings_model = HuggingFaceEmbedding(
                model_name="stjiris/bert-large-portuguese-cased-legal-mlm-sts-v1.0",
                trust_remote_code=True
            )
            print("✓ Modelo de embeddings português jurídico configurado com sucesso")
            print("  - Modelo: stjiris/bert-large-portuguese-cased-legal-mlm-sts-v1.0")
            print("  - Especializado em domínio jurídico português")
        except Exception as e:
            print(f"⚠ Erro ao configurar embeddings: {e}")
            self.embeddings_model = None
    
    def carregar_documentos(self, documentos: List[DocumentoJuris]):
        """
        Carrega e processa documentos, criando nós compartilhados para BM25 e Embeddings.
        
        Args:
            documentos: Lista de documentos jurídicos
        """
        self.documentos = documentos
        print(f"✓ {len(documentos)} documentos carregados para processamento")

        # Configurar tokenizer para truncamento
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("stjiris/bert-large-portuguese-cased-legal-mlm-sts-v1.0")

        # 1. Criar Nós (Nodes) compartilhados a partir do ENUNCIADO
        nodes = []
        textos_truncados = 0
        for doc in documentos:
            texto_limpo = self.preprocessador.remove_html(doc.enunciado)
            
            # Truncar para o limite do modelo de embedding
            tokens = tokenizer.encode(texto_limpo, add_special_tokens=True)
            if len(tokens) > 1024:
                tokens_truncados = tokens[:1024]
                texto_processado = tokenizer.decode(tokens_truncados, skip_special_tokens=True)
                textos_truncados += 1
            else:
                texto_processado = texto_limpo

            node = TextNode(
                text=texto_processado,
                id_=str(doc.id),
                metadata={
                    "id": doc.id,
                    "enunciado": doc.enunciado,
                    "excerto": doc.excerto,
                    "titulo": doc.enunciado[:100] + "..." if len(doc.enunciado) > 100 else doc.enunciado
                }
            )
            nodes.append(node)
        
        print(f"✓ {len(nodes)} nós de texto compartilhados criados (a partir do 'enunciado')")
        if textos_truncados > 0:
            print(f"  - {textos_truncados} textos foram truncados para 1024 tokens.")

        # 2. Configurar BM25 usando os nós compartilhados
        self._configurar_bm25(nodes)
        
        # 3. Configurar Embeddings usando os nós compartilhados
        if self.embeddings_model:
            self._configurar_embeddings(nodes)
        
        # 4. Configurar o retriever híbrido
        self._configurar_retrievers_llama()
    
    def _configurar_bm25(self, nodes: List[TextNode]):
        """Configura o retriever BM25 a partir de nós pré-criados."""
        try:
            self.bm25_retriever = BM25RetrieverCustom(
                nodes=nodes,
                tokenizer=self.preprocessador.tokenizador_pt_remove_html,
                similarity_top_k=10,
                k1=1.2,
                b=0.75
            )
            print("✓ BM25 retriever configurado com sucesso")
            print("  - Parâmetros: k1=1.2, b=0.75")
            print("  - Fonte: Nós compartilhados (apenas 'enunciado')")
            
        except Exception as e:
            print(f"✗ Erro ao configurar BM25: {e}")
            self.bm25_retriever = None
    
    def _configurar_retrievers_llama(self):
        """Configura os retrievers do LlamaIndex para busca híbrida"""
        try:
            # Configurar Vector Retriever se embeddings estão disponíveis
            if self.vector_retriever:
                self.llama_vector_retriever = self.vector_retriever
                
                # Configurar Hybrid Retriever usando QueryFusionRetriever
                # Combina nosso BM25 customizado (com k1 e b) com vector retriever
                self.hybrid_retriever = QueryFusionRetriever(
                    retrievers=[self.bm25_retriever, self.llama_vector_retriever],
                    similarity_top_k=10,
                    num_queries=1,  # Usar apenas a query original
                    mode="reciprocal_rerank",  # Usar Reciprocal Rank Fusion
                    use_async=False
                )
                
                print("Hybrid retriever configurado com QueryFusionRetriever (RRF)")
                print(f"Usando BM25 customizado com k1={self.bm25_retriever.bm25.k1}, b={self.bm25_retriever.bm25.b}")
            else:
                print("Vector retriever não está configurado - usando apenas BM25")
                
        except Exception as e:
            print(f"Erro ao configurar retrievers do LlamaIndex: {e}")
        
    def _configurar_embeddings(self, nodes: List[TextNode]):
        """Configura o retriever de embeddings a partir de nós pré-criados."""
        try:
            # Configurar Settings do LlamaIndex para o modelo de embedding
            Settings.embed_model = self.embeddings_model
            Settings.chunk_size = 1024
            Settings.chunk_overlap = 0
            
            # Criar índice vetorial a partir dos nós existentes
            vector_store = SimpleVectorStore()
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            self.vector_index = VectorStoreIndex(
                nodes=nodes,  # Usar os nós compartilhados
                storage_context=storage_context
            )
            
            self.vector_retriever = self.vector_index.as_retriever(
                similarity_top_k=10
            )
            
            print("✓ Vector retriever configurado com sucesso")
            print("  - Modelo: Embedding português jurídico")
            print("  - Fonte: Nós compartilhados (apenas 'enunciado')")
            
        except Exception as e:
            print(f"✗ Erro ao configurar embeddings: {e}")
            self.vector_retriever = None
    
    def buscar_bm25(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Realiza busca usando BM25
        
        Args:
            query: Consulta de busca
            top_k: Número de resultados a retornar
            
        Returns:
            Lista de resultados com scores
        """
        if not self.bm25_retriever:
            print("⚠ BM25 retriever não está configurado")
            return []
        
        try:
            # Realizar busca
            nodes = self.bm25_retriever.retrieve(query)
            
            # Converter para formato padrão
            resultados = []
            for i, node in enumerate(nodes[:top_k]):
                resultado = {
                    "id": node.metadata.get("id", f"doc_{i}"),
                    "enunciado": node.metadata.get("enunciado", ""),
                    "excerto": node.metadata.get("excerto", ""),
                    "score": getattr(node, 'score', 0.0),  # Score aproximado
                    "texto_completo": node.text,
                    "metodo": "BM25"
                }
                resultados.append(resultado)
            
            return resultados
            
        except Exception as e:
            print(f"✗ Erro na busca BM25: {e}")
            return []
    
    def buscar_embeddings(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Realiza busca usando embeddings
        
        Args:
            query: Consulta de busca
            top_k: Número de resultados a retornar
            
        Returns:
            Lista de resultados com scores
        """
        if not self.vector_retriever:
            print("⚠ Vector retriever não está configurado")
            return []
        
        try:
            # Realizar busca
            nodes = self.vector_retriever.retrieve(query)
            
            # Converter para formato padrão
            resultados = []
            for i, node in enumerate(nodes[:top_k]):
                resultado = {
                    "id": node.metadata.get("id", f"doc_{i}"),
                    "enunciado": node.metadata.get("enunciado", ""),
                    "excerto": node.metadata.get("excerto", ""),
                    "score": getattr(node, 'score', 0.0),  # Score aproximado
                    "texto_completo": node.text,
                    "metodo": "Embeddings"
                }
                resultados.append(resultado)
            
            return resultados
            
        except Exception as e:
            print(f"✗ Erro na busca por embeddings: {e}")
            return []
    
    def buscar_hibrido(self, consulta: str, top_k: int = 10) -> List[Dict]:
        """
        Realiza busca híbrida usando o QueryFusionRetriever (RRF).
        O retriever já foi configurado para usar os nós compartilhados, eliminando duplicatas.

        Args:
            consulta: Consulta de busca.
            top_k: Número de resultados a retornar.

        Returns:
            Lista de resultados únicos ordenados pelo score do RRF.
        """
        print(f"\n=== BUSCA HÍBRIDA com QueryFusionRetriever ===")
        print(f"Consulta: {consulta}")

        if not self.hybrid_retriever:
            print("⚠ Hybrid retriever (QueryFusionRetriever) não está configurado.")
            # Fallback para RRF manual se o retriever híbrido falhar na configuração
            return self._buscar_hibrido_manual_rrf(consulta, top_k)

        try:
            # Usar o QueryFusionRetriever que já aplica RRF e lida com duplicatas
            retrieved_nodes = self.hybrid_retriever.retrieve(consulta)

            print(f"QueryFusionRetriever retornou {len(retrieved_nodes)} nós únicos.")

            # Formatar os resultados para o padrão esperado
            resultados_formatados = []
            for node in retrieved_nodes[:top_k]:
                resultado = {
                    "id": node.metadata.get("id"),
                    "titulo": node.metadata.get("titulo", ""),
                    "conteudo": node.text,
                    "score": node.score,
                    "metodo": "Híbrido (QueryFusionRetriever)",
                    "metadata": {
                        "enunciado": node.metadata.get("enunciado", ""),
                        "excerto": node.metadata.get("excerto", ""),
                    }
                }
                resultados_formatados.append(resultado)
            
            print(f"Retornando os {len(resultados_formatados)} melhores resultados.")
            return resultados_formatados

        except Exception as e:
            print(f"✗ Erro na busca híbrida com QueryFusionRetriever: {e}")
            print("    Fallback para busca híbrida com RRF manual.")
            return self._buscar_hibrido_manual_rrf(consulta, top_k)

    def _buscar_hibrido_manual_rrf(self, consulta: str, top_k: int = 10) -> List[Dict]:
        """
        Fallback: Realiza busca híbrida com RRF manual. Usado apenas se o QueryFusionRetriever falhar.
        """
        print("--- Fallback para RRF Manual ---")
        bm25_results = self.buscar_bm25(consulta, top_k=top_k)
        vector_results = self.buscar_embeddings(consulta, top_k=top_k)
        
        k = 60
        rrf_scores = {}

        # Processar resultados BM25
        for rank, result in enumerate(bm25_results, 1):
            doc_id = result['id']
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {'score': 0.0, 'details': result}
            rrf_scores[doc_id]['score'] += 1.0 / (rank + k)

        # Processar resultados Vector
        for rank, result in enumerate(vector_results, 1):
            doc_id = result['id']
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {'score': 0.0, 'details': result}
            rrf_scores[doc_id]['score'] += 1.0 / (rank + k)

        sorted_results = sorted(rrf_scores.items(), key=lambda item: item[1]['score'], reverse=True)
        
        # Formatar para o padrão de saída
        resultados_formatados = []
        for doc_id, data in sorted_results[:top_k]:
            details = data['details']
            resultado = {
                "id": doc_id,
                "titulo": details['enunciado'][:100] + "...",
                "conteudo": details['texto_completo'],
                "score": data['score'],
                "metodo": "Híbrido (RRF Manual Fallback)",
                "metadata": {
                    "enunciado": details['enunciado'],
                    "excerto": details['excerto']
                }
            }
            resultados_formatados.append(resultado)
            
        return resultados_formatados

    def avaliar_performance(self, query: str) -> Dict[str, Any]:
        """
        Avalia a performance dos diferentes métodos de busca
        
        Args:
            query: Consulta de busca
            
        Returns:
            Dicionário com métricas de performance
        """
        metricas = {}
        
        # Testar BM25
        if self.bm25_retriever:
            inicio = time.time()
            resultados_bm25 = self.buscar_bm25(query)
            tempo_bm25 = time.time() - inicio
            metricas["bm25"] = {
                "tempo": tempo_bm25,
                "resultados": len(resultados_bm25),
                "disponivel": True
            }
        else:
            metricas["bm25"] = {"disponivel": False}
        
        # Testar embeddings
        if self.vector_retriever:
            inicio = time.time()
            resultados_embeddings = self.buscar_embeddings(query)
            tempo_embeddings = time.time() - inicio
            metricas["embeddings"] = {
                "tempo": tempo_embeddings,
                "resultados": len(resultados_embeddings),
                "disponivel": True
            }
        else:
            metricas["embeddings"] = {"disponivel": False}
        
        # Testar híbrido
        if self.bm25_retriever or self.vector_retriever:
            inicio = time.time()
            resultados_hibrido = self.buscar_hibrido(query)
            tempo_hibrido = time.time() - inicio
            metricas["hibrido"] = {
                "tempo": tempo_hibrido,
                "resultados": len(resultados_hibrido),
                "disponivel": True
            }
        else:
            metricas["hibrido"] = {"disponivel": False}
        
        return metricas
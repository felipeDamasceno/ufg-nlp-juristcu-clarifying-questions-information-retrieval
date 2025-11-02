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
        Carrega documentos no sistema
        
        Args:
            documentos: Lista de documentos jurídicos
        """
        self.documentos = documentos
        print(f"✓ {len(documentos)} documentos carregados")
        
        # Criar documentos do LlamaIndex para BM25 (enunciado + excerto)
        bm25_docs = []
        for doc in documentos:
            # Para BM25: combinar enunciado e excerto
            texto_completo = f"{doc.enunciado} {doc.excerto}"
            
            # Criar documento do LlamaIndex
            llama_doc = Document(
                text=texto_completo,
                doc_id=doc.id,
                metadata={
                    "id": doc.id,
                    "enunciado": doc.enunciado,
                    "excerto": doc.excerto
                }
            )
            bm25_docs.append(llama_doc)
        
        # Configurar BM25 com parâmetros específicos
        self._configurar_bm25(bm25_docs)
        
        # Configurar embeddings se disponível
        if self.embeddings_model:
            self._configurar_embeddings(documentos)
        
        # Configurar retrievers do LlamaIndex
        self._configurar_retrievers_llama(bm25_docs)
    
    def _configurar_bm25(self, documentos: List[Document]):
        """Configura o retriever BM25 com parâmetros específicos k1=1.2, b=0.75"""
        try:
            # Converter documentos em nodes (um documento = um node)
            # Não usar splitter para manter integridade do documento e evitar duplicação
            
            nodes = []
            for doc in documentos:
                node = TextNode(
                    text=doc.text,
                    metadata=doc.metadata,
                    id_=doc.doc_id
                )
                nodes.append(node)
            
            # Criar BM25 retriever customizado com parâmetros específicos
            self.bm25_retriever = BM25RetrieverCustom(
                nodes=nodes,
                tokenizer=self.preprocessador.tokenizador_pt_remove_html,
                similarity_top_k=10,
                k1=1.2,  # Parâmetro específico solicitado
                b=0.75   # Parâmetro específico solicitado
            )
            
            print("✓ BM25 retriever configurado com sucesso")
            print("  - Parâmetros: k1=1.2, b=0.75 (customizados)")
            print("  - Preprocessamento: tokenizador_pt_remove_html")
            print("  - Campos: enunciado + excerto")
            print(f"  - Nodes criados: {len(nodes)} (1 documento = 1 node)")
            
        except Exception as e:
            print(f"✗ Erro ao configurar BM25: {e}")
            self.bm25_retriever = None
    
    def _configurar_retrievers_llama(self, documentos: List[Document]):
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
        
    def _configurar_embeddings(self, documentos: List[DocumentoJuris]):
        """Configura o retriever de embeddings"""
        try:
            # Criar documentos do LlamaIndex para embeddings (apenas enunciado)
            embedding_docs = []
            for doc in documentos:
                # Para embeddings: apenas enunciado sem HTML
                texto_limpo = self.preprocessador.remove_html(doc.enunciado)
                
                # Metadados simplificados para evitar erro de tamanho
                metadata_simples = {
                    "id": doc.id,
                    "titulo": doc.enunciado[:100] + "..." if len(doc.enunciado) > 100 else doc.enunciado
                }
                
                llama_doc = Document(
                    text=texto_limpo,
                    doc_id=doc.id,
                    metadata=metadata_simples
                )
                embedding_docs.append(llama_doc)
            
            # Configurar Settings do LlamaIndex com chunk_size maior
            Settings.embed_model = self.embeddings_model
            Settings.chunk_size = 2048  # Aumentar chunk_size para evitar erro de metadata
            Settings.chunk_overlap = 200
            
            # Criar índice vetorial
            vector_store = SimpleVectorStore()
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            self.vector_index = VectorStoreIndex.from_documents(
                embedding_docs,
                storage_context=storage_context
            )
            
            self.vector_retriever = self.vector_index.as_retriever(
                similarity_top_k=10
            )
            
            print("✓ Vector retriever configurado com sucesso")
            print("  - Modelo: Embedding português jurídico")
            print("  - Preprocessamento: remoção de HTML")
            print("  - Campos: apenas enunciado")
            print(f"  - Chunk size: {Settings.chunk_size}")
            
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
    
    def buscar_hibrido(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Realiza busca híbrida usando QueryFusionRetriever do LlamaIndex
        
        Args:
            query: Consulta de busca
            top_k: Número de resultados a retornar
            
        Returns:
            Lista de resultados combinados com scores usando Reciprocal Rank Fusion
        """
        if not self.hybrid_retriever:
            print("Hybrid retriever não configurado. Usando busca BM25 apenas.")
            return self.buscar_bm25(query, top_k)
        
        try:
            from llama_index.core.schema import QueryBundle
            
            # Criar query bundle
            query_bundle = QueryBundle(query_str=query)
            
            # Configurar top_k no retriever
            self.hybrid_retriever.similarity_top_k = top_k
            
            # Executar busca híbrida
            nodes_with_scores = self.hybrid_retriever.retrieve(query_bundle)
            
            # Converter resultados para formato esperado
            resultados = []
            for node_with_score in nodes_with_scores:
                node = node_with_score.node
                score = node_with_score.score
                
                # Extrair metadados
                metadata = node.metadata or {}
                
                resultado = {
                    "id": metadata.get("id", ""),
                    "titulo": metadata.get("titulo", ""),
                    "conteudo": node.text,
                    "score": float(score),
                    "metodo": "Híbrido (QueryFusionRetriever - RRF)",
                    "metadata": metadata
                }
                
                resultados.append(resultado)
            
            print(f"Busca híbrida executada: {len(resultados)} resultados encontrados")
            return resultados
            
        except Exception as e:
            print(f"Erro na busca híbrida: {e}")
            print("Fallback para busca BM25")
            return self.buscar_bm25(query, top_k)
    
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
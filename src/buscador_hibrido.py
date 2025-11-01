"""
Sistema de Busca Híbrida usando LlamaIndex
Combina BM25 e embeddings do Gemini para busca em documentos jurídicos do TCU
"""

import os
import time
from typing import List, Dict, Any, Optional

# Imports do LlamaIndex
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.schema import TextNode

# Imports locais
from src.documento import DocumentoJuris
from src.utils.preprocessamento import PreprocessadorTexto
from src.bm25 import BM25RetrieverCustom


class BuscadorHibridoLlamaIndex:
    """
    Sistema de busca híbrida usando LlamaIndex
    Combina BM25 e embeddings do Gemini
    """
    
    def __init__(self, google_api_key: Optional[str] = None):
        """
        Inicializa o buscador híbrido
        
        Args:
            google_api_key: Chave da API do Google (opcional, pode usar variável de ambiente)
        """
        self.preprocessador = PreprocessadorTexto()
        self.documentos = []
        self.bm25_retriever = None
        self.vector_retriever = None
        self.embeddings_model = None
        
        # Configurar API do Google
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key
        
        # Verificar se a chave está disponível
        if os.getenv("GOOGLE_API_KEY"):
            try:
                self.embeddings_model = GoogleGenAIEmbedding(
                    model_name="models/embedding-001"
                )
                print("✓ Modelo de embeddings Gemini configurado com sucesso")
            except Exception as e:
                print(f"⚠ Erro ao configurar embeddings Gemini: {e}")
                self.embeddings_model = None
        else:
            print("⚠ GOOGLE_API_KEY não encontrada. Embeddings não estarão disponíveis.")
    
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
    
    def _configurar_embeddings(self, documentos: List[DocumentoJuris]):
        """Configura o retriever de embeddings"""
        try:
            # Criar documentos do LlamaIndex para embeddings (apenas enunciado)
            embedding_docs = []
            for doc in documentos:
                # Para embeddings: apenas enunciado sem HTML
                texto_limpo = self.preprocessador.remove_html(doc.enunciado)
                
                llama_doc = Document(
                    text=texto_limpo,
                    doc_id=doc.id,
                    metadata={
                        "id": doc.id,
                        "enunciado": doc.enunciado,
                        "excerto": doc.excerto
                    }
                )
                embedding_docs.append(llama_doc)
            
            # Configurar Settings do LlamaIndex
            Settings.embed_model = self.embeddings_model
            
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
            print("  - Modelo: Gemini embedding-001")
            print("  - Preprocessamento: remoção de HTML")
            print("  - Campos: apenas enunciado")
            
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
    
    def buscar_hibrido(self, query: str, top_k: int = 10, 
                      peso_bm25: float = 0.5, peso_embeddings: float = 0.5) -> List[Dict[str, Any]]:
        """
        Realiza busca híbrida combinando BM25 e embeddings
        
        Args:
            query: Consulta de busca
            top_k: Número de resultados a retornar
            peso_bm25: Peso para os resultados do BM25 (0-1)
            peso_embeddings: Peso para os resultados dos embeddings (0-1)
            
        Returns:
            Lista de resultados combinados com scores
        """
        # Buscar com ambos os métodos
        resultados_bm25 = self.buscar_bm25(query, top_k * 2)  # Buscar mais para ter opções
        resultados_embeddings = self.buscar_embeddings(query, top_k * 2)
        
        if not resultados_bm25 and not resultados_embeddings:
            return []
        
        # Combinar resultados
        scores_combinados = {}
        
        # Processar resultados BM25
        for i, resultado in enumerate(resultados_bm25):
            doc_id = resultado["id"]
            # Normalizar score (rank-based)
            score_normalizado = 1.0 / (i + 1)
            scores_combinados[doc_id] = {
                "score_bm25": score_normalizado,
                "score_embeddings": 0.0,
                "documento": resultado
            }
        
        # Processar resultados embeddings
        for i, resultado in enumerate(resultados_embeddings):
            doc_id = resultado["id"]
            # Normalizar score (rank-based)
            score_normalizado = 1.0 / (i + 1)
            
            if doc_id in scores_combinados:
                scores_combinados[doc_id]["score_embeddings"] = score_normalizado
            else:
                scores_combinados[doc_id] = {
                    "score_bm25": 0.0,
                    "score_embeddings": score_normalizado,
                    "documento": resultado
                }
        
        # Calcular scores finais
        resultados_finais = []
        for doc_id, dados in scores_combinados.items():
            score_final = (peso_bm25 * dados["score_bm25"] + 
                          peso_embeddings * dados["score_embeddings"])
            
            resultado = dados["documento"].copy()
            resultado["score"] = score_final
            resultado["score_bm25"] = dados["score_bm25"]
            resultado["score_embeddings"] = dados["score_embeddings"]
            resultado["metodo"] = "Híbrido"
            
            resultados_finais.append(resultado)
        
        # Ordenar por score final e retornar top_k
        resultados_finais.sort(key=lambda x: x["score"], reverse=True)
        return resultados_finais[:top_k]
    
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
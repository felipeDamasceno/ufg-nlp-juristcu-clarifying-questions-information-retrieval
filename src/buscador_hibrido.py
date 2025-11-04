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
                doc_id=str(doc.id),  # Converter para string
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
        self._configurar_retrievers_llama()
    
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
        
    def _configurar_embeddings(self, documentos: List[DocumentoJuris]):
        """Configura o retriever de embeddings com truncamento para 1024 tokens"""
        try:
            # Configurar tokenizer customizado para o modelo BERT português
            from transformers import AutoTokenizer
            
            class CustomTokenizer:
                def __init__(self, model_name):
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                def encode(self, text):
                    return self.tokenizer.encode(text, add_special_tokens=True)
                
                def decode(self, tokens):
                    return self.tokenizer.decode(tokens, skip_special_tokens=True)
            
            # Configurar Settings do LlamaIndex para o modelo de embedding português (1024 tokens max)
            Settings.embed_model = self.embeddings_model
            
            # Criar instância do tokenizer para uso local
            tokenizer_local = CustomTokenizer("stjiris/bert-large-portuguese-cased-legal-mlm-sts-v1.0")
            
            Settings.chunk_size = 1024    # Limite real do modelo
            Settings.chunk_overlap = 0    # Sem overlap já que não queremos chunks
            Settings.context_window = 1024  # Janela de contexto do modelo
            
            # Criar documentos do LlamaIndex para embeddings (apenas enunciado)
            embedding_docs = []
            textos_truncados = 0
            
            for doc in documentos:
                # Para embeddings: apenas enunciado sem HTML
                texto_limpo = self.preprocessador.remove_html(doc.enunciado)
                
                # Contar tokens reais usando o tokenizer do modelo
                tokens = tokenizer_local.encode(texto_limpo)
                
                # Truncar se exceder o limite de tokens
                if len(tokens) > 1024:
                    # Truncar tokens e decodificar de volta para texto
                    tokens_truncados = tokens[:1024]
                    texto_truncado = tokenizer_local.decode(tokens_truncados)
                    textos_truncados += 1
                else:
                    texto_truncado = texto_limpo
                
                # Metadados simplificados
                metadata_simples = {
                    "id": doc.id,
                    "titulo": doc.enunciado[:100] + "..." if len(doc.enunciado) > 100 else doc.enunciado
                }
                
                llama_doc = Document(
                    text=texto_truncado,
                    doc_id=str(doc.id),  # Converter para string
                    metadata=metadata_simples
                )
                embedding_docs.append(llama_doc)
            
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
            print("  - Modelo: Embedding português jurídico (1024 tokens max)")
            print("  - Preprocessamento: remoção de HTML + truncamento")
            print("  - Campos: apenas enunciado")
            print(f"  - Textos truncados: {textos_truncados}/{len(documentos)}")
            print(f"  - Chunk size: {Settings.chunk_size} (alto para evitar chunking)")
            print("  - Estratégia: truncamento ao invés de chunking")
            
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
        Realiza busca híbrida usando Reciprocal Rank Fusion (RRF) corretamente implementado.
        
        RRF combina rankings de BM25 e Vector usando a fórmula:
        score = Σ 1/(rank + k) onde k=60 (constante padrão)
        
        Args:
            consulta: Consulta de busca
            top_k: Número de resultados a retornar
            
        Returns:
            Lista de resultados ordenados por score RRF
        """
        print(f"\n=== BUSCA HÍBRIDA RRF ===")
        print(f"Consulta: {consulta}")
        print(f"Top K solicitado: {top_k}")
        
        # Obter resultados de ambos os métodos
        bm25_results = self.buscar_bm25(consulta, top_k=top_k)
        vector_results = self.buscar_embeddings(consulta, top_k=top_k)
        
        print(f"\n--- Resultados BM25 ---")
        print(f"BM25 retornou {len(bm25_results)} resultados")
        for i, result in enumerate(bm25_results, 1):
            print(f"  {i}. ID: {result['id']} | Score: {result['score']:.4f}")
        
        print(f"\n--- Resultados Vector ---")
        print(f"Vector retornou {len(vector_results)} resultados")
        for i, result in enumerate(vector_results, 1):
            print(f"  {i}. ID: {result['id']} | Score: {result['score']:.4f}")
        
        # Implementar RRF corretamente
        k = 60  # Constante padrão do RRF
        rrf_scores = {}
        
        # Processar resultados BM25
        for rank, result in enumerate(bm25_results, 1):  # rank começa em 1
            doc_id = result['id']
            rrf_score = 1.0 / (rank + k)
            
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {
                    'id': doc_id,
                    'rrf_score': 0.0,
                    'texto': result['texto_completo'],
                    'enunciado': result['enunciado'],
                    'excerto': result['excerto'],
                    'bm25_rank': None,
                    'vector_rank': None,
                    'bm25_score': None,
                    'vector_score': None
                }
            
            rrf_scores[doc_id]['rrf_score'] += rrf_score
            rrf_scores[doc_id]['bm25_rank'] = rank
            rrf_scores[doc_id]['bm25_score'] = result['score']
        
        # Processar resultados Vector
        for rank, result in enumerate(vector_results, 1):  # rank começa em 1
            doc_id = result['id']
            rrf_score = 1.0 / (rank + k)
            
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {
                    'id': doc_id,
                    'rrf_score': 0.0,
                    'texto': result['texto_completo'],
                    'enunciado': result['enunciado'],
                    'excerto': result['excerto'],
                    'bm25_rank': None,
                    'vector_rank': None,
                    'bm25_score': None,
                    'vector_score': None
                }
            
            rrf_scores[doc_id]['rrf_score'] += rrf_score
            rrf_scores[doc_id]['vector_rank'] = rank
            rrf_scores[doc_id]['vector_score'] = result['score']
            
            # Se o texto do vector for mais completo, usar ele
            if len(result['texto_completo']) > len(rrf_scores[doc_id]['texto']):
                rrf_scores[doc_id]['texto'] = result['texto_completo']
                rrf_scores[doc_id]['enunciado'] = result['enunciado']
                rrf_scores[doc_id]['excerto'] = result['excerto']
        
        print(f"\n--- RRF Fusion (k={k}) ---")
        print(f"Documentos únicos encontrados: {len(rrf_scores)}")
        
        # Ordenar por score RRF (maior para menor)
        resultados_finais = sorted(rrf_scores.values(), key=lambda x: x['rrf_score'], reverse=True)
        
        # Limitar ao top_k solicitado
        resultados_finais = resultados_finais[:top_k]
        
        # Mostrar detalhes do RRF
        for i, result in enumerate(resultados_finais, 1):
            bm25_info = f"BM25 Rank: {result['bm25_rank']}" if result['bm25_rank'] else "BM25: N/A"
            vector_info = f"Vector Rank: {result['vector_rank']}" if result['vector_rank'] else "Vector: N/A"
            print(f"  {i}. ID: {result['id']} | RRF Score: {result['rrf_score']:.6f} | {bm25_info} | {vector_info}")
        
        # Converter para formato de saída padrão
        resultados_formatados = []
        for result in resultados_finais:
            resultado_formatado = {
                "id": result['id'],
                "titulo": result['enunciado'][:100] + "..." if len(result['enunciado']) > 100 else result['enunciado'],
                "conteudo": result['texto'],
                "score": result['rrf_score'],
                "metodo": "Híbrido (RRF)",
                "metadata": {
                    "enunciado": result['enunciado'],
                    "excerto": result['excerto'],
                    "bm25_rank": result['bm25_rank'],
                    "vector_rank": result['vector_rank'],
                    "bm25_score": result['bm25_score'],
                    "vector_score": result['vector_score']
                }
            }
            resultados_formatados.append(resultado_formatado)
        
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
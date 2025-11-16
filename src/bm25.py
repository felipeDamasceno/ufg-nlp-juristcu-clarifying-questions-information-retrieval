from typing import List, Optional, Any
from rank_bm25 import BM25Okapi
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.retrievers import BaseRetriever

class BM25RetrieverCustom(BaseRetriever):
    """BM25Retriever customizado que aceita parâmetros k1 e b"""
    
    def __init__(
        self,
        nodes: List,
        tokenizer: Optional[Any] = None,
        similarity_top_k: int = 10,
        k1: float = 1.2,
        b: float = 0.75,
        **kwargs
    ):
        self._nodes = nodes
        self._similarity_top_k = similarity_top_k
        self._tokenizer = tokenizer or self._default_tokenizer
        
        # Criar corpus tokenizado
        self._corpus = [self._tokenizer(node.get_content()) for node in self._nodes]
        
        # Inicializar BM25 com parâmetros customizados
        self.bm25 = BM25Okapi(self._corpus, k1=k1, b=b)
        
        super().__init__(**kwargs)
    
    def _default_tokenizer(self, text: str) -> List[str]:
        """Tokenizador padrão simples"""
        return text.lower().split()
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Recupera documentos usando BM25"""
        query = query_bundle.query_str
        tokenized_query = self._tokenizer(query)
        
        # Obter scores BM25
        scores = self.bm25.get_scores(tokenized_query)
        
        # Criar lista de (score, node) e ordenar
        scored_nodes = list(zip(scores, self._nodes))
        scored_nodes.sort(key=lambda x: x[0], reverse=True)
        
        # Retornar top-k resultados
        results = []
        for score, node in scored_nodes[:self._similarity_top_k]:
            results.append(NodeWithScore(node=node, score=score))

        return results

    def set_top_k(self, top_k: int):
        self._similarity_top_k = top_k
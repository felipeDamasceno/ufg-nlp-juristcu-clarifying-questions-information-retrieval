"""
Módulo para preprocessamento de texto específico para documentos jurídicos
"""

import re
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from unidecode import unidecode
import string

# Baixar recursos do NLTK se necessário
try:
    stopwords.words('portuguese')
except LookupError:
    nltk.download('stopwords')

try:
    word_tokenize("test", language='portuguese')
except LookupError:
    nltk.download('punkt')

try:
    RSLPStemmer()
except LookupError:
    nltk.download('rslp')

class PreprocessadorTexto:
    """Classe para preprocessamento de texto específico para documentos jurídicos"""
    
    def __init__(self):
        pass
    
    def remove_html(self, html: str) -> str:
        """Remove tags HTML do texto"""
        if not html:
            return ""
        return re.sub("<[^>]*>", "", html).strip()
    
    def tokenizador_pt(self, texto):
        """Tokenizador em português com stemização e remoção de stopwords"""
        if not texto or pd.isna(texto):
            return []
            
        # Remove acentuação e converte para minúsculo
        texto = unidecode(texto.lower())
        
        # Remove pontuação
        texto = ''.join([char if char not in string.punctuation else ' ' for char in texto])
        
        # Tokeniza o texto
        tokens = word_tokenize(texto, language='portuguese')
        
        # Remove stopwords e aplica stemização
        stemmer = RSLPStemmer()
        tokens_processados = [stemmer.stem(token) for token in tokens if token not in stopwords.words('portuguese')]
        
        return tokens_processados
    
    def tokenizador_pt_remove_html(self, texto):
        """Tokenizador que remove HTML antes de processar"""
        return self.tokenizador_pt(self.remove_html(texto))
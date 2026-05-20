"""
Módulo de Preprocesamiento de Texto
====================================
Funciones para limpieza, tokenización, y normalización de texto.
"""

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# Descargar recursos necesarios (ejecutar una sola vez)
for _resource, _pkg in [
    ('tokenizers/punkt_tab', 'punkt_tab'),
    ('tokenizers/punkt', 'punkt'),
    ('corpora/stopwords', 'stopwords'),
]:
    try:
        nltk.data.find(_resource)
    except LookupError:
        nltk.download(_pkg, quiet=True)


def limpiar_texto(texto):
    """
    Limpia el texto eliminando caracteres especiales y puntuación.
    
    Args:
        texto (str): Texto a limpiar
        
    Returns:
        str: Texto limpio en minúsculas
    """
    # Convertir a minúsculas
    texto = texto.lower()
    
    # Eliminar URLs
    texto = re.sub(r'http\S+|www\S+', '', texto)
    
    # Eliminar caracteres especiales
    texto = re.sub(r'[^a-záéíóúñ\s]', '', texto)
    
    # Eliminar espacios múltiples
    texto = re.sub(r'\s+', ' ', texto).strip()
    
    return texto


def tokenizar(texto):
    """
    Tokeniza el texto en palabras.
    
    Args:
        texto (str): Texto a tokenizar
        
    Returns:
        list: Lista de tokens (palabras)
    """
    return word_tokenize(texto, language='spanish')


def eliminar_stopwords(tokens, idioma='spanish'):
    """
    Elimina stopwords del texto.
    
    Args:
        tokens (list): Lista de tokens
        idioma (str): Idioma para stopwords (default: 'spanish')
        
    Returns:
        list: Tokens sin stopwords
    """
    stop_words = set(stopwords.words(idioma))
    return [token for token in tokens if token not in stop_words]


def aplicar_stemming(tokens, idioma='spanish'):
    """
    Aplica stemming a los tokens usando Snowball.
    
    Args:
        tokens (list): Lista de tokens
        idioma (str): Idioma para stemmer (default: 'spanish')
        
    Returns:
        list: Tokens con stemming aplicado
    """
    stemmer = SnowballStemmer(idioma)
    return [stemmer.stem(token) for token in tokens]


def procesar_documento(texto, eliminar_stop=True, aplicar_stem=True):
    """
    Pipeline completo de preprocesamiento de un documento.
    
    Args:
        texto (str): Texto a procesar
        eliminar_stop (bool): Si se deben eliminar stopwords
        aplicar_stem (bool): Si se debe aplicar stemming
        
    Returns:
        list: Tokens procesados
    """
    # Limpiar
    texto_limpio = limpiar_texto(texto)
    
    # Tokenizar
    tokens = tokenizar(texto_limpio)
    
    # Eliminar stopwords
    if eliminar_stop:
        tokens = eliminar_stopwords(tokens)
    
    # Aplicar stemming
    if aplicar_stem:
        tokens = aplicar_stemming(tokens)
    
    return tokens


def procesar_corpus(documentos):
    """
    Procesa una lista de documentos.
    
    Args:
        documentos (list): Lista de textos
        
    Returns:
        list: Lista de documentos procesados (cada uno es una lista de tokens)
    """
    corpus_procesado = [procesar_documento(doc) for doc in documentos]
    return corpus_procesado

"""
Módulo del Modelo LDA
====================
Funciones para entrenar y evaluar modelos de tópicos LDA.
"""

import gensim
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel


def crear_diccionario_y_corpus(documentos_tokens):
    """
    Crea diccionario y corpus gensim a partir de listas de tokens.

    Returns:
        tuple: (dictionary, corpus)
    """
    dictionary = Dictionary(documentos_tokens)
    # Filtrar tokens que aparecen en menos de 2 docs o en más del 90%
    dictionary.filter_extremes(no_below=2, no_above=0.9, keep_n=50000)
    corpus = [dictionary.doc2bow(doc) for doc in documentos_tokens]
    return dictionary, corpus


def entrenar_lda(corpus, dictionary, num_topics=5, passes=15):
    """
    Entrena un modelo LDA con los parámetros dados.

    Args:
        corpus: Corpus gensim (lista de bag-of-words)
        dictionary: Diccionario gensim
        num_topics (int): Número de tópicos K
        passes (int): Pasadas del algoritmo (más = mejor convergencia)

    Returns:
        LdaModel: Modelo entrenado
    """
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        alpha='auto',
        eta='auto',
        minimum_probability=0.0,
        random_state=42,
        per_word_topics=True,
    )
    return lda_model


def calcular_coherencia(lda_model, corpus, diccionario, textos_tokens):
    """
    Calcula la coherencia semántica c_v del modelo.

    Returns:
        float: Score de coherencia (mayor es mejor)
    """
    coherence_model = CoherenceModel(
        model=lda_model,
        corpus=corpus,
        dictionary=diccionario,
        texts=textos_tokens,
        coherence='c_v',
    )
    return coherence_model.get_coherence()


def optimizar_num_topics(corpus, dictionary, textos_tokens,
                         min_topics=2, max_topics=10, step=1, passes=10):
    """
    Busca el K óptimo probando distintos valores y midiendo coherencia.

    Returns:
        dict: {num_topics: coherencia}
    """
    resultados = {}
    for num_topics in range(min_topics, max_topics + 1, step):
        modelo = entrenar_lda(corpus, dictionary, num_topics=num_topics, passes=passes)
        coherencia = calcular_coherencia(modelo, corpus, dictionary, textos_tokens)
        resultados[num_topics] = coherencia
        print(f"  K={num_topics} → coherencia={coherencia:.4f}")
    return resultados


def obtener_topicos(lda_model, num_words=10):
    """
    Extrae las top palabras de cada tópico.

    Returns:
        dict: {Topico_N: string_palabras_pesos}
    """
    topicos = {}
    for idx, topic in lda_model.print_topics(num_topics=-1, num_words=num_words):
        topicos[f'Topico_{idx}'] = topic
    return topicos


def asignar_topicos(lda_model, corpus):
    """
    Asigna el tópico dominante a cada documento.

    Returns:
        list[tuple]: [(topic_id, probabilidad), ...] por documento
    """
    topicos_doc = []
    for doc_topics in lda_model.get_document_topics(corpus):
        if doc_topics:
            dominante = max(doc_topics, key=lambda x: x[1])
            topicos_doc.append(dominante)
        else:
            topicos_doc.append((None, 0.0))
    return topicos_doc


def parsear_topico_palabras(topic_str):
    """
    Convierte el string de gensim 'peso*"palabra" + ...' a dict {palabra: peso}.

    Returns:
        dict: {palabra: peso}
    """
    palabras = {}
    for item in topic_str.split('+'):
        parts = item.strip().split('*')
        if len(parts) == 2:
            try:
                peso = float(parts[0].strip())
                palabra = parts[1].strip().strip('"')
                palabras[palabra] = peso
            except ValueError:
                pass
    return palabras

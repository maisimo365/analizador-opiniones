"""
Analizador de Opiniones de Docentes
UMSS FCyT · Inteligencia Artificial · Grupo 15 · I/2026
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocesamiento import limpiar_texto, tokenizar, eliminar_stopwords, aplicar_stemming, procesar_documento
from src.modelo import crear_diccionario_y_corpus, entrenar_lda, calcular_coherencia, obtener_topicos, asignar_topicos, parsear_topico_palabras
from src.visualizacion import grafico_coherencia_vs_topicos

# ── Configuración ──────────────────────────────────────────
st.set_page_config(
    page_title="Analizador de Opiniones · UMSS",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Estilos CSS ────────────────────────────────────────────
st.markdown("""
<style>
  .step-box {
    background: #f0f4ff;
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
  }
  .explain-box {
    background: #fffbeb;
    border-left: 4px solid #f59e0b;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.92em;
  }
  .result-box {
    background: #f0fdf4;
    border-left: 4px solid #22c55e;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
  }
  .big-metric { font-size: 2em; font-weight: bold; color: #1e40af; }
  .tag {
    display: inline-block;
    background: #dbeafe;
    color: #1e3a8a;
    border-radius: 12px;
    padding: 2px 10px;
    margin: 2px;
    font-size: 0.85em;
  }
  .tag-stem {
    background: #dcfce7;
    color: #14532d;
  }
  .arrow { color: #94a3b8; font-size: 1.3em; margin: 0 6px; }
</style>
""", unsafe_allow_html=True)

# ── Estado de sesión ───────────────────────────────────────
for k, v in {
    "df": None, "corpus_tokens": None, "dictionary": None,
    "corpus_gensim": None, "lda_model": None, "topicos_doc": None,
    "coherencia": None, "num_topics": 5,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Datos automáticos al inicio ────────────────────────────
DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "encuesta_sintetica.csv"
if st.session_state.df is None and DATA_PATH.exists():
    st.session_state.df = pd.read_csv(DATA_PATH)

# ── Header ─────────────────────────────────────────────────
st.markdown("## 🎓 Analizador de Opiniones de Docentes")
st.markdown("**UMSS · Inteligencia Artificial · Grupo 15** — Análisis automático de encuestas con PLN y LDA")
st.divider()

# ── Barra de progreso del pipeline ─────────────────────────
pasos = ["1 · Datos", "2 · Preprocesamiento", "3 · Modelo LDA", "4 · Resultados"]
datos_ok   = st.session_state.df is not None
preproc_ok = st.session_state.corpus_tokens is not None
modelo_ok  = st.session_state.lda_model is not None

cols_step = st.columns(4)
for i, (col, paso) in enumerate(zip(cols_step, pasos)):
    done = [datos_ok, preproc_ok, modelo_ok, modelo_ok][i]
    color = "#22c55e" if done else "#94a3b8"
    col.markdown(f"<div style='text-align:center;color:{color};font-weight:bold'>{paso}<br>{'✅' if done else '⬜'}</div>", unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════
# NAVEGACIÓN
# ══════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📁 Paso 1 · Datos",
    "🔧 Paso 2 · Preprocesamiento",
    "🤖 Paso 3 · Modelo LDA",
    "📊 Paso 4 · Resultados",
])

# ══════════════════════════════════════════════════════════
# TAB 1 — DATOS
# ══════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 📁 Dataset de encuestas estudiantiles")

    st.markdown("""<div class="explain-box">
    <b>¿Qué son estos datos?</b><br>
    Cada fila representa la opinión de un estudiante sobre un docente en una dimensión específica
    (Metodología, Evaluación, Comunicación, Puntualidad o Material).
    Cada opinión tiene un <b>comentario libre</b> en español y una <b>calificación Likert</b> del 1 al 5.
    </div>""", unsafe_allow_html=True)

    col_btn, col_up = st.columns([1, 1])
    with col_btn:
        if st.button("📂 Usar dataset de ejemplo (80 comentarios)", use_container_width=True, type="primary"):
            st.session_state.df = pd.read_csv(DATA_PATH)
            st.session_state.corpus_tokens = None
            st.session_state.lda_model = None
    with col_up:
        f = st.file_uploader("O sube tu propio CSV", type="csv", label_visibility="collapsed")
        if f:
            st.session_state.df = pd.read_csv(f)
            st.session_state.corpus_tokens = None
            st.session_state.lda_model = None

    if st.session_state.df is not None:
        df = st.session_state.df

        # Métricas rápidas
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Comentarios", len(df))
        m2.metric("Docentes", df["docente"].nunique() if "docente" in df.columns else "—")
        m3.metric("Dimensiones", df["dimension"].nunique() if "dimension" in df.columns else "—")
        m4.metric("Likert promedio", f"{df['calificacion_likert'].mean():.1f} / 5" if "calificacion_likert" in df.columns else "—")

        # Vista previa
        st.markdown("**Vista previa del dataset:**")
        st.dataframe(df.head(8), use_container_width=True, hide_index=True)

        # Gráficos
        if "calificacion_likert" in df.columns and "dimension" in df.columns:
            g1, g2 = st.columns(2)

            with g1:
                st.markdown("**Distribución Likert**")
                labels = {1:"Muy malo",2:"Malo",3:"Regular",4:"Bueno",5:"Muy bueno"}
                colors = ["#dc2626","#ea580c","#ca8a04","#16a34a","#1d4ed8"]
                counts = df["calificacion_likert"].value_counts().sort_index()
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.bar([labels[i] for i in counts.index], counts.values,
                       color=[colors[i-1] for i in counts.index])
                ax.set_ylabel("Respuestas"); ax.set_title("Escala Likert", fontweight="bold")
                plt.xticks(rotation=20, fontsize=8); plt.tight_layout()
                st.pyplot(fig); plt.close()

            with g2:
                st.markdown("**Comentarios por dimensión**")
                fig, ax = plt.subplots(figsize=(5, 3))
                df["dimension"].value_counts().sort_values().plot(kind="barh", ax=ax, color="#3b82f6")
                ax.set_xlabel("Cantidad"); ax.set_title("Dimensiones evaluadas", fontweight="bold")
                plt.tight_layout(); st.pyplot(fig); plt.close()

        st.success("Datos listos. Ve al **Paso 2** para preprocesar el texto.")

# ══════════════════════════════════════════════════════════
# TAB 2 — PREPROCESAMIENTO
# ══════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🔧 Pipeline de Preprocesamiento de Texto")

    st.markdown("""<div class="explain-box">
    <b>¿Por qué preprocesar?</b><br>
    Los algoritmos de machine learning no entienden texto crudo. El preprocesamiento
    convierte los comentarios en listas de palabras clave que representan su significado.
    </div>""", unsafe_allow_html=True)

    # Explicación visual del pipeline
    st.markdown("**El pipeline tiene 4 etapas:**")
    etapas_cols = st.columns(4)
    etapas = [
        ("1. Limpieza", "#dbeafe", "Minúsculas, elimina URLs y caracteres especiales"),
        ("2. Tokenización", "#ede9fe", "Divide el texto en palabras individuales (tokens)"),
        ("3. Stopwords", "#fce7f3", "Elimina palabras vacías: 'el', 'la', 'de', 'que'..."),
        ("4. Stemming", "#dcfce7", "Reduce palabras a su raíz: 'explicando' → 'explic'"),
    ]
    for col, (titulo, color, desc) in zip(etapas_cols, etapas):
        col.markdown(f"""<div style="background:{color};border-radius:8px;padding:10px;text-align:center;height:90px">
        <b>{titulo}</b><br><small>{desc}</small></div>""", unsafe_allow_html=True)

    st.markdown("---")

    if st.session_state.df is None:
        st.warning("Primero carga los datos en el **Paso 1**.")
    else:
        df = st.session_state.df

        # Demo interactiva
        st.markdown("**Prueba el pipeline con un comentario:**")
        ejemplo_idx = st.slider("Selecciona un comentario (índice)", 0, len(df)-1, 0)
        texto_orig = str(df["comentario"].iloc[ejemplo_idx])

        col_demo, col_result = st.columns([1, 1])
        with col_demo:
            st.markdown("**Texto original:**")
            st.info(texto_orig)

        # Mostrar cada etapa
        t1 = limpiar_texto(texto_orig)
        t2 = tokenizar(t1)
        t3 = eliminar_stopwords(t2)
        t4 = aplicar_stemming(t3)
        t4 = [w for w in t4 if len(w) >= 3]

        with col_result:
            st.markdown("**Resultado final (tokens):**")
            tokens_html = " ".join([f'<span class="tag tag-stem">{w}</span>' for w in t4])
            st.markdown(tokens_html, unsafe_allow_html=True)

        with st.expander("Ver paso a paso la transformación"):
            pasos_demo = [
                ("Texto limpio", t1),
                ("Tokenizado", " | ".join(t2[:15]) + ("..." if len(t2) > 15 else "")),
                ("Sin stopwords", " | ".join(t3[:15])),
                ("Con stemming", " | ".join(t4)),
            ]
            for nombre, resultado in pasos_demo:
                st.markdown(f"**{nombre}:** `{resultado}`")

        st.markdown("---")

        # Procesar todo
        st.markdown("**Procesar el corpus completo:**")
        col_opt1, col_opt2, col_opt3 = st.columns(3)
        with col_opt1:
            min_len = st.slider("Longitud mínima de token", 2, 5, 3)
        with col_opt2:
            st.markdown("<br>", unsafe_allow_html=True)
            eliminar_stop = st.checkbox("Eliminar stopwords", value=True)
        with col_opt3:
            st.markdown("<br>", unsafe_allow_html=True)
            aplicar_stem = st.checkbox("Aplicar stemming", value=True)

        if st.button("🚀 Preprocesar corpus completo", type="primary", use_container_width=True):
            textos = df["comentario"].astype(str).tolist()
            with st.spinner("Procesando los 80 comentarios..."):
                corpus = []
                for texto in textos:
                    tokens = procesar_documento(texto, eliminar_stop=eliminar_stop, aplicar_stem=aplicar_stem)
                    tokens = [t for t in tokens if len(t) >= min_len]
                    corpus.append(tokens)
                st.session_state.corpus_tokens = corpus
                st.session_state.lda_model = None

            lens = [len(d) for d in corpus]
            all_tokens = [t for d in corpus for t in d]
            vocab = set(all_tokens)

            st.markdown(f"""<div class="result-box">
            <b>Preprocesamiento completado</b><br>
            Documentos procesados: <b>{len(corpus)}</b> &nbsp;|&nbsp;
            Vocabulario único: <b>{len(vocab)}</b> palabras &nbsp;|&nbsp;
            Promedio tokens/doc: <b>{np.mean(lens):.1f}</b>
            </div>""", unsafe_allow_html=True)

            # Top palabras
            top20 = Counter(all_tokens).most_common(20)
            words, freqs = zip(*top20)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.barh(list(words)[::-1], list(freqs)[::-1], color="#3b82f6")
            ax.set_title("Top-20 palabras del corpus (ya procesadas)", fontweight="bold")
            ax.set_xlabel("Frecuencia")
            plt.tight_layout(); st.pyplot(fig); plt.close()

            st.success("Corpus listo. Ve al **Paso 3** para entrenar el modelo LDA.")

        elif st.session_state.corpus_tokens is not None:
            st.info("Corpus ya procesado. Puedes ir al **Paso 3**.")

# ══════════════════════════════════════════════════════════
# TAB 3 — MODELO LDA
# ══════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🤖 Modelo LDA — Latent Dirichlet Allocation")

    st.markdown("""<div class="explain-box">
    <b>¿Qué hace LDA?</b><br>
    LDA asume que cada comentario mezcla varios <b>tópicos</b>, y que cada tópico
    es un grupo de palabras relacionadas. El algoritmo descubre automáticamente estos
    grupos <i>sin que nadie le diga qué buscar</i> — es <b>aprendizaje no supervisado</b>.
    <br><br>
    Ejemplo: si muchos comentarios mencionan "examen", "parcial" y "nota", LDA
    agrupa estas palabras en un tópico de <i>Evaluación</i>.
    </div>""", unsafe_allow_html=True)

    if st.session_state.corpus_tokens is None:
        st.warning("Primero ejecuta el **Preprocesamiento** en el Paso 2.")
    else:
        corpus_tokens = st.session_state.corpus_tokens
        vocab_size = len({t for d in corpus_tokens for t in d})

        col_info, col_params = st.columns([1, 1])

        with col_info:
            st.markdown("**Corpus listo para modelar:**")
            st.markdown(f"""<div class="step-box">
            Documentos: <b>{len(corpus_tokens)}</b><br>
            Vocabulario: <b>{vocab_size}</b> términos únicos<br>
            Tokens totales: <b>{sum(len(d) for d in corpus_tokens)}</b>
            </div>""", unsafe_allow_html=True)

            st.markdown("""<div class="explain-box">
            <b>¿Qué es K (número de tópicos)?</b><br>
            K es cuántos grupos temáticos quieres que LDA descubra.
            Si K=5, el modelo encontrará 5 temas distintos en los comentarios.
            Probar K=2 a 10 y elegir el que tenga mayor <b>coherencia</b> es la práctica estándar.
            </div>""", unsafe_allow_html=True)

        with col_params:
            st.markdown("**Configuración del modelo:**")
            num_topics = st.slider("K — Número de tópicos a descubrir", 2, 10, 5)
            passes = st.slider("Pasadas (mayor = más preciso, más lento)", 5, 30, 15)
            buscar_k = st.checkbox("Buscar K óptimo automáticamente (K=2 a 10)", value=False)

            st.caption("ℹ️ Con 80 documentos, K entre 4 y 6 suele dar buenos resultados.")

        if st.button("🚀 Entrenar Modelo LDA", type="primary", use_container_width=True):
            with st.spinner("Creando diccionario y corpus Bag-of-Words..."):
                dictionary, corpus_gensim = crear_diccionario_y_corpus(corpus_tokens)
                st.session_state.dictionary = dictionary
                st.session_state.corpus_gensim = corpus_gensim

            if buscar_k:
                st.markdown("**Buscando K óptimo...**")
                progress = st.progress(0)
                resultados = {}
                rango = list(range(2, 11))
                for i, k in enumerate(rango):
                    m = entrenar_lda(corpus_gensim, dictionary, num_topics=k, passes=10)
                    c = calcular_coherencia(m, corpus_gensim, dictionary, corpus_tokens)
                    resultados[k] = c
                    progress.progress((i+1)/len(rango))
                num_topics = max(resultados, key=resultados.get)
                st.success(f"K óptimo encontrado: **{num_topics}** (coherencia={resultados[num_topics]:.4f})")
                fig, _ = grafico_coherencia_vs_topicos(resultados)
                st.pyplot(fig); plt.close()

            with st.spinner(f"Entrenando LDA con K={num_topics}, {passes} pasadas..."):
                lda_model = entrenar_lda(corpus_gensim, dictionary, num_topics=num_topics, passes=passes)
                coherencia = calcular_coherencia(lda_model, corpus_gensim, dictionary, corpus_tokens)
                topicos_doc = asignar_topicos(lda_model, corpus_gensim)

            st.session_state.lda_model = lda_model
            st.session_state.topicos_doc = topicos_doc
            st.session_state.coherencia = coherencia
            st.session_state.num_topics = num_topics

            st.markdown(f"""<div class="result-box">
            <b>Modelo entrenado exitosamente</b><br>
            Tópicos (K): <b>{num_topics}</b> &nbsp;|&nbsp;
            Coherencia c_v: <b>{coherencia:.4f}</b> &nbsp;|&nbsp;
            Vocabulario: <b>{len(dictionary)}</b> términos
            </div>""", unsafe_allow_html=True)

            st.markdown("""<div class="explain-box">
            <b>¿Qué significa la coherencia?</b><br>
            La coherencia c_v mide qué tan relacionadas entre sí están las palabras de cada tópico.
            Valores entre <b>0.4 y 0.7</b> son buenos para corpus pequeños.
            Mayor coherencia = tópicos más interpretables.
            </div>""", unsafe_allow_html=True)

            st.success("Modelo listo. Ve al **Paso 4** para explorar los resultados.")

        elif st.session_state.lda_model is not None:
            c = st.session_state.coherencia
            st.info(f"Modelo ya entrenado: K={st.session_state.num_topics} | Coherencia={c:.4f}")

# ══════════════════════════════════════════════════════════
# TAB 4 — RESULTADOS
# ══════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 📊 Resultados: Tópicos y Análisis Likert")

    if st.session_state.lda_model is None:
        st.warning("Primero entrena el modelo en el **Paso 3**.")
    else:
        lda_model   = st.session_state.lda_model
        topicos_doc = st.session_state.topicos_doc
        topicos     = obtener_topicos(lda_model, num_words=10)

        # ─ Sección A: Tópicos ─
        st.markdown("#### A) Tópicos descubiertos por LDA")
        st.markdown("""<div class="explain-box">
        Cada tópico es un grupo de palabras que tienden a aparecer juntas en los comentarios.
        LDA les asigna un <b>peso</b> (probabilidad) a cada palabra: mayor peso = más representativa del tópico.
        </div>""", unsafe_allow_html=True)

        n = len(topicos)
        cols_topics = st.columns(min(n, 3))
        for i, (nombre, palabras_str) in enumerate(topicos.items()):
            col = cols_topics[i % 3]
            palabras_dict = parsear_topico_palabras(palabras_str)
            top5 = sorted(palabras_dict.items(), key=lambda x: x[1], reverse=True)[:5]
            tags = " ".join([f'<span class="tag">{w}</span>' for w, _ in top5])
            col.markdown(f"""<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px;margin:4px 0">
            <b>{nombre}</b><br>{tags}
            </div>""", unsafe_allow_html=True)

        # Nubes de palabras
        try:
            from wordcloud import WordCloud
            st.markdown("**Nubes de palabras por tópico:**")
            n_cols = min(n, 3)
            wc_cols = st.columns(n_cols)
            for i, (nombre, palabras_str) in enumerate(topicos.items()):
                palabras_dict = parsear_topico_palabras(palabras_str)
                if palabras_dict:
                    wc = WordCloud(width=350, height=200, background_color="white",
                                   colormap="Blues", max_words=30).generate_from_frequencies(palabras_dict)
                    fig, ax = plt.subplots(figsize=(4, 2.5))
                    ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
                    ax.set_title(nombre, fontweight="bold", fontsize=9)
                    plt.tight_layout()
                    wc_cols[i % n_cols].pyplot(fig); plt.close()
        except ImportError:
            pass

        # Distribución de tópicos
        dominantes = [t[0] for t in topicos_doc if t[0] is not None]
        if dominantes:
            st.markdown("**¿Cuántos comentarios hablan de cada tópico?**")
            counts = pd.Series(dominantes).value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.bar([f"Tópico {i}" for i in counts.index], counts.values, color="#3b82f6")
            ax.set_ylabel("Comentarios"); ax.set_title("Documentos por tópico dominante", fontweight="bold")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.divider()

        # ─ Sección B: Cruce Likert ─
        if st.session_state.df is not None and "calificacion_likert" in st.session_state.df.columns:
            st.markdown("#### B) Satisfacción Likert por tópico")
            st.markdown("""<div class="explain-box">
            Como el análisis no supervisado no sabe si un comentario es positivo o negativo,
            usamos la <b>calificación Likert</b> como indicador de satisfacción.
            Cruzar tópicos con Likert permite detectar <b>qué temas generan más insatisfacción</b>.
            </div>""", unsafe_allow_html=True)

            df = st.session_state.df.copy()
            n2 = min(len(df), len(topicos_doc))
            df = df.iloc[:n2].copy()
            df["topico_dom"] = [t[0] if t[0] is not None else -1 for t in topicos_doc[:n2]]
            df = df[df["topico_dom"] >= 0]

            avg = df.groupby("topico_dom")["calificacion_likert"].mean()

            g1, g2 = st.columns(2)
            with g1:
                fig, ax = plt.subplots(figsize=(5, 3.5))
                colores = ["#dc2626" if v < 3 else "#ca8a04" if v < 4 else "#16a34a" for v in avg.values]
                ax.bar([f"T{i}" for i in avg.index], avg.values, color=colores)
                ax.axhline(3, color="gray", linestyle="--", alpha=0.7, label="Neutral")
                ax.set_ylim(0, 5.5); ax.set_ylabel("Promedio Likert")
                ax.set_title("Satisfacción media por tópico", fontweight="bold"); ax.legend()
                plt.tight_layout(); st.pyplot(fig); plt.close()

            with g2:
                if "docente" in df.columns:
                    avg_doc = df.groupby("docente")["calificacion_likert"].mean().sort_values(ascending=False)
                    fig, ax = plt.subplots(figsize=(5, 3.5))
                    col_d = ["#16a34a" if v >= 4 else "#ca8a04" if v >= 3 else "#dc2626" for v in avg_doc.values]
                    ax.bar(avg_doc.index, avg_doc.values, color=col_d)
                    ax.axhline(3, color="gray", linestyle="--", alpha=0.5)
                    ax.set_ylim(0, 5.5); ax.set_ylabel("Promedio Likert")
                    ax.set_title("Satisfacción media por docente", fontweight="bold")
                    plt.xticks(rotation=20, fontsize=8); plt.tight_layout(); st.pyplot(fig); plt.close()

            # Tabla resumen
            st.markdown("**Resumen por tópico:**")
            resumen = df.groupby("topico_dom")["calificacion_likert"].agg(
                Media="mean", Total="count", Std="std"
            ).round(2)
            resumen.index = [f"Tópico {i}" for i in resumen.index]
            st.dataframe(resumen, use_container_width=True)

# ── Footer ──────────────────────────────────────────────
st.divider()
st.markdown("<p style='text-align:center;color:#94a3b8;font-size:0.8em'>Analizador de Opiniones · UMSS FCyT · Grupo 15 · I/2026</p>", unsafe_allow_html=True)

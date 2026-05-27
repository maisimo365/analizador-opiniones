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
from wordcloud import WordCloud
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import streamlit.components.v1 as components
import altair as alt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocesamiento import limpiar_texto, tokenizar, eliminar_stopwords, aplicar_stemming, procesar_documento
from src.modelo import crear_diccionario_y_corpus, entrenar_lda, calcular_coherencia, obtener_topicos, asignar_topicos, parsear_topico_palabras
from src.visualizacion import grafico_coherencia_vs_topicos

# ── Configuración Avanzada de Página ─────────────────────────────────
st.set_page_config(
    page_title="Analizador de Opiniones · UMSS",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilos CSS Formales e Institucionales ───────────────────────────
st.markdown("""
<style>
  /* Ocultar elementos heredados no corporativos de Streamlit */
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  
  /* Contenedores tipo Tarjeta de Control */
  .card-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 18px;
    margin-bottom: 12px;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
  }
  
  /* Bloques de Glosario o Instrucción Técnica */
  .card-instruction {
    background: #f1f5f9;
    border-left: 4px solid #475569;
    border-radius: 4px;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 0.95em;
    color: #334155;
  }
  
  /* Notificaciones Exitosas Personalizadas */
  .card-success-custom {
    background: #f0fdf4;
    border-left: 4px solid #16a34a;
    border-radius: 4px;
    padding: 14px 18px;
    margin: 10px 0;
    color: #14532d;
    font-size: 0.95em;
  }
  
  /* Badges Semánticos para el Vocabulario */
  .tag {
    display: inline-block;
    background: #eff6ff;
    color: #1e40af;
    border: 1px solid #bfdbfe;
    border-radius: 4px;
    padding: 2px 8px;
    margin: 3px;
    font-size: 0.85em;
    font-weight: 500;
  }
  .tag-stem {
    background: #f0fdf4;
    color: #15803d;
    border: 1px solid #bbf7d0;
  }
</style>
""", unsafe_allow_html=True)

# ── Inicialización del Estado de Sesión ──────────────────────────────
for k, v in {
    "df": None, "corpus_tokens": None, "dictionary": None,
    "corpus_gensim": None, "lda_model": None, "topicos_doc": None,
    "coherencia": None, "num_topics": 5, 
    "filtro_docente": "Todos", "filtro_materia": "Todas"
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Carga Automatizada de Datos ──────────────────────────────────────
DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "encuesta_sintetica.csv"
if st.session_state.df is None and DATA_PATH.exists():
    st.session_state.df = pd.read_csv(DATA_PATH)

# ── Barra Lateral de Filtros (Sidebar) ───────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Filtros Globales")
    st.markdown("<p style='font-size:0.9em; color:#475569;'>Aplica estos filtros para aislar el análisis a un docente o materia específica en la Pestaña 4.</p>", unsafe_allow_html=True)
    
    if st.session_state.df is not None:
        with st.form(key='filtros_form'):
            if "docente" in st.session_state.df.columns:
                docentes_unicos = ["Todos"] + sorted(st.session_state.df["docente"].dropna().unique().tolist())
                # Recuperar el valor actual si es válido
                doc_idx = docentes_unicos.index(st.session_state.filtro_docente) if st.session_state.filtro_docente in docentes_unicos else 0
                docente_filtro = st.selectbox("👨‍🏫 Docente:", docentes_unicos, index=doc_idx)
            else:
                docente_filtro = "Todos"
                
            if "materia" in st.session_state.df.columns:
                materias_unicas = ["Todas"] + sorted(st.session_state.df["materia"].dropna().unique().tolist())
                mat_idx = materias_unicas.index(st.session_state.filtro_materia) if st.session_state.filtro_materia in materias_unicas else 0
                materia_filtro = st.selectbox("📚 Materia:", materias_unicas, index=mat_idx)
            else:
                materia_filtro = "Todas"
            
            submit_button = st.form_submit_button(label='FILTRAR', type='primary', use_container_width=True)
            if submit_button:
                st.session_state.filtro_docente = docente_filtro
                st.session_state.filtro_materia = materia_filtro
                st.rerun()
    else:
        st.info("Cargue un dataset para habilitar.")

# ── Encabezado Principal de la Aplicación ────────────────────────────
st.markdown("# Analizador Temático de Opiniones Docentes")
st.markdown("**Universidad Mayor de San Simón · Facultad de Ciencias y Tecnología** · *Área: Inteligencia Artificial (Grupo 15)*")
st.markdown("Extracción automatizada de conocimiento en encuestas estudiantiles mediante Procesamiento de Lenguaje Natural (PLN) y Modelado del Tópicos (LDA).")
st.divider()

# ── Monitor de Progreso del Pipeline ─────────────────────────────────
datos_ok   = st.session_state.df is not None
preproc_ok = st.session_state.corpus_tokens is not None
modelo_ok  = st.session_state.lda_model is not None

cols_step = st.columns(4)
pasos = ["Fase 1: Carga de Datos", "Fase 2: Preprocesamiento", "Fase 3: Configuración LDA", "Fase 4: Análisis de Resultados"]
for i, paso in enumerate(pasos):
    done = [datos_ok, preproc_ok, modelo_ok, modelo_ok][i]
    status_text = "COMPLETADO" if done else "PENDIENTE"
    status_color = "#16a34a" if done else "#94a3b8"
    cols_step[i].markdown(f"""
    <div style='text-align:center; padding: 8px; border: 1px solid {status_color}; border-radius: 6px; background: #ffffff;'>
        <span style='color:{status_color}; font-weight:600; font-size:0.9em;'>{paso}</span><br>
        <span style='color:{status_color}; font-size:0.8em;'>● {status_text}</span>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Sistema de Navegación Estilo Panel de Control ───────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Gestión de Datasets",
    "Preprocesamiento de Texto",
    "Entrenamiento del Modelo",
    "Visualización de Resultados",
])

# ══════════════════════════════════════════════════════════
# TAB 1 — GESTIÓN DE DATASETS
# ══════════════════════════════════════════════════════════
with tab1:
    st.subheader("Carga y Estructura del Dataset")

    st.markdown("""<div class="card-instruction">
    <b>Descripción del Corpus:</b> Cada fila mapea la evaluación cualitativa (comentario en texto libre) y cuantitativa (escala Likert de 1 a 5) emitida por un estudiante respecto a una dimensión pedagógica específica (Metodología, Evaluación, Comunicación, Puntualidad o Material).
    </div>""", unsafe_allow_html=True)

    col_btn, col_up = st.columns([1, 1])
    with col_btn:
        if st.button("Cargar conjunto de datos de muestra (80 comentarios)", use_container_width=True, type="primary"):
            st.session_state.df = pd.read_csv(DATA_PATH)
            st.session_state.corpus_tokens = None
            st.session_state.lda_model = None
    with col_up:
        f = st.file_uploader("Cargar archivo CSV personalizado", type="csv", label_visibility="collapsed")
        if f:
            st.session_state.df = pd.read_csv(f)
            st.session_state.corpus_tokens = None
            st.session_state.lda_model = None

    if st.session_state.df is not None:
        df = st.session_state.df

        # Bloque de Métricas Descriptivas Básicas (KPIs)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Volumen de Comentarios", len(df))
        m2.metric("Docentes Registrados", df["docente"].nunique() if "docente" in df.columns else "—")
        m3.metric("Dimensiones Evaluadas", df["dimension"].nunique() if "dimension" in df.columns else "—")
        m4.metric("Puntuación Likert Media", f"{df['calificacion_likert'].mean():.2f} / 5" if "calificacion_likert" in df.columns else "—")

        st.markdown("---")
        st.markdown("**Muestra preliminar de los registros cargados:**")
        st.dataframe(df.head(8), use_container_width=True, hide_index=True)

        if "calificacion_likert" in df.columns and "dimension" in df.columns:
            st.markdown("---")
            g1, g2 = st.columns(2)

            with g1:
                st.markdown("**Distribución de Frecuencias: Escala Likert**")
                labels = {1:"1 - Muy malo", 2:"2 - Malo", 3:"3 - Regular", 4:"4 - Bueno", 5:"5 - Muy bueno"}
                counts = df["calificacion_likert"].value_counts().sort_index()
                
                chart_data = pd.DataFrame({
                    "Escala Likert": [labels[i] for i in counts.index],
                    "Cantidad": counts.values,
                    "Color": ["#dc2626", "#ea580c", "#ca8a04", "#16a34a", "#1e40af"]
                })
                
                chart1 = alt.Chart(chart_data).mark_bar().encode(
                    x=alt.X("Escala Likert", axis=alt.Axis(labelAngle=0)),
                    y="Cantidad",
                    color=alt.Color("Color", scale=None),
                    tooltip=["Escala Likert", "Cantidad"]
                ).properties(height=350)
                st.altair_chart(chart1, use_container_width=True)

            with g2:
                st.markdown("**Distribución por Dimensión Evaluada**")
                counts_dim = df["dimension"].value_counts().reset_index()
                counts_dim.columns = ["Dimensión", "Frecuencia"]
                
                chart2 = alt.Chart(counts_dim).mark_bar(color="#475569").encode(
                    x="Frecuencia",
                    y=alt.Y("Dimensión", sort="-x"),
                    tooltip=["Dimensión", "Frecuencia"]
                ).properties(height=350)
                st.altair_chart(chart2, use_container_width=True)

        st.info("Estructura de datos validada. Continúe a la pestaña de **Preprocesamiento de Texto**.")

# ══════════════════════════════════════════════════════════
# TAB 2 — PREPROCESAMIENTO DE TEXTO
# ══════════════════════════════════════════════════════════
with tab2:
    st.subheader("Pipeline de Normalización y Limpieza Textual")

    st.markdown("""<div class="card-instruction">
    <b>Fundamento de Operación:</b> Para que el algoritmo probabilístico pueda procesar los textos, se debe remover el ruido gramatical. Este pipeline transforma cadenas heterogéneas en matrices estructuradas de tokens semánticos.
    </div>""", unsafe_allow_html=True)

    etapas_cols = st.columns(4)
    etapas = [
        ("1. Limpieza Estructural", "Conversión a minúsculas y eliminación drástica de caracteres especiales, números y URLs."),
        ("2. Tokenización", "Segmentación del texto plano continuo en vectores unitarios de términos aislados."),
        ("3. Filtrado de Stopwords", "Eliminación sistemática de palabras funcionales sin peso semántico (artículos, preposiciones)."),
        ("4. Stemming Léxico", "Reducción de las variantes morfológicas de una palabra a su raíz base común."),
    ]
    for col, (titulo, desc) in zip(etapas_cols, etapas):
        col.markdown(f"""
        <div style="background:#ffffff; border:1px solid #cbd5e1; border-radius:6px; padding:14px; height:115px;">
            <b style="color:#1e3a8a; font-size:0.95em;">{titulo}</b><br>
            <span style="font-size:0.85em; color:#475569;">{desc}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    if st.session_state.df is None:
        st.warning("Acción requerida: Cargue un dataset en la pestaña inicial antes de proceder.")
    else:
        df = st.session_state.df

        st.markdown("**Simulador Unitario de Transformación Léxica:**")
        ejemplo_idx = st.slider("Seleccione el índice del comentario a evaluar:", 0, len(df)-1, 0)
        texto_orig = str(df["comentario"].iloc[ejemplo_idx])

        col_demo, col_result = st.columns([1, 1])
        with col_demo:
            st.markdown("*Texto original en crudo:*")
            st.text_area(label="Texto original", value=texto_orig, disabled=True, label_visibility="collapsed")

        t1 = limpiar_texto(texto_orig)
        t2 = tokenizar(t1)
        t3 = eliminar_stopwords(t2)
        t4 = aplicar_stemming(t3)
        t4 = [w for w in t4 if len(w) >= 3]

        with col_result:
            st.markdown("*Tokens resultantes del pipeline:*")
            tokens_html = f'<div style="background:#ffffff; padding:12px; border:1px dashed #cbd5e1; border-radius:6px; min-height:76px;">' + " ".join([f'<span class="tag tag-stem">{w}</span>' for w in t4]) + '</div>'
            st.markdown(tokens_html, unsafe_allow_html=True)

        with st.expander("Inspeccionar la salida secuencial por fase"):
            st.markdown(f"**Fase 1 (Limpieza):** `{t1}`")
            st.markdown(f"**Fase 2 (Tokenización):** `{' | '.join(t2[:12])}`")
            st.markdown(f"**Fase 3 (Filtro Stopwords):** `{' | '.join(t3[:12])}`")
            st.markdown(f"**Fase 4 (Stemming):** `{' | '.join(t4)}`")

        st.markdown("---")
        st.markdown("**Procesamiento en Bloque del Corpus:**")
        col_opt1, col_opt2, col_opt3 = st.columns(3)
        with col_opt1:
            min_len = st.slider("Longitud mínima admitida por palabra (tokens):", 2, 5, 3)
        with col_opt2:
            st.markdown("<br>", unsafe_allow_html=True)
            eliminar_stop = st.checkbox("Remover Stopwords obligatoriamente", value=True)
        with col_opt3:
            st.markdown("<br>", unsafe_allow_html=True)
            aplicar_stem = st.checkbox("Ejecutar Stemming en el corpus", value=True)

        if st.button("Ejecutar transformación sobre la matriz completa", type="primary", use_container_width=True):
            textos = df["comentario"].astype(str).tolist()
            with st.spinner("Procesando matriz textual de datos..."):
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

            st.markdown(f"""<div class="card-success-custom">
            <b>Pipeline global finalizado correctamente:</b><br>
            Documentos procesados: {len(corpus)} | Dimensión del vocabulario único: {len(vocab)} términos lógicos | Densidad media: {np.mean(lens):.2f} tokens/documento.
            </div>""", unsafe_allow_html=True)

            top20 = Counter(all_tokens).most_common(20)
            words, freqs = zip(*top20)
            
            df_top = pd.DataFrame({"Término": words, "Frecuencia": freqs})
            
            chart_top = alt.Chart(df_top).mark_bar(color="#2563eb").encode(
                x="Frecuencia",
                y=alt.Y("Término", sort="-x"),
                tooltip=["Término", "Frecuencia"]
            ).properties(title="Frecuencia de Ocurrencia: Top 20 Términos", height=400)
            st.altair_chart(chart_top, use_container_width=True)

        elif st.session_state.corpus_tokens is not None:
            st.info("El corpus ya se encuentra procesado y almacenado en la memoria activa de la sesión.")

# ══════════════════════════════════════════════════════════
# TAB 3 — CONFIGURACIÓN Y MODELO LDA
# ══════════════════════════════════════════════════════════
with tab3:
    st.subheader("Entrenamiento del Algoritmo Matemático LDA")

    st.markdown("""<div class="card-instruction">
    <b>Definición Matemática:</b> Latent Dirichlet Allocation (LDA) modela los documentos como mezclas de tópicos latentes distribuidos bajo una distribución Dirichlet, donde cada tópico se define a su vez como una distribución de probabilidad de palabras clave co-ocurrentes.
    </div>""", unsafe_allow_html=True)

    if st.session_state.corpus_tokens is None:
        st.warning("Módulo bloqueado: Requiere la ejecución y consolidación de la Fase 2 de Preprocesamiento.")
    else:
        corpus_tokens = st.session_state.corpus_tokens
        vocab_size = len({t for d in corpus_tokens for t in d})

        col_info, col_params = st.columns([1, 1])

        with col_info:
            st.markdown("**Atributos actuales del Corpus:**")
            st.markdown(f"""<div class="card-box">
            • Documentos estructurados: <b>{len(corpus_tokens)}</b><br>
            • Tamaño del vocabulario activo: <b>{vocab_size}</b> términos<br>
            • Volumen consolidado de tokens: <b>{sum(len(d) for d in corpus_tokens)}</b>
            </div>""", unsafe_allow_html=True)

        with col_params:
            st.markdown("**Ajuste de Hiperparámetros:**")
            num_topics = st.slider("Cantidad de tópicos latentes a inferir (K):", 2, 10, 5)
            passes = st.slider("Iteraciones de optimización (Passes):", 5, 30, 15)
            buscar_k = st.checkbox("Optimización automática del parámetro K (Maximizar Coherencia)", value=False)

        if st.button("Iniciar Ejecución del Modelo probabilístico", type="primary", use_container_width=True):
            dictionary, corpus_gensim = crear_diccionario_y_corpus(corpus_tokens)
            st.session_state.dictionary = dictionary
            st.session_state.corpus_gensim = corpus_gensim

            if buscar_k:
                st.markdown("**Evaluando consistencia estructural para K óptimo...**")
                progress = st.progress(0)
                resultados = {}
                rango = list(range(2, 11))
                for i, k in enumerate(rango):
                    m = entrenar_lda(corpus_gensim, dictionary, num_topics=k, passes=10)
                    c = calcular_coherencia(m, corpus_gensim, dictionary, corpus_tokens)
                    resultados[k] = c
                    progress.progress((i+1)/len(rango))
                num_topics = max(resultados, key=resultados.get)
                st.success(f"Configuración óptima determinada en K = {num_topics} (Métrica de Coherencia C_v = {resultados[num_topics]:.4f})")
                fig, _ = grafico_coherencia_vs_topicos(resultados)
                st.pyplot(fig)
                plt.close()

            with st.spinner("Optimizando las matrices Dirichlet latentes..."):
                lda_model = entrenar_lda(corpus_gensim, dictionary, num_topics=num_topics, passes=passes)
                coherencia = calcular_coherencia(lda_model, corpus_gensim, dictionary, corpus_tokens)
                topicos_doc = asignar_topicos(lda_model, corpus_gensim)

            st.session_state.lda_model = lda_model
            st.session_state.topicos_doc = topicos_doc
            st.session_state.coherencia = coherencia
            st.session_state.num_topics = num_topics

            st.markdown(f"""<div class="card-success-custom">
            <b>Entrenamiento finalizado de forma convergente:</b><br>
            Componentes temáticos deducidos (K): {num_topics} | Índice de coherencia semántica C_v: {coherencia:.4f} | Atributos vectoriales: {len(dictionary)} variables lógicas.
            </div>""", unsafe_allow_html=True)

        elif st.session_state.lda_model is not None:
            c = st.session_state.coherencia
            st.info(f"Modelo activo en la sesión actual: K = {st.session_state.num_topics} | Coherencia C_v = {c:.4f}")

# ══════════════════════════════════════════════════════════
# TAB 4 — VISUALIZACIÓN DE RESULTADOS
# ══════════════════════════════════════════════════════════
with tab4:
    st.subheader("Clasificación Temática e Intersección Métrica")

    if st.session_state.lda_model is None:
        st.warning("Módulo bloqueado: Requiere la compilación del modelo matemático en la pestaña anterior.")
    else:
        lda_model   = st.session_state.lda_model
        topicos_doc = st.session_state.topicos_doc
        topicos     = obtener_topicos(lda_model, num_words=10)

        st.markdown("#### Distribución de Términos por Clúster Temático")
        st.markdown("""<div class="card-instruction">
        El modelado agrupa palabras con una alta probabilidad de aparecer de forma conjunta en los documentos originales. A continuación se presentan los 5 términos más determinantes de cada tópico deducido.
        </div>""", unsafe_allow_html=True)

        n = len(topicos)
        cols_topics = st.columns(min(n, 3))
        for i, (nombre, palabras_str) in enumerate(topicos.items()):
            col = cols_topics[i % 3]
            palabras_dict = parsear_topico_palabras(palabras_str)
            
            # Generar Nube de Palabras
            wc = WordCloud(width=600, height=350, background_color="white", colormap="Blues", prefer_horizontal=0.8).generate_from_frequencies(palabras_dict)
            fig_wc, ax_wc = plt.subplots(figsize=(6, 3.5))
            ax_wc.imshow(wc, interpolation="bilinear")
            ax_wc.axis("off")
            
            with col:
                st.markdown(f"<div style='text-align:center; color:#1e3a8a; font-weight:700; margin-bottom:5px;'>{nombre.upper()}</div>", unsafe_allow_html=True)
                st.pyplot(fig_wc)
                plt.close(fig_wc)

        st.markdown("---")
        st.markdown("#### Explorador Interactivo de Tópicos (pyLDAvis)")
        st.markdown("<div class='card-instruction'>Mapa de distancia inter-tópica (PCA). Los círculos representan los tópicos; su tamaño indica la prevalencia y la distancia indica cuán distintos son entre sí. Pasa el ratón sobre ellos o sobre las palabras para ver sus relaciones.</div>", unsafe_allow_html=True)
        
        with st.spinner("Renderizando motor de visualización pyLDAvis..."):
            vis_data = gensimvis.prepare(lda_model, st.session_state.corpus_gensim, st.session_state.dictionary)
            html_string = pyLDAvis.prepared_data_to_html(vis_data)
            components.html(html_string, width=1300, height=800, scrolling=True)

        # Matriz de volumen por clúster
        dominantes = [t[0] for t in topicos_doc if t[0] is not None]
        if dominantes:
            st.markdown("---")
            st.markdown("**Densidad cuantitativa de documentos por tópico dominante:**")
            counts = pd.Series(dominantes).value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.bar([f"Tópico {i}" for i in counts.index], counts.values, color="#1d4ed8", width=0.45)
            ax.set_ylabel("Volumen de Comentarios")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.divider()

        # Intersección analítica con Escala Likert
        if st.session_state.df is not None and "calificacion_likert" in st.session_state.df.columns:
            st.markdown("#### Matriz de Intersección: Correlación Temática y Escala Likert")
            st.markdown("""<div class="card-instruction">
            <b>Enfoque Analítico Híbrido:</b> Al cruzar el modelado temático no supervisado con las calificaciones numéricas, podemos identificar con precisión qué discusiones o temas latentes se correlacionan con la insatisfacción estudiantil.
            </div>""", unsafe_allow_html=True)

            df = st.session_state.df.copy()
            n2 = min(len(df), len(topicos_doc))
            df = df.iloc[:n2].copy()
            df["topico_dom"] = [t[0] if t[0] is not None else -1 for t in topicos_doc[:n2]]
            df = df[df["topico_dom"] >= 0]

            # Aplicar los filtros del Sidebar
            if st.session_state.filtro_docente != "Todos" and "docente" in df.columns:
                df = df[df["docente"] == st.session_state.filtro_docente]
            if st.session_state.filtro_materia != "Todas" and "materia" in df.columns:
                df = df[df["materia"] == st.session_state.filtro_materia]

            if df.empty:
                st.warning("⚠️ No hay comentarios que coincidan con los filtros seleccionados. Intenta cambiar los filtros en la barra lateral.")
            else:
                avg = df.groupby("topico_dom")["calificacion_likert"].mean()
    
                g1, g2 = st.columns(2)
                with g1:
                    st.markdown("**Satisfacción Promedio por Tópico**")
                    df_g1 = pd.DataFrame({
                        "Tópico": [f"Tópico {i}" for i in avg.index],
                        "Puntuación": avg.values.round(2),
                        "Color": ["#dc2626" if v < 3 else "#d97706" if v < 4 else "#16a34a" for v in avg.values]
                    })
                    
                    base1 = alt.Chart(df_g1).encode(
                        x=alt.X("Tópico", axis=alt.Axis(labelAngle=0)),
                        y=alt.Y("Puntuación", scale=alt.Scale(domain=[0, 5.5])),
                        tooltip=["Tópico", "Puntuación"]
                    )
                    bar1 = base1.mark_bar(size=40).encode(color=alt.Color("Color", scale=None))
                    rule1 = alt.Chart(pd.DataFrame({'y': [3]})).mark_rule(color="#64748b", strokeDash=[5, 5]).encode(y='y')
                    
                    st.altair_chart((bar1 + rule1).properties(height=350), use_container_width=True)
    
                with g2:
                    if "docente" in df.columns:
                        st.markdown("**Satisfacción Promedio por Docente**")
                        avg_doc = df.groupby("docente")["calificacion_likert"].mean().sort_values(ascending=False)
                        df_g2 = pd.DataFrame({
                            "Docente": avg_doc.index,
                            "Puntuación": avg_doc.values.round(2),
                            "Color": ["#16a34a" if v >= 4 else "#d97706" if v >= 3 else "#dc2626" for v in avg_doc.values]
                        })
                        
                        base2 = alt.Chart(df_g2).encode(
                            x=alt.X("Docente", sort=None, axis=alt.Axis(labelAngle=-45)),
                            y=alt.Y("Puntuación", scale=alt.Scale(domain=[0, 5.5])),
                            tooltip=["Docente", "Puntuación"]
                        )
                        bar2 = base2.mark_bar(size=40).encode(color=alt.Color("Color", scale=None))
                        rule2 = alt.Chart(pd.DataFrame({'y': [3]})).mark_rule(color="#64748b", strokeDash=[5, 5]).encode(y='y')
                        
                        st.altair_chart((bar2 + rule2).properties(height=350), use_container_width=True)
    
                st.markdown("**Resumen de Datos Descriptivos por Componente:**")
                resumen = df.groupby("topico_dom")["calificacion_likert"].agg(
                    Puntuacion_Media="mean", Volumen_Muestral="count", Desviacion_Estandar="std"
                ).round(2)
                resumen.index = [f"Tópico {i}" for i in resumen.index]
                st.dataframe(resumen, use_container_width=True)
    
                st.markdown("---")
                col_down1, col_down2 = st.columns([1, 1])
                with col_down1:
                    csv_data = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=" Descargar Dataset Filtrado (CSV)",
                        data=csv_data,
                        file_name="encuestas_clasificadas_lda.csv",
                        mime="text/csv",
                        use_container_width=True,
                        type="primary"
                    )
                with col_down2:
                    st.markdown("<div style='padding-top:10px; color:#475569; font-size:0.85em;'>Exporta los resultados cruzados con los tópicos detectados para análisis en herramientas externas (Excel, PowerBI).</div>", unsafe_allow_html=True)

# ── Pie de Página Institucional ──────────────────────────────────────
st.divider()
st.markdown("<p style='text-align:center; color:#64748b; font-size:0.85em;'>Universidad Mayor de San Simón · Facultad de Ciencias y Tecnología<br>Laboratorio de Inteligencia Artificial · Gestión I/2026</p>", unsafe_allow_html=True)
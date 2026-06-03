# Analizador de opiniones de estudiantes sobre docentes/materias

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B.svg)
![NLP](https://img.shields.io/badge/NLP-NLTK%20%7C%20Gensim-brightgreen.svg)

Este proyecto propone un sistema que aprovecha la inteligencia artificial para analizar de forma automática textos de respuestas abiertas en encuestas de evaluación docente. Utiliza técnicas de Procesamiento del Lenguaje Natural (PLN) y aprendizaje no supervisado para descubrir los temas que los estudiantes mencionan, adaptándose a lo que realmente les preocupa sin necesidad de definir categorías predefinidas.

## 🎯 Objetivos

### Objetivo General
Desarrollar un sistema basado en Procesamiento del Lenguaje Natural y aprendizaje no supervisado que analice las respuestas abiertas de encuestas de evaluación docente y descubra automáticamente los temas y subtemas recurrentes en las opiniones de los estudiantes, combinando los resultados con las calificaciones Likert para entregar a los coordinadores académicos una visión integral de la percepción estudiantil.

### Objetivo Específico
Identificar los temas transversales que los estudiantes expresan libremente, mediante la aplicación de LDA (Latent Dirichlet Allocation) al bloque de preguntas abiertas generales, permitiendo detectar preocupaciones o reconocimientos que escapan a las dimensiones predefinidas.

## ⚙️ Funcionamiento y Flujo del Sistema

El prototipo funciona a través de una aplicación web interactiva desarrollada en **Streamlit** y consta de 4 flujos principales:

1. **Paso 1 (Datos):** Una interfaz sencilla para subir los archivos CSV con las encuestas.
2. **Paso 2 (Preprocesamiento):** Un entorno interactivo donde el texto crudo pasa por un pipeline de NLP (Limpieza, Tokenización, eliminación de Stopwords y Stemming con SnowballStemmer).
3. **Paso 3 (Modelo LDA):** Panel de control del algoritmo para entrenar el modelo en tiempo real, ajustar manualmente la cantidad de tópicos ($K$) y el número de pasadas, o ejecutar una búsqueda automática para encontrar el valor óptimo.
4. **Paso 4 (Resultados):** Reporte final que visualiza los hallazgos mediante nubes de palabras, gráficos estadísticos y mapas interactivos, cruzando los temas extraídos por la IA con la calificación original (Likert) del estudiante.

## 🛠️ Tecnologías y Herramientas Utilizadas

*   **Lenguaje Base:** Python (versión 3.10 o superior).
*   **Procesamiento de Lenguaje Natural (PLN):** Biblioteca `NLTK` para tokenización y stemming.
*   **Modelado y Machine Learning:** `Gensim` (LdaModel, CoherenceModel) como motor prioritario. `Scikit-Learn` para vectorización (TfidfVectorizer, CountVectorizer).
*   **Manipulación de Datos:** `Pandas` y `NumPy`.
*   **Desarrollo de Interfaz Web:** `Streamlit`.
*   **Visualización Científica:** 
    *   `Matplotlib` y `Seaborn` (gráficos estáticos).
    *   `Altair` (gráficos dinámicos e interactivos).
    *   `WordCloud` (nubes de palabras).
    *   `pyLDAvis` (mapa de proyección de dimensionalidad PCA de tópicos latentes).

## 🧠 Marco Teórico

### Procesamiento de Lenguaje Natural (PLN)
Se aplica un pipeline de preprocesamiento en español:
*   **Limpieza:** Eliminación de caracteres especiales, URLs, números y signos de puntuación.
*   **Tokenización:** División del texto en palabras.
*   **Stopwords:** Eliminación de palabras sin carga semántica (artículos, preposiciones) y palabras de dominio poco informativas ("docente", "materia").
*   **Stemming:** Uso de `SnowballStemmer` para reducir las palabras a su raíz morfológica común.
*   **Vectorización:** Representación de palabras en formato numérico usando Bag of Words (BoW) y TF-IDF.

### Modelado de Tópicos (LDA)
El modelado se basa en **Latent Dirichlet Allocation (LDA)**, un modelo probabilístico generativo. El algoritmo agrupa palabras que tienden a aparecer juntas para inferir los tópicos latentes en el conjunto de comentarios.

### Calificación Likert como Proxy de Polaridad
A diferencia del análisis de sentimientos tradicional, este sistema aprovecha la estructura de la encuesta: asocia cada comentario abierto a su calificación cuantitativa en la escala Likert (ej. de 1 a 5), permitiendo identificar si los temas mencionados tienen una connotación positiva o negativa.

## 🚀 Instalación y Ejecución

Sigue estos pasos para instalar y ejecutar el proyecto en tu máquina local:

1. **Clonar el repositorio** (o descargar el código fuente):
   ```bash
   git clone https://github.com/AndresGGfd/analizador-opiniones.git
   cd analizador-opiniones
   ```

2. **Crear y activar un entorno virtual** (recomendado):
   ```bash
   # En Windows
   python -m venv venv
   .\venv\Scripts\activate

   # En macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Instalar las dependencias**:
   Asegúrate de estar en la carpeta raíz del proyecto y ejecuta:
   ```bash
   pip install -r requirements.txt
   ```

4. **Ejecutar la aplicación Streamlit**:
   Una vez instaladas las dependencias, lanza la aplicación web con:
   ```bash
   streamlit run app/app.py
   ```
   *Esto abrirá automáticamente una pestaña en tu navegador web por defecto (generalmente en `http://localhost:8501`).*

## 👥 Equipo (Grupo 15)
*   Huayllani López Lisandro Antonio
*   Vela Uribe William
*   Zeballos Romero Jhoel Andres
*   Massi Geronimo Miguel Angel
*   Lizarazu Ferrufino Melina
*   Pérez Navia René Andrés
*   Ayra Torrico Gonzalo
*   Ayala Claros Cristian

*Universidad Mayor de San Simón - Facultad de Ciencias y Tecnología*
*Materia: Inteligencia Artificial*

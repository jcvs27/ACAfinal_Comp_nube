# =============================================================================
#  app.py  —  RAG PDF Expert Chat (Robust 429 Flow Control)
# -----------------------------------------------------------------------------
#
#  Convención:
#    · El comentario a la derecha de una línea explica ESA línea.
#    · El comentario encima de un bloque explica el bloque.
#    · Las líneas en blanco solo separan bloques y no llevan explicación.
# =============================================================================

# ----------------------------------------------------------------------------
# 1) IMPORTACIONES — las herramientas que usa el programa
# ----------------------------------------------------------------------------

import streamlit as st            # Framework web: convierte este script en una app interactiva en el navegador
import numpy as np                # Álgebra numérica: arrays y cálculo de la similitud coseno
from pypdf import PdfReader       # Lector de PDF: abre el archivo y extrae el texto de cada página
import time                       # time.sleep(): pausa el programa durante los reintentos (backoff)
import os                         # os.environ: lee variables de entorno, aquí la API key
import re                         # re: busca el "retry in 1.49s" que devuelve Gemini al agotar cuota
import inspect                    # inspect: comprueba si st.file_uploader acepta max_upload_size
import random                     # random: introduce 'jitter' (ruido aleatorio) para evitar colisiones en reintentos
from google import genai          # SDK oficial de Google Gen AI (Gemini): cliente para embeddings y chat
from google.genai import types    # Tipos auxiliares (Content, Part) para construir las peticiones al modelo
from google.genai.errors import APIError  # Excepción del SDK; se importa aunque el código no la use directamente

# ----------------------------------------------------------------------------
# 2) CONSTANTES DE CONFIGURACIÓN
# ----------------------------------------------------------------------------

# --- UTILIDADES DEL PIPELINE RAG ---   # Marcador de sección: aquí empiezan las funciones del motor RAG
# (línea en blanco: solo separa bloques)
EMBEDDING_MODEL = "gemini-embedding-2"   # Nombre del modelo que convierte texto → vector; se usa en TODO el programa
EMBEDDING_BATCH_SIZE = 10         # Reducido de 20 a 10 para mitigar picos de consumo de tokens y rate-limiting de golpe

# Cuota del nivel gratuito de Gemini para embeddings: peticiones por minuto.
EMBEDDING_RPM = 80                # Bajado de 90 a 80 para mayor margen de seguridad con respecto al límite Free Tier

# Tamaño máximo por archivo que se acepta en el cargador (en MB).
MAX_PDF_MB = 20                   # 20 MB: valor por defecto de Streamlit y de la API inline

# ----------------------------------------------------------------------------
# 3) EXTRACCIÓN DE TEXTO DEL PDF
# ----------------------------------------------------------------------------

def extract_text_from_pdf(pdf_file):   # Define la función; recibe un archivo PDF
    """Extrae el texto de un PDF, ya sea una ruta de archivo o un archivo abierto (file-like) de Streamlit."""   # Docstring
    reader = PdfReader(pdf_file)   # Abre el PDF con pypdf
    text = ""                      # Acumulador
    for page in reader.pages:      # Recorre cada página en orden
        page_text = page.extract_text()   # Extrae el texto
        if page_text:                     # Si no es nulo...
            text += page_text + "\n"      # ...añade al acumulador
    return text                    # Devuelve todo el texto

# ----------------------------------------------------------------------------
# 4) TROCEADO (CHUNKING) DEL TEXTO
# ----------------------------------------------------------------------------

def chunk_text(text, chunk_size=1000, overlap=100):   # Parte el texto en trozos
    """Parte el texto en fragmentos de 'chunk_size' caracteres, con solape de 'overlap'."""   # Docstring
    chunks = []                    # Lista vacía
    start = 0                      # Posición inicial
    while start < len(text):       # Mientras no termine...
        end = start + chunk_size   # ...calcula fin
        chunks.append(text[start:end])   # Añade fragmento
        start += (chunk_size - overlap)  # Avanza considerando solape
    return chunks                  # Devuelve fragmentos

# ----------------------------------------------------------------------------
# 5) FORMATO DE TEXTO PARA EL MODELO (prefijos de tarea)
# ----------------------------------------------------------------------------

def format_query(text):   # Prepara pregunta
    """Prefijo de tarea para consultas (gemini-embedding-2)."""   # Docstring
    return f"task: search result | query: {text}"   # Retorna con prefijo de búsqueda

def format_document(text, title=None):   # Prepara fragmento
    """Prefijo de tarea para documentos indexados."""   # Docstring
    return f"title: {title or 'none'} | text: {text}"   # Retorna prefijo de documento

# ----------------------------------------------------------------------------
# 6) EMBEDDINGS CON REINTENTOS Y MITIGACIÓN AVANZADA DE 429
# ----------------------------------------------------------------------------

class RateLimiter:                # Control de ventana de peticiones
    """Ventana deslizante de 60 s para evitar cuota excedida."""   # Docstring

    def __init__(self, rpm):      # Constructor
        self.rpm = max(1, int(rpm))   # Límite RPM mínimo de 1
        self.sent = []                # Marcas temporales de peticiones en curso

    def wait(self, n=1):          # Espera activa
        """Bloquea la ejecución hasta asegurar cupo en la ventana de 60 segundos."""   # Docstring
        n = min(n, self.rpm)      # Evita loops infinitos si el lote supera la cuota máxima
        while True:               # Bucle infinito controlado
            now = time.time()     # Hora actual
            self.sent = [t for t in self.sent if now - t < 60]   # Purga marcas antiguas (> 60s)
            if len(self.sent) + n <= self.rpm:   # ¿Hay espacio libre para n peticiones?
                break             # Sí, salimos
            time.sleep(max(0.2, 60 - (now - self.sent[0])))   # Espera inteligente al próximo hueco
        self.sent.extend([time.time()] * n)   # Añade marcas del lote actual

def server_retry_delay(message, default=2.0):   # Extrae delay mandado por Google
    """Busca expresiones de tiempo recomendadas de espera en el cuerpo del error."""   # Docstring
    match = re.search(r"(?:retry in|after) ([0-9.]+)\s?s", message, re.IGNORECASE)   # Regex robusta para capturar segundos
    if match:
        return float(match.group(1)) + 1.0  # Añade 1s extra de colchón para evitar el límite exacto
    return default   # Retorna el default si no hay matches

_limiter = RateLimiter(EMBEDDING_RPM)   # Instancia del RateLimiter

def get_embeddings_with_retry(client, texts, max_retries=6, initial_delay=3.0):   # Firma robusta
    """Llama a la API de embeddings con backoff exponencial, jitter y parseo de cuota 429."""   # Docstring
    delay = initial_delay          # Delay base de reintento
    for attempt in range(max_retries):   # Intentos sucesivos
        _limiter.wait(len(texts))        # Espera ventana RPM
        try:                             # Bloque con control de excepciones
            response = client.models.embed_content(   # API call
                model=EMBEDDING_MODEL,                # gemini-embedding-2
                contents=[
                    types.Content(parts=[types.Part.from_text(text=t)])
                    for t in texts
                ],
            )
            return [list(e.values) for e in response.embeddings]   # Retorno exitoso
        except Exception as e:            # Captura de errores de API
            msg = str(e)                  # Conversión a string
            is_429 = any(code in msg for code in ("429", "RESOURCE_EXHAUSTED"))   # Identifica 429
            is_server_error = any(code in msg for code in ("500", "503", "UNAVAILABLE", "INTERNAL"))   # Identifica 5xx
            
            if (is_429 or is_server_error) and attempt < max_retries - 1:   # Si aplica reintento...
                extracted_delay = server_retry_delay(msg, default=delay)   # Busca recomendación del server
                jitter = random.uniform(0.5, 1.5)   # Ruido aleatorio (jitter) para romper sincronizaciones concurrentes
                sleep_time = (extracted_delay * jitter)   # Tiempo final ponderado
                
                # Usamos toast en Streamlit para avisar al usuario si está corriendo interactivamente
                try:
                    st.toast(f"⏳ Límite de cuota alcanzado. Reintentando bloque en {sleep_time:.2f}s...", icon="⚠️")
                except Exception:
                    pass
                
                time.sleep(sleep_time)   # Pausa el thread
                delay *= 2.0             # Escala de backoff exponencial estándar
            else:
                raise                     # Falla definitiva si agotó intentos o es error no reintentable

def get_embedding_with_retry(client, text_chunk):   # Envuelve un único string
    """Compatibilidad con llamada de un solo chunk."""   # Docstring
    return get_embeddings_with_retry(client, [text_chunk])[0]   # Desempaqueta

# ----------------------------------------------------------------------------
# 7) LA "BASE DE DATOS VECTORIAL" en memoria
# ----------------------------------------------------------------------------

class SimpleVectorDB:   # Clase contenedora del RAG
    """Estructura vectorial liviana que persiste en memoria de sesión."""   # Docstring
    def __init__(self):            # Constructor
        self.chunks = []           # Almacena texto plano
        self.embeddings = []       # Almacena vectores float32
        self.metadata = []         # Metadata
        self.model = EMBEDDING_MODEL   # Modelo de embedding asociado

    def clear(self):               # Limpia completo
        self.chunks = []
        self.embeddings = []
        self.metadata = []

    def remove_document(self, source):   # Remoción interactiva
        """Elimina todos los bloques indexados asociados a un archivo fuente."""   # Docstring
        keep = [i for i, meta in enumerate(self.metadata) if meta.get("source") != source]
        removed = len(self.metadata) - len(keep)
        if removed:
            self.chunks = [self.chunks[i] for i in keep]
            self.embeddings = [self.embeddings[i] for i in keep]
            self.metadata = [self.metadata[i] for i in keep]
        return removed

    def add_document(self, client, file_name, file_content):   # Indexador robusto
        """Procesa, chunkifica e indexa un PDF completo de forma atómica."""   # Docstring
        text = extract_text_from_pdf(file_content)
        if not text.strip():
            raise ValueError("No se pudo extraer texto del PDF (¿es un escaneo sin capa OCR?)")

        doc_chunks = [c for c in chunk_text(text) if c.strip()]
        if not doc_chunks:
            raise ValueError("El PDF no produjo fragmentos válidos")

        new_chunks, new_vectors, new_meta = [], [], []
        for start in range(0, len(doc_chunks), EMBEDDING_BATCH_SIZE):
            batch = doc_chunks[start:start + EMBEDDING_BATCH_SIZE]
            # Formateamos documentos con prefijos para optimizar gemini-embedding-2
            formatted_batch = [format_document(c, file_name) for c in batch]
            vectors = get_embeddings_with_retry(client, formatted_batch)   # Petición con reintentos avanzados y tolerante a 429
            for offset, (chunk, vector) in enumerate(zip(batch, vectors)):
                new_chunks.append(chunk)
                new_vectors.append(vector)
                new_meta.append({"source": file_name, "chunk_id": start + offset})

        self.chunks.extend(new_chunks)
        self.embeddings.extend(new_vectors)
        self.metadata.extend(new_meta)
        return len(new_chunks)

    def search(self, client, query, top_k=3):   # Búsqueda coseno
        """Busca fragmentos relevantes calculando similitud de coseno."""   # Docstring
        if not self.embeddings:
            return []

        # Embed query
        query_vector = np.array(
            get_embeddings_with_retry(client, [format_query(query)])[0], dtype=np.float32
        )
        db_vectors = np.array(self.embeddings, dtype=np.float32)

        dot_product = np.dot(db_vectors, query_vector)
        norms_db = np.linalg.norm(db_vectors, axis=1)
        norm_query = np.linalg.norm(query_vector)

        norms_db[norms_db == 0] = 1e-9
        if norm_query == 0:
            norm_query = 1e-9

        similarities = dot_product / (norms_db * norm_query)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "text": self.chunks[idx],
                "metadata": self.metadata[idx],
                "score": float(similarities[idx])
            })
        return results

# ----------------------------------------------------------------------------
# 8) GENERACIÓN DE RESPUESTA
# ----------------------------------------------------------------------------

def preguntar_al_rag(pregunta, db, client, modelo_principal, top_k=3):   # Pipeline RAG
    """Pipeline RAG unificado: recupera fuentes y genera respuesta usando LLM."""   # Docstring
    resultados = db.search(client=client, query=pregunta, top_k=top_k)

    if not resultados:
        contexto = "No hay contexto disponible ya que no se han cargado documentos en la base de datos vectorial."
    else:
        contexto = "\n---\n".join([r['text'] for r in resultados])

    prompt = f"""Eres un asistente experto de soporte técnico y legal. Tu tarea es responder la pregunta del usuario utilizando únicamente el contexto proporcionado de forma precisa, clara y detallada.

Instrucciones estrictas:
1. Utiliza única y exclusivamente la información contenida en el Contexto.
2. Si la respuesta a la pregunta no se puede deducir o no está presente en el Contexto, responde amablemente indicando que no posees suficiente información en los documentos cargados para responder con certeza.
3. No intentes inventar, extrapolar o alucinar información fuera del Contexto proporcionado.

Contexto:
{contexto}

Pregunta: {pregunta}
Respuesta:"""

    try:
        response = client.models.generate_content(
            model=modelo_principal,
            contents=prompt
        )
        return response.text, resultados
    except Exception as e:
        raise e

# ----------------------------------------------------------------------------
# 9) CONFIGURACIÓN DE PÁGINA (Streamlit)
# ----------------------------------------------------------------------------

st.set_page_config(page_title="RAG PDF Expert Chat", page_icon="💬", layout="wide")   # Configura pestaña
st.title("💬 RAG Interactivo con Gemini & SimpleVectorDB")   # Título app
st.write("Consulte sus documentos PDF de manera inteligente utilizando procesamiento semántico en tiempo real.")   # Subtítulo

# ----------------------------------------------------------------------------
# 10) ESTADO DE LA SESIÓN
# ----------------------------------------------------------------------------

if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = SimpleVectorDB()
elif getattr(st.session_state.vector_db, "model", None) != EMBEDDING_MODEL:
    st.session_state.vector_db = SimpleVectorDB()
    st.session_state.processed_files = set()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'ignored_files' not in st.session_state:
    st.session_state.ignored_files = set()
if 'uploader_names' not in st.session_state:
    st.session_state.uploader_names = None

# ----------------------------------------------------------------------------
# 11) BARRA LATERAL (sidebar)
# ----------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuración del Sistema")

    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        st.success("🔒 Clave cargada desde la variable de entorno `GEMINI_API_KEY`.")
        gemini_key = env_key
        if st.checkbox("Usar otra clave manualmente", value=False):
            gemini_key = st.text_input(
                "Clave de API de Gemini",
                type="password",
                value="",
                placeholder="Pega aquí tu clave",
            )
    else:
        gemini_key = st.text_input(
            "Clave de API de Gemini",
            type="password",
            value="",
            placeholder="Pega aquí tu clave",
        )

    # Listado de modelos soportados
    model_options = [
        'gemini-2.5-flash',       # Modelo predeterminado
        'gemini-2.5-pro',
        'gemini-1.5-flash',
        'gemini-1.5-pro'
    ]
    selected_model = st.selectbox("Modelo de Generación", model_options, index=0)

    st.caption(f"🔎 Modelo de embeddings: `{EMBEDDING_MODEL}`")
    if st.button("🗑️ Vaciar índice de documentos", use_container_width=True):
        st.session_state.vector_db = SimpleVectorDB()
        st.session_state.processed_files = set()
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.subheader("📂 Cargador de Documentos")
    
    uploader_extra = {}
    if "max_upload_size" in inspect.signature(st.file_uploader).parameters:
        uploader_extra["max_upload_size"] = MAX_PDF_MB
        
    uploaded_files = st.file_uploader(
        "Subir archivos PDF",
        type=['pdf'],
        accept_multiple_files=True,
        help=f"Tamaño máximo: {MAX_PDF_MB} MB.",
        **uploader_extra
    )

    if uploaded_files:
        total_mb = sum(getattr(f, "size", 0) for f in uploaded_files) / (1024 * 1024)
        st.caption(f"📎 {len(uploaded_files)} archivo(s) · {total_mb:.1f} MB en total")

    sync_with_uploader = st.toggle(
        "Al quitar un archivo, borrar del RAG",
        value=True
    )

# ----------------------------------------------------------------------------
# 12) INICIALIZACIÓN CLIENTE
# ----------------------------------------------------------------------------

client = None
if gemini_key:
    try:
        client = genai.Client(api_key=gemini_key)
    except Exception as e:
        st.error(f"❌ No se pudo crear el cliente: {e}")
else:
    st.warning("⚠️ GEMINI_API_KEY no configurada en la barra lateral.")

# ----------------------------------------------------------------------------
# 13) SINCRONIZACIÓN
# ----------------------------------------------------------------------------

current_names = {f.name for f in uploaded_files} if uploaded_files else set()
if st.session_state.uploader_names is None:
    st.session_state.uploader_names = current_names
else:
    gone = st.session_state.uploader_names - current_names
    for name in sorted(gone):
        st.session_state.ignored_files.discard(name)
        if sync_with_uploader and name in st.session_state.processed_files:
            n = st.session_state.vector_db.remove_document(name)
            st.session_state.processed_files.discard(name)
            st.toast(f"🗑️ {name} eliminado del RAG ({n} fragmentos)", icon="🗑_")
    st.session_state.uploader_names = current_names

# ----------------------------------------------------------------------------
# 14) INDEXACIÓN CON CONTROL DE EXCEPCIONES Y 429
# ----------------------------------------------------------------------------

if client and uploaded_files:
    for uploaded_file in uploaded_files:
        if (uploaded_file.name not in st.session_state.processed_files 
                and uploaded_file.name not in st.session_state.ignored_files):
            size_mb = getattr(uploaded_file, "size", 0) / (1024 * 1024)
            if size_mb > MAX_PDF_MB:
                with st.sidebar:
                    st.error(f"❌ {uploaded_file.name} supera {MAX_PDF_MB} MB.")
                continue
            with st.sidebar:
                with st.spinner(f"Procesando {uploaded_file.name}... "):
                    try:
                        n_chunks = st.session_state.vector_db.add_document(
                            client=client,
                            file_name=uploaded_file.name,
                            file_content=uploaded_file
                        )
                        st.session_state.processed_files.add(uploaded_file.name)
                        st.success(f"✅ {uploaded_file.name} indexado.")
                    except Exception as e:
                        msg = str(e)
                        st.error(f"❌ Error al indexar {uploaded_file.name}: {msg}")
                        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                            st.info("⚠️ El RAG se pausará brevemente debido a límites de cuota (429) en el Free Tier de Gemini. La lógica de control intentará continuar con mayor espera.")

# ----------------------------------------------------------------------------
# 15) REVISAR ÍNDICE DE DOCUMENTOS
# ----------------------------------------------------------------------------

with st.sidebar:
    st.divider()
    st.subheader("Documentos Indexados")
    if st.session_state.processed_files:
        for doc in sorted(st.session_state.processed_files):
            col_name, col_del = st.columns([5, 1])
            col_name.write(f"📄 {doc}")
            if col_del.button("🗑️", key=f"del_{doc}"):
                n = st.session_state.vector_db.remove_document(doc)
                st.session_state.processed_files.discard(doc)
                st.session_state.ignored_files.add(doc)
                st.toast(f"🗑️ {doc} eliminado", icon="🗑️")
                st.rerun()
        st.caption(f"{len(st.session_state.vector_db.chunks)} fragmentos en memoria")
    else:
        st.info("No hay documentos indexados en el sistema.")

# ----------------------------------------------------------------------------
# 16) RENDERIZADO DEL CHAT
# ----------------------------------------------------------------------------

for chat in st.session_state.chat_history:
    with st.chat_message(chat['role']):
        st.markdown(chat['content'])
        if chat['role'] == 'assistant' and 'sources' in chat:
            with st.expander("🔍 Ver Fragmentos Recuperados (Fuentes)"):
                for i, source in enumerate(chat['sources']):
                    st.markdown(f"**Fragmento {i+1} - {source['metadata']['source']} (Similitud: {source['score']:.4f})**")
                    st.code(source['text'])

# ----------------------------------------------------------------------------
# 17) PROCESAR ENTRADA DEL CHAT
# ----------------------------------------------------------------------------

if user_query := st.chat_input("Escriba su consulta basándose en los documentos..."):
    st.chat_message("user").markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    if not client:
        st.error("❌ API Key de Gemini requerida para responder.")
    elif len(st.session_state.vector_db.chunks) == 0:
        with st.chat_message("assistant"):
            err_msg = "⚠️ No se han cargado documentos en el sistema. Por favor suba archivos PDF."
            st.markdown(err_msg)
            st.session_state.chat_history.append({"role": "assistant", "content": err_msg})
    else:
        with st.chat_message("assistant"):
            with st.spinner("Buscando respuestas..."):
                try:
                    response_text, sources = preguntar_al_rag(
                        pregunta=user_query,
                        db=st.session_state.vector_db,
                        client=client,
                        modelo_principal=selected_model
                    )
                    st.markdown(response_text)
                    with st.expander("🔍 Ver Fragmentos Recuperados"):
                        for i, source in enumerate(sources):
                            st.markdown(f"**Fragmento {i+1} - {source['metadata']['source']} (Similitud: {source['score']:.4f})**")
                            st.code(source['text'])

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response_text,
                        "sources": sources
                    })
                except Exception as e:
                    st.error(f"Ocurrió un error al procesar la solicitud: {e}")

# =============================================================================
#  FIN DEL ARCHIVO
# =============================================================================

import streamlit as st
import numpy as np
from pypdf import PdfReader
import time
import os
from google import genai
from google.genai import types
from google.genai.errors import APIError

# --- RAG PIPELINE UTILITIES ---

# Modelo de embeddings vigente (noviembre 2026):
#   - "text-embedding-004"          -> retirado (404 NOT_FOUND)
#   - "gemini-embedding-2-preview"  -> retirado el 10-ago-2026 (404 NOT_FOUND)
#   - "gemini-embedding-001"        -> sigue disponible, pero es SOLO texto (2048 tokens)
#   - "gemini-embedding-2"          -> estable y recomendado (multimodal, 8192 tokens)
EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_BATCH_SIZE = 100  # límite de :batchEmbedContents

def extract_text_from_pdf(pdf_file):
    """Extracts text from a file-like object or file path using pypdf."""
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def chunk_text(text, chunk_size=1000, overlap=100):
    """Splits text into chunks of chunk_size with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

def format_query(text):
    """Prefijo de tarea recomendado por Google para consultas (gemini-embedding-2)."""
    return f"task: search result | query: {text}"

def format_document(text, title=None):
    """Prefijo de tarea recomendado por Google para documentos indexados."""
    return f"title: {title or 'none'} | text: {text}"

def get_embeddings_with_retry(client, texts, max_retries=5, initial_delay=2):
    """Devuelve UN embedding por cada texto (no un embedding agregado), con backoff exponencial.

    Nota: con gemini-embedding-2 hay que envolver cada texto en su propio types.Content;
    si se pasa una lista de strings "sueltos", el modelo devuelve un único vector agregado.
    """
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=[
                    types.Content(parts=[types.Part.from_text(text=t)])
                    for t in texts
                ],
            )
            return [list(e.values) for e in response.embeddings]
        except Exception as e:
            msg = str(e)
            retriable = any(
                code in msg
                for code in ("429", "RESOURCE_EXHAUSTED", "500", "503",
                             "UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED")
            )
            if retriable and attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                raise

def get_embedding_with_retry(client, text_chunk):
    """Compatibilidad: embedding de un único texto."""
    return get_embeddings_with_retry(client, [text_chunk])[0]

class SimpleVectorDB:
    """A simple, lightweight vector database suitable for memory storage inside Streamlit's state."""
    def __init__(self):
        self.chunks = []
        self.embeddings = []
        self.metadata = []
        self.model = EMBEDDING_MODEL

    def clear(self):
        self.chunks = []
        self.embeddings = []
        self.metadata = []

    def remove_document(self, source):
        """Elimina del índice todos los fragmentos de un documento. Devuelve cuántos quitó."""
        keep = [i for i, meta in enumerate(self.metadata) if meta.get("source") != source]
        removed = len(self.metadata) - len(keep)
        if removed:
            self.chunks = [self.chunks[i] for i in keep]
            self.embeddings = [self.embeddings[i] for i in keep]
            self.metadata = [self.metadata[i] for i in keep]
        return removed

    def add_document(self, client, file_name, file_content):
        """Extracts, chunks, and indexes a PDF document into the database.

        Indexa en lotes (mucho más rápido y con menos cuota que 1 llamada por chunk).
        Lanza excepción si no se pudo indexar nada, para no marcar el PDF como procesado.
        """
        text = extract_text_from_pdf(file_content)
        if not text.strip():
            raise ValueError(
                "no se pudo extraer texto del PDF (¿es un escaneo sin capa de texto/OCR?)"
            )

        doc_chunks = [c for c in chunk_text(text) if c.strip()]
        if not doc_chunks:
            raise ValueError("el PDF no produjo fragmentos de texto utilizables")

        new_chunks, new_vectors, new_meta = [], [], []
        for start in range(0, len(doc_chunks), EMBEDDING_BATCH_SIZE):
            batch = doc_chunks[start:start + EMBEDDING_BATCH_SIZE]
            vectors = get_embeddings_with_retry(
                client, [format_document(c, file_name) for c in batch]
            )
            for offset, (chunk, vector) in enumerate(zip(batch, vectors)):
                new_chunks.append(chunk)
                new_vectors.append(vector)
                new_meta.append({"source": file_name, "chunk_id": start + offset})

        self.chunks.extend(new_chunks)
        self.embeddings.extend(new_vectors)
        self.metadata.extend(new_meta)
        return len(new_chunks)

    def search(self, client, query, top_k=3):
        """Performs cosine similarity search over stored vectors."""
        if not self.embeddings:
            return []

        query_vector = np.array(
            get_embeddings_with_retry(client, [format_query(query)])[0], dtype=np.float32
        )
        db_vectors = np.array(self.embeddings, dtype=np.float32)

        dot_product = np.dot(db_vectors, query_vector)
        norms_db = np.linalg.norm(db_vectors, axis=1)
        norm_query = np.linalg.norm(query_vector)

        # Prevent division by zero
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

def preguntar_al_rag(pregunta, db, client, modelo_principal, top_k=3):
    """Queries the RAG pipeline using context retrieval and Gemini generation."""
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

# --- STREAMLIT UI CONFIGURATION ---
st.set_page_config(page_title="RAG PDF Expert Chat", page_icon="💬", layout="wide")

st.title("💬 RAG Interactivo con Gemini & SimpleVectorDB")
st.write("Consulte sus documentos PDF de manera inteligente utilizando procesamiento semántico en tiempo real.")

# --- SESSION STATE INITIALIZATION ---
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = SimpleVectorDB()
elif getattr(st.session_state.vector_db, "model", None) != EMBEDDING_MODEL:
    # Índice creado con otro modelo de embeddings: los vectores no son comparables.
    st.session_state.vector_db = SimpleVectorDB()
    st.session_state.processed_files = set()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'ignored_files' not in st.session_state:
    # Archivos que el usuario quitó del índice pero que siguen en el cargador.
    st.session_state.ignored_files = set()
if 'uploader_names' not in st.session_state:
    # Última lista de nombres vista en el file_uploader (para detectar borrados).
    st.session_state.uploader_names = None

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Configuración del Sistema")
    
    # Retrieve Gemini API Key safely
    gemini_key = st.text_input(
        "Clave de API de Gemini",
        type="password",
        value=os.environ.get('GEMINI_API_KEY', ''),
        help="Ingrese su clave API de Gemini. Si ya está configurada en las variables de entorno, se usará automáticamente."
    )
    
    # Select Generation Model
    # Nota: gemini-2.5-* aparece en models.list() pero devuelve 404 en generateContent.
    model_options = [
        'gemini-3.1-flash-lite',   # probado OK, rápido
        'gemini-3.8-flash',
        'gemini-flash-latest',
        'gemini-3.5-flash',
        'gemini-pro-latest',       # suele requerir facturación (429 en capa gratuita)
    ]
    selected_model = st.selectbox("Modelo de Generación", model_options, index=0)
    custom_model = st.text_input(
        "…o escribe otro modelo (opcional)",
        value="",
        placeholder="ej. gemini-3.8-flash",
    )
    if custom_model.strip():
        selected_model = custom_model.strip()

    st.caption(f"🔎 Modelo de embeddings: `{EMBEDDING_MODEL}`")
    if st.button("🗑️ Vaciar índice de documentos", use_container_width=True):
        st.session_state.vector_db = SimpleVectorDB()
        st.session_state.processed_files = set()
        st.session_state.chat_history = []
        st.rerun()
    
    st.divider()
    
    st.subheader("📂 Cargador de Documentos")
    uploaded_files = st.file_uploader(
        "Subir archivos PDF",
        type=['pdf'],
        accept_multiple_files=True
    )
    sync_with_uploader = st.toggle(
        "Al quitar un archivo del cargador, borrarlo del RAG",
        value=True,
        help="Si lo desactivas, el documento seguirá indexado aunque lo quites del cargador "
             "(útil si borraste el archivo por error o quieres seguir consultándolo)."
    )

# Initialize Google GenAI client if key is provided
client = None
if gemini_key:
    try:
        client = genai.Client(api_key=gemini_key)
    except Exception as e:
        st.error(f"❌ No se pudo crear el cliente de Gemini: {e}")
else:
    st.warning("⚠️ Por favor, ingrese su GEMINI_API_KEY en la barra lateral para comenzar.")

# --- SYNC: archivos quitados del cargador -> fuera del índice ---
current_names = {f.name for f in uploaded_files} if uploaded_files else set()
if st.session_state.uploader_names is None:
    # Primera ejecución de la sesión: solo memorizamos, no borramos nada.
    st.session_state.uploader_names = current_names
else:
    gone = st.session_state.uploader_names - current_names
    for name in sorted(gone):
        # Ya no está en el cargador: dejamos de ignorarlo
        st.session_state.ignored_files.discard(name)
        if sync_with_uploader and name in st.session_state.processed_files:
            n = st.session_state.vector_db.remove_document(name)
            st.session_state.processed_files.discard(name)
            st.toast(f"🗑️ {name} eliminado del RAG ({n} fragmentos)", icon="🗑️")
    st.session_state.uploader_names = current_names

# --- PDF PROCESSING LOGIC ---
if client and uploaded_files:
    for uploaded_file in uploaded_files:
        if (uploaded_file.name not in st.session_state.processed_files
                and uploaded_file.name not in st.session_state.ignored_files):
            with st.sidebar:
                with st.spinner(f"Procesando {uploaded_file.name}..."):
                    try:
                        n_chunks = st.session_state.vector_db.add_document(
                            client=client,
                            file_name=uploaded_file.name,
                            file_content=uploaded_file
                        )
                        st.session_state.processed_files.add(uploaded_file.name)
                        st.success(f"✅ {uploaded_file.name} indexado ({n_chunks} fragmentos).")
                    except Exception as e:
                        msg = str(e)
                        st.error(f"❌ No se pudo indexar {uploaded_file.name}: {msg}")
                        if "NOT_FOUND" in msg or "not found" in msg:
                            st.info(
                                "Ese modelo de embeddings ya no existe o no está disponible para tu "
                                f"clave/región. Modelo en uso: `{EMBEDDING_MODEL}`. "
                                "Prueba `gemini-embedding-001` o revisa que la API de embeddings "
                                "esté habilitada para tu proyecto."
                            )

# Show loaded documents status
with st.sidebar:
    st.divider()
    st.subheader("Documentos Indexados")
    if st.session_state.processed_files:
        for doc in sorted(st.session_state.processed_files):
            col_name, col_del = st.columns([5, 1])
            col_name.write(f"📄 {doc}")
            if col_del.button(
                "🗑️",
                key=f"del_{doc}",
                help=f"Quitar «{doc}» del índice vectorial",
            ):
                n = st.session_state.vector_db.remove_document(doc)
                st.session_state.processed_files.discard(doc)
                # Evita que se re-indexe mientras siga en el cargador
                st.session_state.ignored_files.add(doc)
                st.toast(f"🗑️ {doc} eliminado del RAG ({n} fragmentos)", icon="🗑️")
                st.rerun()
        st.caption(f"{len(st.session_state.vector_db.chunks)} fragmentos en memoria")
        if st.session_state.ignored_files:
            st.caption(
                "ℹ️ Ignorados (siguen en el cargador, fuera del RAG): "
                + ", ".join(sorted(st.session_state.ignored_files))
            )
            if st.button("🔄 Volver a indexar ignorados", use_container_width=True):
                st.session_state.ignored_files = set()
                st.rerun()
    else:
        st.info("No hay documentos indexados en el sistema.")
        if st.session_state.ignored_files:
            st.caption(
                "ℹ️ Ignorados: " + ", ".join(sorted(st.session_state.ignored_files))
            )
            if st.button("🔄 Volver a indexar ignorados", use_container_width=True):
                st.session_state.ignored_files = set()
                st.rerun()

# --- MAIN INTERACTIVE CHAT AREA ---
# Render chat history
for chat in st.session_state.chat_history:
    with st.chat_message(chat['role']):
        st.markdown(chat['content'])
        if chat['role'] == 'assistant' and 'sources' in chat:
            with st.expander("🔍 Ver Fragmentos Recuperados (Fuentes)"):
                for i, source in enumerate(chat['sources']):
                    st.markdown(f"**Fragmento {i+1} - {source['metadata']['source']} (Similitud: {source['score']:.4f})**")
                    st.code(source['text'])

# Handle new queries
if user_query := st.chat_input("Escriba su consulta basándose en los documentos..."):
    # Display user query
    st.chat_message("user").markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    
    if not client:
        st.error("❌ API Key de Gemini requerida para responder.")
    elif len(st.session_state.vector_db.chunks) == 0:
        with st.chat_message("assistant"):
            err_msg = "⚠️ No se han cargado documentos en el sistema. Por favor suba archivos PDF para habilitar la búsqueda semántica."
            st.markdown(err_msg)
            st.session_state.chat_history.append({"role": "assistant", "content": err_msg})
    else:
        with st.chat_message("assistant"):
            with st.spinner("Buscando respuestas en los documentos..."):
                try:
                    response_text, sources = preguntar_al_rag(
                        pregunta=user_query,
                        db=st.session_state.vector_db,
                        client=client,
                        modelo_principal=selected_model
                    )
                    st.markdown(response_text)
                    with st.expander("🔍 Ver Fragmentos Recuperados (Fuentes)"):
                        for i, source in enumerate(sources):
                            st.markdown(f"**Fragmento {i+1} - {source['metadata']['source']} (Similitud: {source['score']:.4f})**")
                            st.code(source['text'])
                    
                    # Save to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response_text,
                        "sources": sources
                    })
                except Exception as e:
                    st.error(f"Ocurrió un error al procesar la solicitud: {e}")
                    if "NOT_FOUND" in str(e):
                        st.info(
                            "Suele ser un nombre de modelo inválido o retirado. "
                            f"Embeddings: `{EMBEDDING_MODEL}` · Generación: `{selected_model}`."
                        )

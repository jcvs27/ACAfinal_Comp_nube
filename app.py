import streamlit as st
import numpy as np
from pypdf import PdfReader
import time
import os
from google import genai
from google.genai.errors import APIError

# --- RAG PIPELINE UTILITIES ---

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

def get_embedding_with_retry(client, text_chunk, model='gemini-embedding-2-preview', max_retries=5, initial_delay=2):
    """Obtains the vector embedding of a text chunk with Exponential Backoff."""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            response = client.models.embed_content(
                model=model,
                contents=text_chunk
            )
            time.sleep(0.5)
            return response.embeddings[0].values
        except Exception as e:
            is_quota_error = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
            if is_quota_error and attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                raise e

class SimpleVectorDB:
    """A simple, lightweight vector database suitable for memory storage inside Streamlit's state."""
    def __init__(self):
        self.chunks = []
        self.embeddings = []
        self.metadata = []

    def add_document(self, client, file_name, file_content):
        """Extracts, chunks, and indexes a PDF document into the database."""
        text = extract_text_from_pdf(file_content)
        doc_chunks = chunk_text(text)

        for i, chunk in enumerate(doc_chunks):
            try:
                vector = get_embedding_with_retry(client, chunk)
                self.chunks.append(chunk)
                self.embeddings.append(vector)
                self.metadata.append({"source": file_name, "chunk_id": i})
            except Exception as e:
                st.error(f"❌ Error processing chunk {i} of {file_name}: {e}")

    def search(self, client, query, top_k=3):
        """Performs cosine similarity search over stored vectors."""
        if not self.embeddings:
            return []

        query_vector = np.array(get_embedding_with_retry(client, query))
        db_vectors = np.array(self.embeddings)

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
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = SimpleVectorDB()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()

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
    model_options = ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-1.5-flash', 'gemini-1.5-pro']
    selected_model = st.selectbox("Modelo de Generación", model_options, index=0)
    
    st.divider()
    
    st.subheader("📂 Cargador de Documentos")
    uploaded_files = st.file_uploader(
        "Subir archivos PDF",
        type=['pdf'],
        accept_multiple_files=True
    )

# Initialize Google GenAI client if key is provided
client = None
if gemini_key:
    client = genai.Client(api_key=gemini_key)
else:
    st.warning("⚠️ Por favor, ingrese su GEMINI_API_KEY en la barra lateral para comenzar.")

# --- PDF PROCESSING LOGIC ---
if client and uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.processed_files:
            with st.sidebar:
                with st.spinner(f"Procesando {uploaded_file.name}..."):
                    st.session_state.vector_db.add_document(
                        client=client,
                        file_name=uploaded_file.name,
                        file_content=uploaded_file
                    )
                    st.session_state.processed_files.add(uploaded_file.name)
                    st.success(f"✅ {uploaded_file.name} cargado con éxito!")

# Show loaded documents status
with st.sidebar:
    st.divider()
    st.subheader("Documentos Indexados")
    if st.session_state.processed_files:
        for doc in st.session_state.processed_files:
            st.write(f"📄 {doc}")
    else:
        st.info("No hay documentos indexados en el sistema.")

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

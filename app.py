# =============================================================================
#  app.py  —  RAG PDF Expert Chat
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
from google import genai          # SDK oficial de Google Gen AI (Gemini): cliente para embeddings y chat
from google.genai import types    # Tipos auxiliares (Content, Part) para construir las peticiones al modelo
from google.genai.errors import APIError  # Excepción del SDK; se importa aunque el código no la use directamente

# ----------------------------------------------------------------------------
# 2) CONSTANTES DE CONFIGURACIÓN
# ----------------------------------------------------------------------------

# --- UTILIDADES DEL PIPELINE RAG ---   # Marcador de sección: aquí empiezan las funciones del motor RAG
# (línea en blanco: solo separa bloques)
# Modelo de embeddings vigente (noviembre 2026):   # Comentario informativo: contexto de por qué este modelo
#   - "text-embedding-004"          -> retirado (404 NOT_FOUND)          # Modelo antiguo: la API responde 404
#   - "gemini-embedding-2-preview"  -> retirado el 10-ago-2026 (404 NOT_FOUND)  # La versión "preview" también murió
#   - "gemini-embedding-001"        -> sigue disponible, pero es SOLO texto (2048 tokens)  # Alternativa válida
#   - "gemini-embedding-2"          -> estable y recomendado (multimodal, 8192 tokens)     # El que usamos
EMBEDDING_MODEL = "gemini-embedding-2"   # Nombre del modelo que convierte texto → vector; se usa en TODO el programa
EMBEDDING_BATCH_SIZE = 20         # Textos por petición: lotes pequeños para no agotar la cuota por minuto

# Cuota del nivel gratuito de Gemini para embeddings: peticiones por minuto.   # Dato de la documentación
# Cada texto del lote cuenta como una petición, así que hay que espaciarlas.   # Por eso no se manda todo de golpe
EMBEDDING_RPM = 90                # 90 < 100: margen de seguridad sobre el límite real del Free Tier

# Tamaño máximo por archivo que se acepta en el cargador (en MB).   # Límite parametrizable
# Debe coincidir con [server] maxUploadSize de .streamlit/config.toml.   # Se valida en dos niveles
MAX_PDF_MB = 20                   # 20 MB: valor por defecto de Streamlit y de la API inline

# ----------------------------------------------------------------------------
# 3) EXTRACCIÓN DE TEXTO DEL PDF
# ----------------------------------------------------------------------------

def extract_text_from_pdf(pdf_file):   # Define la función; recibe un archivo PDF (ruta o file-like de Streamlit)
    """Extrae el texto de un PDF, ya sea una ruta de archivo o un archivo abierto (file-like) de Streamlit."""   # Docstring: documenta qué hace la función
    reader = PdfReader(pdf_file)   # Abre el PDF con pypdf; no carga todo el texto, solo la estructura
    text = ""                      # Acumulador: empieza vacío y va creciendo página a página
    for page in reader.pages:      # Recorre cada página del documento en orden
        page_text = page.extract_text()   # Extrae el texto de esa página (puede devolver None si es una imagen)
        if page_text:                     # Si la página tenía texto (no None y no vacío)...
            text += page_text + "\n"      # ...lo añade al acumulador con un salto de línea de separación
    return text                    # Devuelve TODO el texto del PDF como un único string

# ----------------------------------------------------------------------------
# 4) TROCEADO (CHUNKING) DEL TEXTO
# ----------------------------------------------------------------------------

def chunk_text(text, chunk_size=1000, overlap=100):   # Parte el texto en trozos; 1000 caracteres con 100 de solape
    """Parte el texto en fragmentos de 'chunk_size' caracteres, con solape de 'overlap' entre ellos."""   # Docstring de la función
    chunks = []                    # Lista vacía donde se irán guardando los fragmentos
    start = 0                      # Posición inicial del primer fragmento dentro del texto
    while start < len(text):       # Mientras no hayamos llegado al final del texto...
        end = start + chunk_size   # ...calcula dónde termina este fragmento (inicio + tamaño)
        chunks.append(text[start:end])   # Recorta el texto [start:end] y lo guarda como un fragmento
        start += (chunk_size - overlap)  # Avanza 900 posiciones (1000-100), así el siguiente fragmento solapa 100
    return chunks                  # Devuelve la lista completa de fragmentos

# ----------------------------------------------------------------------------
# 5) FORMATO DE TEXTO PARA EL MODELO (prefijos de tarea)
#    gemini-embedding-2 ya NO acepta el parámetro task_type: hay que poner la
#    tarea escrita dentro del propio texto, con este formato exacto.
# ----------------------------------------------------------------------------

def format_query(text):   # Prepara una PREGUNTA del usuario antes de convertirla en vector
    """Prefijo de tarea recomendado por Google para consultas (gemini-embedding-2)."""   # Docstring
    return f"task: search result | query: {text}"   # Añade el prefijo "búsqueda" delante de la pregunta

def format_document(text, title=None):   # Prepara un FRAGMENTO de documento antes de convertirlo en vector
    """Prefijo de tarea recomendado por Google para documentos indexados."""   # Docstring
    return f"title: {title or 'none'} | text: {text}"   # Prefijo con el título (o "none" si no hay) + el texto

# ----------------------------------------------------------------------------
# 6) EMBEDDINGS CON REINTENTOS (la parte que habla con la API de Google)
# ----------------------------------------------------------------------------

class RateLimiter:                # Controla cuántas peticiones se envían por minuto
    """Ventana deslizante de 60 s: evita superar las peticiones por minuto permitidas."""   # Docstring

    def __init__(self, rpm):      # Constructor: recibe el límite por minuto
        self.rpm = max(1, int(rpm))   # Límite por minuto (nunca menos de 1)
        self.sent = []                # Marcas de tiempo de las peticiones ya enviadas

    def wait(self, n=1):          # Espera lo necesario antes de enviar `n` peticiones
        """Espera lo necesario para que enviar `n` peticiones más no supere el límite."""   # Docstring
        n = min(n, self.rpm)      # Un lote mayor que el límite se recorta (evita esperar para siempre)
        while True:               # Repite hasta que haya hueco en la ventana de 60 s
            now = time.time()     # Hora actual
            self.sent = [t for t in self.sent if now - t < 60]   # Descarta lo enviado hace más de 60 s
            if len(self.sent) + n <= self.rpm:   # ¿Caben las `n` nuevas peticiones?
                break             # Sí: salimos del bucle
            time.sleep(max(0.5, 60 - (now - self.sent[0])))   # No: espera a que caduque la más antigua
        self.sent.extend([time.time()] * n)   # Anota las peticiones que se van a enviar

def server_retry_delay(message, default=2.0):   # Lee el "retry in 1.49s" que devuelve Gemini
    """Lee el «Please retry in 1.49s» del error de Gemini y espera al menos eso."""   # Docstring
    match = re.search(r"retry in ([0-9.]+)s", message)   # Busca el número de segundos en el mensaje
    return float(match.group(1)) + 0.5 if match else default   # Devuelve ese tiempo (+0.5 s de margen) o el valor por defecto

# Instancia compartida: todas las llamadas a embeddings pasan por el mismo límite.   # Una sola ventana global
_limiter = RateLimiter(EMBEDDING_RPM)   # Limita a EMBEDDING_RPM peticiones por minuto

def get_embeddings_with_retry(client, texts, max_retries=5, initial_delay=2):   # Convierte una LISTA de textos en vectores
    """Devuelve UN embedding por cada texto (no un embedding agregado), con backoff exponencial.   # Inicio del docstring

    Nota: con gemini-embedding-2 hay que envolver cada texto en su propio types.Content;   # Explica el "porqué" del formato
    si se pasa una lista de strings "sueltos", el modelo devuelve un único vector agregado.   # Aviso importante: agregaría todo
    """   # Fin del docstring
    delay = initial_delay          # Espera inicial entre reintentos: 2 segundos
    for attempt in range(max_retries):   # Intenta hasta 5 veces (attempt = 0,1,2,3,4)
        # Respeta la cuota: espera si hace falta antes de enviar el lote.   # Comentario
        _limiter.wait(len(texts))        # Bloquea hasta que haya hueco en la ventana de 60 s
        try:                             # Bloque protegido: si la API falla, saltamos al except
            response = client.models.embed_content(   # Llama al endpoint de embeddings del SDK
                model=EMBEDDING_MODEL,                # Modelo a usar: "gemini-embedding-2"
                contents=[                            # Lista de contenidos: uno por cada texto
                    types.Content(parts=[types.Part.from_text(text=t)])   # Cada texto va en su PROPIO Content → vector separado
                    for t in texts                        # Comprensión de listas: repite lo anterior para cada texto
                ],                                        # Cierra la lista de contenidos
            )                                             # Cierra la llamada a embed_content
            return [list(e.values) for e in response.embeddings]   # Convierte cada respuesta a lista de floats y la devuelve
        except Exception as e:            # Si algo falló, capturamos el error
            msg = str(e)                  # Convertimos la excepción a texto para poder inspeccionarla
            retriable = any(              # ¿Es un error temporal (merece reintento)?
                code in msg               # Comprueba si el texto del error contiene el código...
                for code in ("429", "RESOURCE_EXHAUSTED", "500", "503",   # ...de cuota agotada o de servidor caído
                             "UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED")  # ...o de servicio no disponible
            )                             # Fin del any(): True si encontró alguno
            if retriable and attempt < max_retries - 1:   # Si es temporal Y aún quedan intentos...
                # Si el servidor dice cuánto esperar ("retry in 1.49s"), se respeta.   # Comentario
                time.sleep(max(delay, server_retry_delay(msg)))   # ...espera lo que pida el servidor o el backoff
                delay *= 2                # Duplica la espera para el siguiente intento (backoff exponencial)
            else:                         # Si NO es temporal o ya no quedan intentos...
                raise                     # ...relanza la excepción para que la maneje quien llamó

def get_embedding_with_retry(client, text_chunk):   # Envoltorio cómodo para UN solo texto
    """Compatibilidad: embedding de un único texto."""   # Docstring
    return get_embeddings_with_retry(client, [text_chunk])[0]   # Llama a la versión múltiple con 1 elemento y toma el primero

# ----------------------------------------------------------------------------
# 7) LA "BASE DE DATOS VECTORIAL" (en realidad, tres listas en memoria)
# ----------------------------------------------------------------------------

class SimpleVectorDB:   # Clase que agrupa fragmentos, vectores y metadatos
    """Base de datos vectorial simple y ligera, pensada para vivir en memoria dentro de st.session_state."""   # Docstring de la clase
    def __init__(self):            # Constructor: se ejecuta al crear el objeto
        self.chunks = []           # Lista de fragmentos de texto (strings)
        self.embeddings = []       # Lista de vectores; el índice i corresponde al chunk i
        self.metadata = []         # Lista de diccionarios {source, chunk_id} en el mismo orden
        self.model = EMBEDDING_MODEL   # Guarda con qué modelo se generaron; sirve para invalidar el índice

    def clear(self):               # Vacía la base de datos entera
        self.chunks = []           # Borra los textos
        self.embeddings = []       # Borra los vectores
        self.metadata = []         # Borra los metadatos

    def remove_document(self, source):   # Borra TODOS los fragmentos de un documento concreto
        """Elimina del índice todos los fragmentos de un documento. Devuelve cuántos quitó."""   # Docstring
        keep = [i for i, meta in enumerate(self.metadata) if meta.get("source") != source]   # Índices que SÍ se conservan
        removed = len(self.metadata) - len(keep)   # Cuántos se van a eliminar (total - conservados)
        if removed:                                # Si hay algo que borrar...
            self.chunks = [self.chunks[i] for i in keep]         # Reconstruye la lista de textos sin los borrados
            self.embeddings = [self.embeddings[i] for i in keep]  # Reconstruye los vectores en el MISMO orden
            self.metadata = [self.metadata[i] for i in keep]      # Reconstruye los metadatos igual
        return removed                             # Devuelve el número de fragmentos eliminados

    def add_document(self, client, file_name, file_content):   # Indexa un PDF completo
        """Extrae el texto, lo trocea y lo indexa en la base de datos.   # Inicio del docstring

        Indexa en lotes (mucho más rápido y con menos cuota que 1 llamada por chunk).   # Ventaja del batching
        Lanza excepción si no se pudo indexar nada, para no marcar el PDF como procesado.   # Contrato de errores
        """   # Fin del docstring
        text = extract_text_from_pdf(file_content)   # Paso 1: saca todo el texto del PDF
        if not text.strip():                         # Si el texto está vacío o solo son espacios...
            raise ValueError(                        # ...es un error: el PDF no tiene texto (probablemente escaneado)
                "no se pudo extraer texto del PDF (¿es un escaneo sin capa de texto/OCR?)"   # Mensaje explicativo
            )                                        # Cierra el raise

        doc_chunks = [c for c in chunk_text(text) if c.strip()]   # Paso 2: trocea y descarta fragmentos vacíos
        if not doc_chunks:                                        # Si no quedó ningún fragmento útil...
            raise ValueError("el PDF no produjo fragmentos de texto utilizables")   # ...error explícito

        new_chunks, new_vectors, new_meta = [], [], []   # Listas temporales: se llenan y al final se añaden todas juntas
        for start in range(0, len(doc_chunks), EMBEDDING_BATCH_SIZE):   # Recorre los chunks de 100 en 100
            batch = doc_chunks[start:start + EMBEDDING_BATCH_SIZE]      # Toma el lote actual (máx. 100 fragmentos)
            vectors = get_embeddings_with_retry(                        # Paso 3: convierte el lote a vectores
                client, [format_document(c, file_name) for c in batch]  # Cada chunk con su prefijo "title/text"
            )                                                           # Cierra la llamada
            for offset, (chunk, vector) in enumerate(zip(batch, vectors)):   # Recorre chunk y vector emparejados
                new_chunks.append(chunk)                        # Guarda el texto del fragmento
                new_vectors.append(vector)                      # Guarda su vector correspondiente
                new_meta.append({"source": file_name, "chunk_id": start + offset})   # Origen y número de fragmento

        self.chunks.extend(new_chunks)        # Añade todos los textos nuevos a la base de datos
        self.embeddings.extend(new_vectors)   # Añade todos los vectores nuevos
        self.metadata.extend(new_meta)        # Añade todos los metadatos nuevos
        return len(new_chunks)                # Devuelve cuántos fragmentos se indexaron

    def search(self, client, query, top_k=3):   # Busca los fragmentos más parecidos a la pregunta
        """Realiza una búsqueda por similitud coseno sobre los vectores almacenados."""   # Docstring
        if not self.embeddings:          # Si la base está vacía no hay nada que buscar...
            return []                    # ...devuelve lista vacía (evita errores más abajo)

        query_vector = np.array(         # Convierte la pregunta en un vector numpy...
            get_embeddings_with_retry(client, [format_query(query)])[0], dtype=np.float32   # ...pidiendo su embedding a la API
        )                                # float32 ahorra memoria y es suficiente precisión
        db_vectors = np.array(self.embeddings, dtype=np.float32)   # Matriz (n_fragmentos × dimensiones) con todos los vectores

        dot_product = np.dot(db_vectors, query_vector)   # Producto escalar de la pregunta con CADA fragmento (una fila)
        norms_db = np.linalg.norm(db_vectors, axis=1)    # Longitud (norma) de cada vector de la base; axis=1 = por fila
        norm_query = np.linalg.norm(query_vector)        # Longitud del vector de la pregunta

        # Evitar la división por cero si algún vector es nulo
        norms_db[norms_db == 0] = 1e-9   # Sustituye las normas nulas por un valor diminuto
        if norm_query == 0:              # Si la pregunta produjo un vector nulo...
            norm_query = 1e-9            # ...también lo sustituye para no romper la fórmula

        similarities = dot_product / (norms_db * norm_query)   # Fórmula del coseno: dot / (|a|·|b|) → valor entre -1 y 1
        top_indices = np.argsort(similarities)[::-1][:top_k]   # Ordena de mayor a menor y toma los top_k mejores

        results = []                     # Lista donde se armará la respuesta final
        for idx in top_indices:          # Para cada uno de los índices ganadores...
            results.append({             # ...crea un diccionario con toda la información
                "text": self.chunks[idx],            # El fragmento de texto original
                "metadata": self.metadata[idx],      # De qué archivo viene y su número
                "score": float(similarities[idx])    # La similitud (convertida a float normal, no numpy)
            })                           # Cierra el diccionario
        return results                   # Devuelve la lista de los mejores fragmentos

# ----------------------------------------------------------------------------
# 8) GENERACIÓN DE LA RESPUESTA (el "G" de RAG)
# ----------------------------------------------------------------------------

def preguntar_al_rag(pregunta, db, client, modelo_principal, top_k=3):   # Orquesta: recuperar + preguntar al LLM
    """Ejecuta el pipeline RAG: recupera el contexto y genera la respuesta con Gemini."""   # Docstring
    resultados = db.search(client=client, query=pregunta, top_k=top_k)   # Paso R: recupera los fragmentos relevantes

    if not resultados:                   # Si no se recuperó nada...
        contexto = "No hay contexto disponible ya que no se han cargado documentos en la base de datos vectorial."   # Texto de relleno
    else:                                # Si sí hay fragmentos...
        contexto = "\n---\n".join([r['text'] for r in resultados])   # Los une separados por "---" para el prompt

    # --- AQUÍ EMPIEZA EL PROMPT (texto entre triple comilla: NO se puede comentar por dentro) ---
    # f""" ... """ = cadena formateada: las {llaves} se sustituyen por variables.
    # El prompt define el rol ("asistente experto"), las reglas anti-alucinación
    # y luego inyecta el contexto recuperado y la pregunta del usuario.
    # El prompt tiene 3 partes: (1) rol del modelo, (2) reglas anti-alucinación,
    # (3) contexto recuperado + pregunta. IMPORTANTE: aquí no se pueden poner
    # comentarios dentro de las triple comillas, porque todo ese texto se envía
    # tal cual al modelo. Por eso la explicación va antes y después del bloque.
    prompt = f"""Eres un asistente experto de soporte técnico y legal. Tu tarea es responder la pregunta del usuario utilizando únicamente el contexto proporcionado de forma precisa, clara y detallada.

Instrucciones estrictas:
1. Utiliza única y exclusivamente la información contenida en el Contexto.
2. Si la respuesta a la pregunta no se puede deducir o no está presente en el Contexto, responde amablemente indicando que no posees suficiente información en los documentos cargados para responder con certeza.
3. No intentes inventar, extrapolar o alucinar información fuera del Contexto proporcionado.

Contexto:
{contexto}

Pregunta: {pregunta}
Respuesta:"""
    # ↑ Dentro del bloque anterior:
    #   · "Instrucciones estrictas" + 3 reglas → impiden que el modelo invente.
    #   · {contexto} → aquí se inyectan los fragmentos recuperados.
    #   · {pregunta} → aquí se inyecta lo que escribió el usuario.
    #   · "Respuesta:" al final empuja al modelo a continuar escribiendo.
    # --- FIN DEL PROMPT ---

    try:                                       # Protege la llamada por si la API falla
        response = client.models.generate_content(   # Pide al modelo de generación que responda
            model=modelo_principal,                  # Modelo elegido en la barra lateral
            contents=prompt                          # El prompt completo que acabamos de construir
        )                                            # Cierra la llamada
        return response.text, resultados             # Devuelve la respuesta Y las fuentes (para mostrarlas en la UI)
    except Exception as e:                         # Si hubo error...
        raise e                                    # ...lo relanza (la UI lo capturará y mostrará el mensaje)

# ----------------------------------------------------------------------------
# 9) CONFIGURACIÓN DE LA PÁGINA (Streamlit)
# ----------------------------------------------------------------------------

# --- CONFIGURACIÓN DE LA INTERFAZ (STREAMLIT) ---   # Marcador de sección: interfaz de usuario
st.set_page_config(page_title="RAG PDF Expert Chat", page_icon="💬", layout="wide")   # Título de pestaña, icono y ancho completo

st.title("💬 RAG Interactivo con Gemini & SimpleVectorDB")   # Título grande de la app
st.write("Consulte sus documentos PDF de manera inteligente utilizando procesamiento semántico en tiempo real.")   # Subtítulo descriptivo

# ----------------------------------------------------------------------------
# 10) ESTADO DE LA SESIÓN
#     Streamlit re-ejecuta TODO el script en cada interacción. Para que los
#     datos no se pierdan, se guardan en st.session_state (un diccionario
#     que sobrevive entre ejecuciones de la misma pestaña del navegador).
# ----------------------------------------------------------------------------

# --- INICIALIZACIÓN DEL ESTADO DE SESIÓN ---   # Marcador de sección
if 'processed_files' not in st.session_state:   # ¿Ya existe la clave 'processed_files'?
    st.session_state.processed_files = set()    # No: créala como conjunto vacío (nombres de PDFs ya indexados)
if 'vector_db' not in st.session_state:         # ¿Ya existe la base de datos vectorial?
    st.session_state.vector_db = SimpleVectorDB()   # No: créala
elif getattr(st.session_state.vector_db, "model", None) != EMBEDDING_MODEL:   # Sí existe, ¿pero con otro modelo?
    # Índice creado con otro modelo de embeddings: los vectores no son comparables.   # Explicación del caso
    st.session_state.vector_db = SimpleVectorDB()   # Lo descarta y crea uno nuevo
    st.session_state.processed_files = set()         # Y olvida qué archivos estaban indexados
if 'chat_history' not in st.session_state:          # ¿Existe el historial de chat?
    st.session_state.chat_history = []              # No: lista vacía (cada mensaje será un diccionario)
if 'ignored_files' not in st.session_state:         # ¿Existe el conjunto de archivos "ignorados"?
    # Archivos que el usuario quitó del índice pero que siguen en el cargador.   # Para qué sirve este conjunto
    st.session_state.ignored_files = set()          # No: créalo vacío
if 'uploader_names' not in st.session_state:        # ¿Existe el registro de lo que había en el cargador?
    # Última lista de nombres vista en el file_uploader (para detectar borrados).   # Utilidad: detectar la ✖ del usuario
    st.session_state.uploader_names = None          # None = "todavía no sé nada" (primera ejecución)

# ----------------------------------------------------------------------------
# 11) BARRA LATERAL (sidebar): clave, modelos, botones y cargador
# ----------------------------------------------------------------------------

# --- CONFIGURACIÓN DE LA BARRA LATERAL ---   # Marcador de sección
with st.sidebar:                 # Todo lo indentado debajo se dibuja en la barra lateral
    st.header("⚙️ Configuración del Sistema")   # Encabezado de la barra lateral

    # Obtener la clave de API de Gemini de forma segura   # No se escribe nunca dentro del código
    env_key = os.environ.get("GEMINI_API_KEY", "").strip()   # Lee la clave del entorno y quita espacios sobrantes
    if env_key:                                    # ¿Hay clave en la variable de entorno?
        # Si la clave viene del entorno NO se pasa como `value`: nunca viaja al navegador.   # Comentario de seguridad
        st.success("🔒 Clave cargada desde la variable de entorno `GEMINI_API_KEY` (no se muestra).")   # Aviso sin mostrar el secreto
        gemini_key = env_key                       # Usa esa clave para el cliente de Gemini
        if st.checkbox("Usar otra clave manualmente", value=False):   # Casilla opcional para escribir otra clave
            gemini_key = st.text_input(            # Campo manual de la clave
                "Clave de API de Gemini",          # Etiqueta del campo
                type="password",                   # type="password" → se muestran puntos en vez del texto
                value="",                          # Empieza vacío: no se precarga ningún secreto
                placeholder="Pega aquí tu clave",  # Texto de ejemplo dentro del campo
                help="El texto se oculta (puntos) mientras escribes.",   # Tooltip
            )                                      # Cierra el text_input
    else:                                          # Si NO hay clave en el entorno...
        gemini_key = st.text_input(                # ...se pide al usuario que la escriba
            "Clave de API de Gemini",              # Etiqueta del campo
            type="password",                       # Oculta el texto mientras se escribe
            value="",                              # Empieza vacío
            placeholder="Pega aquí tu clave",      # Texto de ejemplo dentro del campo
            help="El texto se oculta (puntos) mientras escribes. También puedes definir la variable de entorno GEMINI_API_KEY.",   # Tooltip
        )                                          # Cierra el text_input

    # Selección del modelo de generación (el que redacta las respuestas)
    # Nota: gemini-2.5-* aparece en models.list() pero devuelve 404 en generateContent.   # Aviso aprendido en pruebas
    model_options = [               # Lista de modelos ofrecidos en el desplegable
        'gemini-3.1-flash-lite',   # probado OK, rápido   # Opción por defecto
        'gemini-3.8-flash',        # Modelo flash más reciente
        'gemini-flash-latest',     # Alias que Google actualiza solo
        'gemini-3.5-flash',        # Otra versión flash
        'gemini-pro-latest',       # suele requerir facturación (429 en capa gratuita)   # Aviso de cuota
    ]                               # Cierra la lista
    selected_model = st.selectbox("Modelo de Generación", model_options, index=0)   # Desplegable; index=0 = el primero
    custom_model = st.text_input(   # Campo opcional para escribir un modelo no listado
        "…o escribe otro modelo (opcional)",   # Etiqueta
        value="",                              # Empieza vacío
        placeholder="ej. gemini-3.8-flash",    # Texto de ayuda dentro del campo
    )                                          # Cierra el text_input
    if custom_model.strip():                   # Si el usuario escribió algo (quitando espacios)...
        selected_model = custom_model.strip()  # ...ese nombre manda sobre el desplegable

    st.caption(f"🔎 Modelo de embeddings: `{EMBEDDING_MODEL}`")   # Texto pequeño informativo con el modelo de embeddings
    if st.button("🗑️ Vaciar índice de documentos", use_container_width=True):   # Botón que ocupa todo el ancho
        st.session_state.vector_db = SimpleVectorDB()    # Sustituye la base de datos por una vacía
        st.session_state.processed_files = set()         # Olvida los archivos procesados
        st.session_state.chat_history = []               # Borra el historial de conversación
        st.rerun()                                       # Fuerza a Streamlit a redibujar la app ya mismo

    st.divider()   # Línea horizontal separadora

    st.subheader("📂 Cargador de Documentos")   # Subtítulo de la zona de carga
    # El texto "200MB per file" lo pinta Streamlit con SU límite; esto lo baja a MAX_PDF_MB.   # Comentario
    # Se aplica solo si la versión de Streamlit lo soporta (si no, se ignora sin romper nada).   # Red de seguridad
    uploader_extra = {}                         # Diccionario de parámetros opcionales del cargador
    if "max_upload_size" in inspect.signature(st.file_uploader).parameters:   # ¿La versión lo soporta?
        uploader_extra["max_upload_size"] = MAX_PDF_MB   # Sí: se pasa el límite al propio widget
    uploaded_files = st.file_uploader(          # Componente de subida de archivos
        "Subir archivos PDF",                   # Etiqueta
        type=['pdf'],                           # Solo acepta .pdf
        accept_multiple_files=True,             # Permite varios archivos a la vez
        help=f"Tamaño máximo por archivo: {MAX_PDF_MB} MB.",   # Muestra el límite al usuario
        **uploader_extra                        # Aquí entra max_upload_size (si existe)
    )                                           # Devuelve una lista de objetos (o None si no hay nada)
    # Texto propio que NO desaparece al subir un archivo (el de Streamlit sí desaparece).   # Comentario
    if uploaded_files:                          # ¿Hay archivos subidos?
        total_mb = sum(getattr(f, "size", 0) for f in uploaded_files) / (1024 * 1024)   # Suma el tamaño en MB
        st.caption(f"📎 {len(uploaded_files)} archivo(s) · {total_mb:.1f} MB en total · máx. {MAX_PDF_MB} MB por archivo")   # Resumen permanente
    else:                                       # Si no hay archivos...
        st.caption(f"📎 Máximo {MAX_PDF_MB} MB por archivo · solo PDF")   # ...muestra solo el límite
    # Diagnóstico desplegable: límite que Streamlit tiene cargado de verdad.   # Comentario
    with st.expander("🔧 Diagnóstico de configuración"):   # Cajón cerrado por defecto
        st.write(f"Carpeta de trabajo: `{os.getcwd()}`")   # Desde qué carpeta se lanzó la app
        st.write(f"Límite efectivo de Streamlit: **{st.get_option('server.maxUploadSize')} MB**")   # Valor real cargado
    sync_with_uploader = st.toggle(             # Interruptor on/off
        "Al quitar un archivo del cargador, borrarlo del RAG",   # Texto del interruptor
        value=True,                                              # Activado por defecto
        help="Si lo desactivas, el documento seguirá indexado aunque lo quites del cargador "   # Tooltip parte 1
             "(útil si borraste el archivo por error o quieres seguir consultándolo)."          # Tooltip parte 2
    )                                                          # Cierra el toggle

# ----------------------------------------------------------------------------
# 12) CREACIÓN DEL CLIENTE DE GEMINI
# ----------------------------------------------------------------------------

# Crear el cliente de Google GenAI si hay clave
client = None            # Por defecto no hay cliente (aún no sabemos si hay clave)
if gemini_key:           # Si el campo de la clave tiene algo...
    try:                 # ...intenta crear el cliente (puede fallar si la clave es inválida)
        client = genai.Client(api_key=gemini_key)   # Cliente autenticado que usarán embeddings y generación
    except Exception as e:                          # Si la creación falla...
        st.error(f"❌ No se pudo crear el cliente de Gemini: {e}")   # ...muestra el error en la app
else:                    # Si no hay clave...
    st.warning("⚠️ Por favor, ingrese su GEMINI_API_KEY en la barra lateral para comenzar.")   # ...avisa al usuario

# ----------------------------------------------------------------------------
# 13) SINCRONIZACIÓN CARGADOR → ÍNDICE
#     Detecta qué archivos había antes en el uploader y ya no están (el
#     usuario pulsó la ✖) para borrarlos también del RAG.
# ----------------------------------------------------------------------------

# --- SINCRONIZACIÓN: archivos quitados del cargador → fuera del índice ---   # Marcador de sección
current_names = {f.name for f in uploaded_files} if uploaded_files else set()   # Conjunto de nombres presentes AHORA
if st.session_state.uploader_names is None:   # ¿Es la primera ejecución de la sesión?
    # Primera ejecución de la sesión: solo memorizamos, no borramos nada.   # Evita borrados falsos al arrancar
    st.session_state.uploader_names = current_names   # Guarda la foto actual como referencia
else:                                         # Si ya teníamos una foto anterior...
    gone = st.session_state.uploader_names - current_names   # Diferencia de conjuntos = archivos que desaparecieron
    for name in sorted(gone):                 # Recorre los desaparecidos en orden alfabético
        # Ya no está en el cargador: dejamos de ignorarlo
        st.session_state.ignored_files.discard(name)   # Quita el nombre de la lista de ignorados (ya no aplica)
        if sync_with_uploader and name in st.session_state.processed_files:   # ¿Sincronización activa y estaba indexado?
            n = st.session_state.vector_db.remove_document(name)   # Borra sus fragmentos del índice
            st.session_state.processed_files.discard(name)         # Lo quita de "procesados"
            st.toast(f"🗑️ {name} eliminado del RAG ({n} fragmentos)", icon="🗑️")   # Notificación flotante
    st.session_state.uploader_names = current_names   # Actualiza la foto para la próxima ejecución

# ----------------------------------------------------------------------------
# 14) INDEXADO DE LOS PDFs NUEVOS
# ----------------------------------------------------------------------------

# --- LÓGICA DE PROCESAMIENTO DE LOS PDFs ---   # Marcador de sección
if client and uploaded_files:              # Solo si hay cliente Y hay archivos subidos
    for uploaded_file in uploaded_files:   # Recorre cada archivo subido
        if (uploaded_file.name not in st.session_state.processed_files   # ¿No está ya indexado?
                and uploaded_file.name not in st.session_state.ignored_files):   # ¿Y no está en la lista de ignorados?
            size_mb = getattr(uploaded_file, "size", 0) / (1024 * 1024)   # Tamaño del archivo en MB
            if size_mb > MAX_PDF_MB:               # ¿Supera el límite permitido?
                with st.sidebar:                   # Abre la barra lateral para el aviso
                    st.error(f"❌ {uploaded_file.name} pesa {size_mb:.1f} MB; el máximo permitido es {MAX_PDF_MB} MB.")   # Mensaje de rechazo
                continue                           # Salta este archivo y sigue con el siguiente
            with st.sidebar:               # Vuelve a entrar en la barra lateral para poner ahí los mensajes
                with st.spinner(f"Procesando {uploaded_file.name}..."):   # Rueda de carga con el nombre del archivo
                    try:                   # Protege el indexado
                        n_chunks = st.session_state.vector_db.add_document(   # Indexa el PDF completo
                            client=client,                  # Cliente de Gemini
                            file_name=uploaded_file.name,   # Nombre del archivo (se guarda como 'source')
                            file_content=uploaded_file      # El archivo en sí (Streamlit lo da como file-like)
                        )                                   # Devuelve el número de fragmentos indexados
                        st.session_state.processed_files.add(uploaded_file.name)   # Márcalo como procesado
                        st.success(f"✅ {uploaded_file.name} indexado ({n_chunks} fragmentos).")   # Mensaje de éxito
                    except Exception as e:          # Si falló el indexado...
                        msg = str(e)                # ...convierte el error a texto
                        st.error(f"❌ No se pudo indexar {uploaded_file.name}: {msg}")   # Muéstralo en rojo
                        if "NOT_FOUND" in msg or "not found" in msg:   # Si el error es "modelo no encontrado"...
                            st.info(                                   # ...da una pista útil
                                "Ese modelo de embeddings ya no existe o no está disponible para tu "   # Parte 1 del consejo
                                f"clave/región. Modelo en uso: `{EMBEDDING_MODEL}`. "                   # Parte 2: modelo actual
                                "Prueba `gemini-embedding-001` o revisa que la API de embeddings "       # Parte 3: alternativa
                                "esté habilitada para tu proyecto."                                     # Parte 4
                            )                                          # Cierra el st.info
                        elif "429" in msg or "RESOURCE_EXHAUSTED" in msg:   # Si es cuota agotada...
                            st.info("Cuota gratuita agotada: el nivel gratuito permite unas 100 peticiones de embeddings por minuto y cada fragmento cuenta como una. Espera un minuto y reintenta, sube menos PDFs a la vez o activa la facturación en Google AI Studio.")   # ...explica qué pasó y qué hacer

# ----------------------------------------------------------------------------
# 15) LISTA DE DOCUMENTOS INDEXADOS (con borrado individual)
# ----------------------------------------------------------------------------

# Estado de los documentos indexados
with st.sidebar:                             # De nuevo en la barra lateral
    st.divider()                             # Separador
    st.subheader("Documentos Indexados")     # Título de la lista
    if st.session_state.processed_files:     # Si hay documentos indexados...
        for doc in sorted(st.session_state.processed_files):   # Recórrelos ordenados alfabéticamente
            col_name, col_del = st.columns([5, 1])   # Dos columnas: 5 partes para el nombre, 1 para el botón
            col_name.write(f"📄 {doc}")              # Nombre del archivo en la columna ancha
            if col_del.button(                       # Botón de papelera en la columna estrecha
                "🗑️",                                # Texto del botón
                key=f"del_{doc}",                    # key única: sin ella Streamlit confunde los botones
                help=f"Quitar «{doc}» del índice vectorial",   # Tooltip
            ):                                       # Si el usuario pulsó este botón...
                n = st.session_state.vector_db.remove_document(doc)   # Borra sus fragmentos del índice
                st.session_state.processed_files.discard(doc)         # Quítalo de la lista de procesados
                # Evita que se reindexe solo mientras siga en el cargador
                st.session_state.ignored_files.add(doc)               # Añádelo a ignorados
                st.toast(f"🗑️ {doc} eliminado del RAG ({n} fragmentos)", icon="🗑️")   # Aviso flotante
                st.rerun()                                            # Redibuja para reflejar el borrado
        st.caption(f"{len(st.session_state.vector_db.chunks)} fragmentos en memoria")   # Contador de fragmentos
        if st.session_state.ignored_files:   # Si hay archivos ignorados...
            st.caption(                      # ...muestra cuáles son
                "ℹ️ Ignorados (siguen en el cargador, fuera del RAG): "   # Prefijo del mensaje
                + ", ".join(sorted(st.session_state.ignored_files))       # Une los nombres con comas
            )                            # Cierra el caption
            if st.button("🔄 Volver a indexar ignorados", use_container_width=True):   # Botón de deshacer
                st.session_state.ignored_files = set()   # Vacía la lista de ignorados
                st.rerun()                               # Redibuja: se reindexarán automáticamente
    else:                                # Si NO hay documentos indexados...
        st.info("No hay documentos indexados en el sistema.")   # ...mensaje informativo
        if st.session_state.ignored_files:                      # Y si aun así hay ignorados...
            st.caption(                                         # ...muéstralos también aquí
                "ℹ️ Ignorados: " + ", ".join(sorted(st.session_state.ignored_files))   # Lista de nombres
            )                                                   # Cierra el caption
            if st.button("🔄 Volver a indexar ignorados", use_container_width=True):   # Botón de deshacer
                st.session_state.ignored_files = set()   # Vacía la lista
                st.rerun()                               # Redibuja

# ----------------------------------------------------------------------------
# 16) ZONA DE CHAT: pintar el historial
# ----------------------------------------------------------------------------

# --- ZONA PRINCIPAL DE CHAT ---   # Marcador de sección
# Pintar el historial de conversación
for chat in st.session_state.chat_history:   # Recorre todos los mensajes guardados
    with st.chat_message(chat['role']):      # Burbuja de chat con estilo de 'user' o 'assistant'
        st.markdown(chat['content'])         # Pinta el texto con formato markdown
        if chat['role'] == 'assistant' and 'sources' in chat:   # Si es una respuesta y guarda sus fuentes...
            with st.expander("🔍 Ver Fragmentos Recuperados (Fuentes)"):   # ...sección desplegable
                for i, source in enumerate(chat['sources']):   # Recorre las fuentes con su índice
                    st.markdown(f"**Fragmento {i+1} - {source['metadata']['source']} (Similitud: {source['score']:.4f})**")   # Cabecera con archivo y puntuación
                    st.code(source['text'])   # Muestra el fragmento en un bloque monoespaciado

# ----------------------------------------------------------------------------
# 17) ZONA DE CHAT: procesar una pregunta nueva
# ----------------------------------------------------------------------------

# Procesar una consulta nueva
if user_query := st.chat_input("Escriba su consulta basándose en los documentos..."):   # Caja de texto fija abajo; el walrus (:=) asigna y evalúa en un paso
    # Mostrar la pregunta del usuario
    st.chat_message("user").markdown(user_query)   # Pinta la pregunta del usuario en su burbuja
    st.session_state.chat_history.append({"role": "user", "content": user_query})   # Guárdala en el historial

    if not client:                       # Caso 1: no hay cliente (falta la API key)
        st.error("❌ API Key de Gemini requerida para responder.")   # Muestra error
    elif len(st.session_state.vector_db.chunks) == 0:   # Caso 2: no hay documentos indexados
        with st.chat_message("assistant"):              # Abre burbuja de asistente
            err_msg = "⚠️ No se han cargado documentos en el sistema. Por favor suba archivos PDF para habilitar la búsqueda semántica."   # Texto del aviso
            st.markdown(err_msg)                        # Píntalo
            st.session_state.chat_history.append({"role": "assistant", "content": err_msg})   # Guárdalo en el historial
    else:                                # Caso 3: todo listo → responder
        with st.chat_message("assistant"):   # Burbuja del asistente
            with st.spinner("Buscando respuestas en los documentos..."):   # Rueda de carga mientras se consulta
                try:                        # Protege la llamada al modelo
                    response_text, sources = preguntar_al_rag(   # Ejecuta el pipeline RAG completo
                        pregunta=user_query,                        # La pregunta
                        db=st.session_state.vector_db,              # La base de datos vectorial
                        client=client,                              # El cliente de Gemini
                        modelo_principal=selected_model             # El modelo elegido en la barra lateral
                    )                                               # Devuelve (texto_respuesta, lista_de_fuentes)
                    st.markdown(response_text)                      # Pinta la respuesta en la burbuja
                    with st.expander("🔍 Ver Fragmentos Recuperados (Fuentes)"):   # Desplegable de fuentes
                        for i, source in enumerate(sources):   # Recorre cada fuente
                            st.markdown(f"**Fragmento {i+1} - {source['metadata']['source']} (Similitud: {source['score']:.4f})**")   # Cabecera con archivo y similitud
                            st.code(source['text'])            # Texto del fragmento

                    # Guardar la respuesta en el historial
                    st.session_state.chat_history.append({   # Añade el mensaje del asistente
                        "role": "assistant",                # Quién habla
                        "content": response_text,           # Qué dijo
                        "sources": sources                  # Y qué fragmentos usó (para repintarlos luego)
                    })                                      # Cierra el diccionario
                except Exception as e:                      # Si algo falló...
                    st.error(f"Ocurrió un error al procesar la solicitud: {e}")   # ...muestra el error
                    if "NOT_FOUND" in str(e):               # Si es un modelo inexistente...
                        st.info(                            # ...sugiere qué revisar
                            "Suele ser un nombre de modelo inválido o retirado. "   # Explicación
                            f"Embeddings: `{EMBEDDING_MODEL}` · Generación: `{selected_model}`."   # Modelos en uso
                        )                                       # Cierra el st.info

# =============================================================================
#  FIN DEL ARCHIVO
#  Flujo resumido:
#    subir PDF → extraer texto → trocear → (formatear) → embeddings → guardar
#    preguntar → embedding de la pregunta → coseno → top 3 → prompt → Gemini
# =============================================================================

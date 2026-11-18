import streamlit as st
import streamlit.components.v1 as components
import ollama
from gtts import gTTS
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder
from pathlib import Path
import asyncio
import base64
import hashlib
import io
import json
import os
import re
import tempfile
import unicodedata
import uuid

BASE = Path(__file__).resolve().parent


def _leer_curso() -> dict:
    por_defecto = {
        "titulo": "Maniobra de Reversa Asistida",
        "descripcion": "una capacitación de Maniobra de Reversa Asistida del Programa Ruta 66 de Seguridad y Salud Ocupacional (SST)",
        "publico": "choferes y auxiliares (ayudantes) de reparto",
        "vocabulario": "maniobra de reversa, reversa asistida, puntos ciegos, auxiliar, ayudante, conductor, camión, "
                       "vehículo pesado, técnica 360 grados, luces intermitentes, alarma de reversa, contacto visual, "
                       "espejo, señales manuales, ALTO, AVANCE, fin de la maniobra, paro total, esquinas posteriores, "
                       "ángulo de 90 grados, acera, intersección, sentido de circulación, tránsito, 6 metros, 2 metros, "
                       "Programa Ruta 66, Seguridad y Salud Ocupacional"}
    try:
        with open(BASE / "curso.json", encoding="utf-8") as f:
            por_defecto.update(json.load(f))
    except Exception:
        pass
    return por_defecto


CURSO = _leer_curso()

st.set_page_config(page_title=f"Aula Virtual SST - {CURSO['titulo']}", layout="wide", page_icon="🚚")

# =======================================================
# CONFIGURACIÓN
# =======================================================
CARPETA_LAMINAS = BASE / "laminas"
CARPETA_VIDEOS  = BASE / "videos"
ARCHIVO_GUIONES = BASE / "guiones.json"
CARPETA_AUDIO   = BASE / "audio_cache"

MODELO = "qwen2.5:3b"
NUM_CTX = 4096
STT_MOTOR = "whisper"
WHISPER_MODELO = "small"
TTS_MOTOR = "edge"
VOZ_EDGE = "es-PE-AlexNeural"
VELOCIDAD_EDGE = "-5%"

CARPETA_LAMINAS.mkdir(parents=True, exist_ok=True)
CARPETA_VIDEOS.mkdir(parents=True, exist_ok=True)
CARPETA_AUDIO.mkdir(parents=True, exist_ok=True)


# =======================================================
# CARGA DE CONTENIDO
# =======================================================
def _clave_natural(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.stem)]


@st.cache_data(show_spinner=False)
def cargar_guiones() -> dict:
    with open(ARCHIVO_GUIONES, encoding="utf-8") as f:
        return {int(g["n"]): g for g in json.load(f)}


@st.cache_data(show_spinner=False)
def listar_laminas() -> list:
    if not CARPETA_LAMINAS.exists():
        return []
    archivos = [p for p in CARPETA_LAMINAS.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    return sorted(archivos, key=_clave_natural)


try:
    GUIONES = cargar_guiones()
except Exception as e:
    st.error(f"No pude leer `guiones.json`. Revisa que exista en: {ARCHIVO_GUIONES}  ({e})")
    st.stop()

LAMINAS = listar_laminas()
TOTAL = len(GUIONES)
TOTAL_IMAGENES = len(LAMINAS)
LAMINAS = LAMINAS[:TOTAL]

if TOTAL_IMAGENES == 0:
    st.error(
        f"No encontré imágenes en la carpeta `laminas/`.\n\n"
        f"Ruta esperada: `{CARPETA_LAMINAS}`\n\n"
        f"Crea esa carpeta y coloca dentro tus {TOTAL} imágenes como `1.png`, `2.png`, ... `{TOTAL}.png`."
    )
    st.stop()
elif TOTAL_IMAGENES != TOTAL:
    st.warning(f"Hay {TOTAL_IMAGENES} imágenes en `laminas/` pero {TOTAL} guiones en `guiones.json`. Deben coincidir.")


def multimedia_de_lamina(n: int):
    """Devuelve (tipo, ruta, aviso). Busca la imagen por NÚMERO en el nombre
    (1.png, 03.jpg, lamina_5.png, etc.) en lugar de por posición, así no importa
    si falta alguna imagen o están en desorden."""
    g = GUIONES.get(n, {})

    # 1) Video tiene prioridad si el guion lo define y el archivo existe
    if g.get("video"):
        ruta_v = CARPETA_VIDEOS / g["video"]
        if ruta_v.exists():
            return "video", ruta_v, None
        return "video", ruta_v, (
            f"⚠️ Esta lámina es el video **{g['video']}**, pero no encuentro el archivo.\n\n"
            f"Colócalo aquí: `{ruta_v}`"
        )

    # 2) Buscar imagen por número dentro del nombre (sin depender del orden)
    for ruta in LAMINAS:
        m = re.search(r"(\d+)", ruta.stem)   # primera secuencia de dígitos
        if m and int(m.group(1)) == n:
            return "imagen", ruta, None

    # 3) Nada
    return None, None, f"No hay imagen para la lámina {n} (busco un archivo con “{n}” en el nombre)."
# =======================================================
# AUDIO (TEXTO -> VOZ) con caché en disco
# =======================================================
def limpiar_para_voz(texto: str) -> str:
    return re.sub(r"[\*\#\_\[\]\-\—\•]", "", texto)


async def _edge_async(texto: str) -> bytes:
    import edge_tts
    com = edge_tts.Communicate(texto, VOZ_EDGE, rate=VELOCIDAD_EDGE)
    datos = bytearray()
    async for chunk in com.stream():
        if chunk["type"] == "audio":
            datos.extend(chunk["data"])
    return bytes(datos)


def _tts_pyttsx3(texto: str) -> bytes:
    import pyttsx3
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    motor = pyttsx3.init()
    motor.setProperty("rate", 155)
    motor.save_to_file(texto, tmp.name)
    motor.runAndWait()
    with open(tmp.name, "rb") as f:
        datos = f.read()
    os.remove(tmp.name)
    return datos


def sintetizar(texto: str):
    texto = limpiar_para_voz(texto)
    intentos = [TTS_MOTOR] + [m for m in ("gtts",) if m != TTS_MOTOR]
    for motor in intentos:
        try:
            if motor == "edge":
                return asyncio.run(_edge_async(texto)), "audio/mpeg"
            if motor == "pyttsx3":
                return _tts_pyttsx3(texto), "audio/wav"
            buf = io.BytesIO()
            gTTS(text=texto, lang="es").write_to_fp(buf)
            return buf.getvalue(), "audio/mpeg"
        except Exception:
            continue
    return None, None


def audio_de_texto(texto: str, usar_cache: bool = True):
    ext = "wav" if TTS_MOTOR == "pyttsx3" else "mp3"
    clave = hashlib.md5(f"{TTS_MOTOR}|{VOZ_EDGE}|{VELOCIDAD_EDGE}|{texto}".encode()).hexdigest()
    ruta = CARPETA_AUDIO / f"{clave}.{ext}"
    mime = "audio/wav" if ext == "wav" else "audio/mpeg"
    if usar_cache and ruta.exists():
        return ruta.read_bytes(), mime
    datos, mime_real = sintetizar(texto)
    if datos and usar_cache:
        ruta.write_bytes(datos)
    return datos, (mime_real or mime)


# =======================================================
# RENDER: TODO EN st.html (sin iframe) → control total desde el padre
# =======================================================
def _b64_de_archivo(ruta: Path) -> str:
    return base64.b64encode(ruta.read_bytes()).decode()


def _mime_imagen(ruta: Path) -> str:
    return "image/png" if ruta.suffix.lower() == ".png" else "image/jpeg"


def _mime_video(ruta: Path) -> str:
    ext = ruta.suffix.lower()
    return {"mp4": "video/mp4", "webm": "video/webm",
            "ogg": "video/ogg", "mov": "video/quicktime"}.get(ext[1:], "video/mp4")


def render_lámina(slide_n: int, tipo: str, ruta: Path,
                  audio_bytes, audio_mime, auto_avanzar: bool):
    """
    Inserta imagen/video + audio DIRECTO en el documento padre (st.html, sin iframe).
    Cada audio al montarse:
      1) Pausa y destruye audios/videos hermanos (los viejos) — así NUNCA suena la lámina anterior.
      2) Se marca como "actual".
      3) Fuerza su propia reproducción (por si autoplay fue bloqueado).
      4) Al terminar, si auto_avanzar, pulsa 'Siguiente' SOLO si sigue siendo el actual.
    """
    marca = f"lam{slide_n}-{uuid.uuid4().hex[:8]}"
    id_audio = f"au-{marca}"
    id_video = f"vi-{marca}"

    partes = []

    if tipo == "imagen":
        b64 = _b64_de_archivo(ruta)
        partes.append(
            f'<div style="display:flex;justify-content:center;">'
            f'<img src="data:{_mime_imagen(ruta)};base64,{b64}" '
            f'style="max-width:100%;max-height:56vh;border-radius:8px;">'
            f'</div>'
        )
    elif tipo == "video":
        b64 = _b64_de_archivo(ruta)
        mime_v = _mime_video(ruta)
        partes.append(
            f'<video id="{id_video}" controls autoplay playsinline preload="auto" '
            f'style="width:100%;max-height:56vh;border-radius:8px;background:#000;display:block;">'
            f'<source src="data:{mime_v};base64,{b64}" type="{mime_v}"></video>'
        )

    if audio_bytes and tipo != "video":
        b64 = base64.b64encode(audio_bytes).decode()
        partes.append(
            f'<audio id="{id_audio}" controls autoplay preload="auto" '
            f'style="width:100%;margin-top:10px;">'
            f'<source src="data:{audio_mime};base64,{b64}" type="{audio_mime}">'
            f'</audio>'
        )

    auto_js = ""
    if auto_avanzar:
        auto_js = """
        function siguiente() {
            if (P.__audioActual !== id_audio && P.__videoActual !== id_video) return;
            setTimeout(function() {
                const botones = P.document.querySelectorAll('button');
                for (const b of botones) {
                    if (b.innerText.includes('Siguiente') && !b.disabled) { b.click(); break; }
                }
            }, 800);
        }
        if (document.getElementById(id_audio)) {
            document.getElementById(id_audio).addEventListener('ended', siguiente);
        }
        if (document.getElementById(id_video)) {
            document.getElementById(id_video).addEventListener('ended', siguiente);
        }
        """

    script = f"""
    <script>
    (function() {{
      const P = window.parent;
      const id_audio = "{id_audio}";
      const id_video = "{id_video}";

      // 1) Matar cualquier audio/video hermano (los de la lámina anterior)
      try {{
        P.document.querySelectorAll('audio, video').forEach(function(el) {{
          if (el.id === id_audio || el.id === id_video) return;
          try {{
            el.pause();
            el.currentTime = 0;
            el.removeAttribute('src');
            el.load();
          }} catch(e) {{}}
        }});
      }} catch(e) {{}}

      // 2) Registrar el nuevo como "actual"
      P.__audioActual = id_audio;
      P.__videoActual = id_video;

      // 3) Forzar reproducción del nuevo (por si el autoplay fue bloqueado)
      setTimeout(function() {{
        const a = document.getElementById(id_audio);
        if (a) {{ a.muted = false; a.play().catch(function(){{}}); }}
        const v = document.getElementById(id_video);
        if (v) {{ v.play().catch(function(){{}}); }}
      }}, 60);

      // 4) Auto-avance al terminar
      {auto_js}
    }})();
    </script>
    """

    partes.append(script)
    st.html("".join(partes))


# =======================================================
# VOZ -> TEXTO
# =======================================================
@st.cache_resource(show_spinner="Cargando modelo de voz (solo la primera vez)...")
def cargar_whisper():
    from faster_whisper import WhisperModel
    return WhisperModel(WHISPER_MODELO, device="cpu", compute_type="int8")


def _wav_a_numpy(audio_bytes: bytes):
    import wave
    import numpy as np
    with wave.open(io.BytesIO(audio_bytes), "rb") as w:
        canales, ancho, tasa = w.getnchannels(), w.getsampwidth(), w.getframerate()
        crudo = w.readframes(w.getnframes())
    if ancho == 2:
        a = np.frombuffer(crudo, dtype=np.int16).astype(np.float32) / 32768.0
    elif ancho == 4:
        a = np.frombuffer(crudo, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif ancho == 1:
        a = (np.frombuffer(crudo, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Formato WAV no soportado ({ancho * 8} bits)")
    if canales > 1:
        a = a[: len(a) // canales * canales].reshape(-1, canales).mean(axis=1)
    a = _remuestrear(a, tasa, 16000)
    pico = float(np.abs(a).max()) if len(a) else 0.0
    if pico > 0.003:
        a = (a * min(0.9 / pico, 12.0)).astype(np.float32)
    return a


def _remuestrear(a, tasa: int, destino: int = 16000):
    import numpy as np
    if tasa == destino or len(a) < 2:
        return a
    if tasa > destino:
        fc = 0.45 * destino / tasa
        n = 101
        t = np.arange(n) - (n - 1) / 2
        h = np.sinc(2 * fc * t) * np.hamming(n)
        h /= h.sum()
        a = np.convolve(a, h, mode="same")
    n_out = int(len(a) * destino / tasa)
    return np.interp(np.linspace(0, len(a) - 1, n_out), np.arange(len(a)), a).astype(np.float32)


def _transcribir_google(audio_bytes: bytes):
    rec = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
        audio = rec.record(source)
    return rec.recognize_google(audio, language="es-PE")


def transcribir_audio(audio_bytes: bytes):
    if STT_MOTOR == "whisper":
        try:
            audio = _wav_a_numpy(audio_bytes)
            if len(audio) < 8000 or float(abs(audio).max()) < 0.01:
                st.session_state.error_audio = "El audio es muy corto o casi no se oye (revisa el micrófono)."
                return None
            pista = (f"Pregunta de un chofer o auxiliar de reparto sobre {CURSO['titulo']}. "
                     f"Vocabulario: {CURSO['vocabulario']}.")
            segmentos, _ = cargar_whisper().transcribe(
                audio, language="es", beam_size=5, initial_prompt=pista,
                condition_on_previous_text=False)
            texto = " ".join(seg.text for seg in segmentos).strip()
            return texto or None
        except Exception as e:
            st.session_state.error_audio = f"Whisper: {e}"
    try:
        texto = _transcribir_google(audio_bytes)
        st.session_state.pop("error_audio", None)
        return texto
    except Exception as e:
        prev = st.session_state.get("error_audio", "")
        st.session_state.error_audio = (prev + " | " if prev else "") + f"Google: {e}"
        return None


# =======================================================
# ASISTENTE DE DUDAS (OLLAMA)
# =======================================================
SYSTEM_PROMPT_BASE = """Eres el Profesor Jorge, instructor de Seguridad y Salud en el Trabajo (SST) en Perú.
Dictas {descripcion}, dirigida a: {publico}.

REGLAS ESTRICTAS:
- Responde en español, en texto plano (sin markdown), máximo 4 líneas.
- Responde ÚNICAMENTE con la información de los FRAGMENTOS DE LA CLASE. Copia los datos concretos (reglas, pasos, señales, distancias, nombres) tal como aparecen ahí.
- NO agregues procedimientos, distancias, señales ni cifras que no estén en los fragmentos, aunque los conozcas. Esto es seguridad: un dato inventado puede costar una vida.
- Si la pregunta no aparece en los fragmentos, di: "Eso no lo vemos en esta capacitación" y sugiere preguntar sobre el tema de la clase.
- Si la pregunta no se entiende (puede venir de un micrófono), pide amablemente que la repita. No repitas tu respuesta anterior.
- Sé cordial, respetuoso y pedagógico. Termina con una pregunta corta."""

SYSTEM_PROMPT = SYSTEM_PROMPT_BASE.format(descripcion=CURSO["descripcion"], publico=CURSO["publico"])

OPCIONES_LLM = {"temperature": 0.1, "num_predict": 200, "num_ctx": NUM_CTX}

_STOP = set("""de la el los las un una unos unas y o a en por para con sin que se es son al del lo le les su sus mi me te nos
como mas muy si no ni ya hay esta este esto esa ese eso estos estas cual cuales cuando donde quien quienes porque
debe deben debo puede pueden puedo usar uso usa utiliza utilizan sirve sirven hacer hace hacen ser ver tiene tienen
dime dame explicame explica hazme pregunta lamina clase profesor
maniobra reversa retroceder camion vehiculo""".split())


def _norm(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", t.lower()) if unicodedata.category(c) != "Mn")


def _raices(t: str) -> set:
    return {w[:5] for w in re.findall(r"[a-z0-9]+", _norm(t)) if len(w) > 2 and w not in _STOP}


def fragmentos_relevantes(pregunta: str, n_actual: int, k: int = 3) -> list:
    q = _raices(pregunta)
    puntajes = []
    for i, g in GUIONES.items():
        if i == n_actual or not q:
            continue
        titulo = _raices(g["titulo"])
        pts = 3 * len(q & titulo) + len(q & _raices(g["resumen"] + " " + g["guion"]))
        if titulo and titulo <= q:
            pts += 5
        if pts > 0:
            puntajes.append((-pts, i))
    puntajes.sort()
    return [i for _, i in puntajes[:k]]


def construir_mensajes(pregunta: str, n: int, historial: list) -> list:
    indice = "\n".join(f"Lámina {k}: {g['titulo']}" for k, g in GUIONES.items())
    ids = [n] + fragmentos_relevantes(pregunta, n)
    fragmentos = "\n\n".join(f"[Lámina {i}: {GUIONES[i]['titulo']}]\n{GUIONES[i]['guion']}" for i in ids)
    sistema = (f"{SYSTEM_PROMPT}\n\nTEMARIO DE LA CLASE:\n{indice}\n\n"
               f"FRAGMENTOS DE LA CLASE (única fuente permitida; el alumno está viendo la lámina {n}):\n{fragmentos}")
    msgs = [{"role": "system", "content": sistema}]
    msgs += [{"role": m["role"], "content": m["content"]} for m in historial[-4:]]
    msgs.append({"role": "user", "content": f"Pregunta del alumno: {pregunta}"})
    return msgs


def stream_profesor(pregunta: str, n: int, historial: list):
    for chunk in ollama.chat(model=MODELO, messages=construir_mensajes(pregunta, n, historial),
                             stream=True, keep_alive="30m", options=OPCIONES_LLM):
        yield chunk["message"]["content"]


def verificar_ollama():
    try:
        ollama.show(MODELO)
        return None
    except Exception as e:
        t = str(e).lower()
        if "not found" in t or "404" in t:
            return f"Falta descargar el modelo del asistente. En una terminal ejecuta:  ollama pull {MODELO}"
        return "No puedo conectar con Ollama. Ábrelo (o ejecuta `ollama serve`) y recarga la página."


# =======================================================
# ESTADO Y CALLBACKS
# =======================================================
def init_estado():
    defaults = {
        "slide": 1,
        "selector_slide": 1,
        "slide_anterior": 0,
        "modo_clase": False,
        "forzar_narracion": False,
        "audio_slide_n": 0,
        "audio_bytes": None,
        "audio_mime": None,
        "historial": [{"role": "assistant",
                       "content": f"Hola, soy el Profesor Jorge. Pulsa «Iniciar clase» para comenzar la capacitación de {CURSO['titulo']}, o pregúntame lo que quieras."}],
        "responder_pendiente": False,
        "ollama_ok": False,
        "proyector": False,
        "borrador": None,
        "borrador_edit": "",
        "ultima_proy": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def cambiar_slide(delta: int):
    st.session_state.slide = min(max(1, st.session_state.slide + delta), TOTAL)
    st.session_state.selector_slide = st.session_state.slide


def ir_a_slide(n: int):
    st.session_state.slide = int(n)


def iniciar_clase():
    total = sum(1 for g in GUIONES.values() if not g.get("video"))
    progreso = st.progress(0, text="🎙️ Preparando la voz del profesor...")
    hechos = 0
    for i, g in GUIONES.items():
        if g.get("video"):
            continue
        audio_de_texto(g["guion"])
        hechos += 1
        progreso.progress(hechos / max(total, 1), text=f"🎙️ Preparando voz {hechos}/{total}...")
    progreso.empty()

    st.session_state.modo_clase = True
    st.session_state.slide_anterior = 0
    st.session_state.audio_slide_n = 0
    st.session_state.audio_bytes = None
    st.session_state.audio_mime = None


def pausar_clase():
    st.session_state.modo_clase = False
    st.session_state.audio_slide_n = 0
    st.session_state.audio_bytes = None
    st.session_state.audio_mime = None


def escuchar_lamina():
    st.session_state.forzar_narracion = True
    st.session_state.slide_anterior = 0


def enviar_pregunta(texto: str):
    pausar_clase()
    st.session_state.historial.append({"role": "user", "content": texto})
    st.session_state.responder_pendiente = True


def guardar_borrador(texto: str):
    st.session_state.borrador = texto
    st.session_state.borrador_edit = texto


def enviar_borrador():
    texto = (st.session_state.get("borrador_edit") or "").strip()
    st.session_state.borrador = None
    if texto:
        enviar_pregunta(texto)


def descartar_borrador():
    st.session_state.borrador = None


def panel_borrador():
    if st.session_state.borrador is not None:
        st.text_input("✏️ Esto entendí. Corrígelo si hace falta y envía:", key="borrador_edit")
        c1, c2 = st.columns(2)
        c1.button("📨 Enviar pregunta", use_container_width=True, on_click=enviar_borrador, type="primary")
        c2.button("🗑️ Descartar", use_container_width=True, on_click=descartar_borrador)


def alternar_proyector():
    st.session_state.proyector = not st.session_state.proyector


def reiniciar():
    pausar_clase()
    st.session_state.slide = 1
    st.session_state.selector_slide = 1
    st.session_state.slide_anterior = 0
    st.session_state.historial = st.session_state.historial[:1]
    st.session_state.ultima_proy = None
    st.session_state.borrador = None
    st.session_state.responder_pendiente = False


CSS_PROYECTOR = """
<style>
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"],
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], footer {display: none !important;}
.block-container {padding: 0.4rem 1.2rem 0.4rem 1.2rem !important; max-width: 100% !important;}
</style>
"""


def barra_atajos():
    components.html("""
    <button id="fs" style="padding:6px 14px;border-radius:8px;border:1px solid #888;background:#fff;cursor:pointer;font-size:14px">
        ⛶ Pantalla completa
    </button>
    <span style="font-size:12px;color:#666;margin-left:10px">Teclas: ← → cambian de lámina · F pantalla completa</span>
    <script>
      const P = window.parent;
      function pantallaCompleta() {
        const d = P.document;
        if (d.fullscreenElement) { d.exitFullscreen(); } else { d.documentElement.requestFullscreen(); }
      }
      document.getElementById('fs').addEventListener('click', pantallaCompleta);
      if (!P.__aulaTeclas) {
        P.__aulaTeclas = true;
        P.document.addEventListener('keydown', function (e) {
          const t = ((e.target && e.target.tagName) || '').toLowerCase();
          if (t === 'input' || t === 'textarea') return;
          const clic = function (txt) {
            for (const b of P.document.querySelectorAll('button')) {
              if (b.innerText.includes(txt) && !b.disabled) { b.click(); return; }
            }
          };
          if (e.key === 'ArrowRight' || e.key === 'PageDown') clic('Siguiente');
          else if (e.key === 'ArrowLeft' || e.key === 'PageUp') clic('Anterior');
          else if (e.key === 'f' || e.key === 'F') pantallaCompleta();
        });
      }
    </script>
    """, height=44)


init_estado()

# =======================================================
# LÓGICA: al cambiar de lámina, calcular el audio nuevo
# =======================================================
if st.session_state.slide != st.session_state.slide_anterior:
    st.session_state.slide_anterior = st.session_state.slide

    st.session_state.audio_slide_n = 0
    st.session_state.audio_bytes = None
    st.session_state.audio_mime = None

    narrar = st.session_state.modo_clase or st.session_state.pop("forzar_narracion", False)
    st.session_state.forzar_narracion = False

    lamina_actual = GUIONES.get(st.session_state.slide, {})
    if lamina_actual.get("video"):
        narrar = False

    if narrar and not st.session_state.responder_pendiente:
        datos, mime = audio_de_texto(GUIONES[st.session_state.slide]["guion"])
        if datos:
            st.session_state.audio_bytes = datos
            st.session_state.audio_mime = mime
            st.session_state.audio_slide_n = st.session_state.slide
        else:
            st.session_state.modo_clase = False
            st.error("No pude generar la voz. Si usas 'edge', necesitas internet; prueba TTS_MOTOR = 'pyttsx3'.")

# =======================================================
# INTERFAZ
# =======================================================
proy = st.session_state.proyector

if not proy:
    st.title(f"🚚 Aula Virtual SST — {CURSO['titulo']}")

if not proy and not st.session_state.ollama_ok:
    err = verificar_ollama()
    if err:
        st.error(err + "\n\n(La clase dictada funciona sin Ollama, pero el asistente de dudas no.)")
    else:
        st.session_state.ollama_ok = True

# ---- Barra lateral ----
with st.sidebar:
    st.button("📽️ Modo proyector", use_container_width=True, on_click=alternar_proyector, type="primary")
    st.divider()
    st.header("🎓 Acciones rápidas")
    st.button("🧒 Explícamelo más simple", use_container_width=True, on_click=enviar_pregunta,
              args=("Explícame esta lámina de forma más simple, como si fuera mi primer día como ayudante de reparto.",))
    st.button("🚚 Dame un ejemplo de mi ruta", use_container_width=True, on_click=enviar_pregunta,
              args=("Dame un ejemplo práctico de esta lámina en mi trabajo diario, retrocediendo el camión en una ruta de reparto.",))
    st.button("📍 ¿Dónde me ubico como auxiliar?", use_container_width=True, on_click=enviar_pregunta,
              args=("¿Dónde debe ubicarse el auxiliar durante una maniobra de reversa, qué señales debe usar y qué hace el conductor si pierde el contacto visual?",))
    st.button("📝 Hazme una pregunta (quiz)", use_container_width=True, on_click=enviar_pregunta,
              args=("Hazme UNA sola pregunta de práctica sobre esta lámina y espera mi respuesta. "
                    "No des la respuesta todavía. Cuando yo responda, dime si estoy correcto y explica brevemente.",))
    st.divider()
    st.selectbox("Ir a la lámina", options=list(GUIONES.keys()),
                 format_func=lambda k: f"{k}. {GUIONES[k]['titulo']}",
                 key="selector_slide",
                 on_change=lambda: ir_a_slide(st.session_state.selector_slide))
    st.button("🔄 Reiniciar todo", use_container_width=True, on_click=reiniciar)

    st.divider()
    st.caption("🔍 Diagnóstico")
    st.caption(f"Lámina visible: **{st.session_state.slide}**")
    st.caption(f"Audio cargado: **lámina {st.session_state.audio_slide_n or '—'}**")
    if st.session_state.audio_bytes:
        h = hashlib.md5(st.session_state.audio_bytes).hexdigest()[:8]
        st.caption(f"Hash audio: `{h}`")
    st.caption(f"Modo clase: {'ON' if st.session_state.modo_clase else 'OFF'}")

    st.caption(f"Asistente: `{MODELO}` · Voz→texto: `{STT_MOTOR}` · Texto→voz: `{TTS_MOTOR}`")

chat_box = None
resp_box = None
audio_grabado = None
texto_chat = None

if proy:
    # ---------- MODO PROYECTOR ----------
    st.markdown(CSS_PROYECTOR, unsafe_allow_html=True)
    col_clase = st.container()
    col_chat = None
    with col_clase:
        tipo, ruta, aviso = multimedia_de_lamina(st.session_state.slide)
        media_box = st.empty()
        if tipo == "video" and ruta and ruta.exists():
            with media_box:
                render_lámina(st.session_state.slide, "video", ruta, None, None,
                              auto_avanzar=st.session_state.modo_clase)
        elif tipo == "video":
            st.error(aviso)
        elif tipo == "imagen":
            with media_box:
                render_lámina(st.session_state.slide, "imagen", ruta,
                              st.session_state.audio_bytes, st.session_state.audio_mime,
                              auto_avanzar=st.session_state.modo_clase and bool(st.session_state.audio_bytes))
        elif aviso:
            st.warning(aviso)

        q1, q2, q3, q4 = st.columns(4)
        q1.button("⬅️ Anterior", use_container_width=True, on_click=cambiar_slide, args=(-1,),
                  disabled=st.session_state.slide <= 1)
        if st.session_state.modo_clase:
            q2.button("⏸️ Pausar clase", use_container_width=True, on_click=pausar_clase, type="primary")
        else:
            q2.button("▶️ Iniciar / Reanudar", use_container_width=True, on_click=iniciar_clase, type="primary")
        q3.button("Siguiente ➡️", use_container_width=True, on_click=cambiar_slide, args=(1,),
                  disabled=st.session_state.slide >= TOTAL)
        q4.button("✖ Salir del proyector", use_container_width=True, on_click=alternar_proyector)
        st.progress(st.session_state.slide / TOTAL)
        barra_atajos()

        m1, m2 = st.columns([0.28, 0.72])
        with m1:
            audio_grabado = mic_recorder(start_prompt="🎤 Preguntar con voz", stop_prompt="🟢 Enviar pregunta",
                                         just_once=True, format="wav", key="mic_proy")
        resp_box = m2.container()
        if audio_grabado and audio_grabado.get("bytes"):
            with st.spinner("🎙️ Transcribiendo..."):
                texto = transcribir_audio(audio_grabado["bytes"])
            if texto:
                guardar_borrador(texto)
            else:
                st.warning("No pude entender el audio, intenta de nuevo.")
        with resp_box:
            panel_borrador()
        if st.session_state.ultima_proy and not st.session_state.responder_pendiente:
            with resp_box:
                st.markdown(f"**🎤 Pregunta:** {st.session_state.ultima_proy[0]}")
                st.write(st.session_state.ultima_proy[1])
else:
    # ---------- MODO INSTRUCTOR ----------
    col_clase, col_chat = st.columns([0.58, 0.42])

    with col_clase:
        b1, b2, b3, b4 = st.columns(4)
        if st.session_state.modo_clase:
            b1.button("✋ Tengo una duda", use_container_width=True, on_click=pausar_clase, type="primary")
        else:
            b1.button("▶️ Iniciar / Reanudar clase", use_container_width=True, on_click=iniciar_clase, type="primary")
        b2.button("🔊 Escuchar lámina", use_container_width=True, on_click=escuchar_lamina)
        b3.button("⬅️ Anterior", use_container_width=True, on_click=cambiar_slide, args=(-1,),
                  disabled=st.session_state.slide <= 1)
        b4.button("Siguiente ➡️", use_container_width=True, on_click=cambiar_slide, args=(1,),
                  disabled=st.session_state.slide >= TOTAL)

        estado = "🟢 Clase en curso" if st.session_state.modo_clase else "⏸️ Clase en pausa"
        st.markdown(f"**Lámina {st.session_state.slide} de {TOTAL}** — {GUIONES[st.session_state.slide]['titulo']} &nbsp;·&nbsp; {estado}")
        st.progress(st.session_state.slide / TOTAL)

        tipo, ruta, aviso = multimedia_de_lamina(st.session_state.slide)
        media_box = st.empty()
        if tipo == "video" and ruta and ruta.exists():
            with media_box:
                render_lámina(st.session_state.slide, "video", ruta, None, None,
                              auto_avanzar=st.session_state.modo_clase)
        elif tipo == "video":
            st.error(aviso)
        elif tipo == "imagen":
            with media_box:
                render_lámina(st.session_state.slide, "imagen", ruta,
                              st.session_state.audio_bytes, st.session_state.audio_mime,
                              auto_avanzar=st.session_state.modo_clase and bool(st.session_state.audio_bytes))
        elif aviso:
            st.warning(aviso)

        with st.expander("📜 Ver el texto que dicta el profesor en esta lámina"):
            st.write(GUIONES[st.session_state.slide]["guion"])

    with col_chat:
        st.subheader("💬 Dudas al Profesor")
        chat_box = st.container(height=440)
        audio_grabado = mic_recorder(start_prompt="🔴 Grabar voz", stop_prompt="🟢 Enviar audio",
                                     just_once=True, format="wav", key="mic_aula")
        texto_chat = st.chat_input("Escribe tu duda aquí...")

        if audio_grabado and audio_grabado.get("bytes"):
            with st.spinner("🎙️ Transcribiendo..."):
                texto = transcribir_audio(audio_grabado["bytes"])
            if texto:
                guardar_borrador(texto)
            else:
                st.warning("No pude entender el audio, intenta de nuevo.")
        elif texto_chat:
            enviar_pregunta(texto_chat)
        panel_borrador()

# ---- Chat: historial + respuesta con streaming ----
if chat_box is not None:
    with chat_box:
        for msg in st.session_state.historial:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if st.session_state.responder_pendiente:
            st.session_state.responder_pendiente = False
            pregunta = st.session_state.historial[-1]["content"]
            contexto = st.session_state.historial[:-1]
            with st.chat_message("assistant"):
                try:
                    respuesta = st.write_stream(stream_profesor(pregunta, st.session_state.slide, contexto))
                except Exception as e:
                    respuesta = f"El asistente no está disponible. Verifica que Ollama esté abierto. ({e})"
                    st.write(respuesta)
            st.session_state.historial.append({"role": "assistant", "content": respuesta})
            datos, mime = audio_de_texto(respuesta, usar_cache=False)
            if datos:
                st.session_state.audio_bytes = datos
                st.session_state.audio_mime = mime
                st.session_state.audio_slide_n = st.session_state.slide

elif resp_box is not None and st.session_state.responder_pendiente:
    st.session_state.responder_pendiente = False
    pregunta = st.session_state.historial[-1]["content"]
    contexto = st.session_state.historial[:-1]
    with resp_box:
        st.markdown(f"**🎤 Pregunta:** {pregunta}")
        try:
            respuesta = st.write_stream(stream_profesor(pregunta, st.session_state.slide, contexto))
        except Exception as e:
            respuesta = f"El asistente no está disponible. Verifica que Ollama esté abierto. ({e})"
            st.write(respuesta)
    st.session_state.historial.append({"role": "assistant", "content": respuesta})
    st.session_state.ultima_proy = (pregunta, respuesta)
    datos, mime = audio_de_texto(respuesta, usar_cache=False)
    if datos:
        st.session_state.audio_bytes = datos
        st.session_state.audio_mime = mime
        st.session_state.audio_slide_n = st.session_state.slide
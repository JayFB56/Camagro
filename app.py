import os
import time
import threading
import logging

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify, render_template_string


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CAMERA_STREAM_URL = os.environ.get(
    "CAMERA_STREAM_URL",
    "https://46ec-190-107-238-11.ngrok-free.app/stream",
).strip()
RECONNECT_DELAY_SECONDS = 3
MAX_READ_FAILURES = 3

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
_hand_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=4,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

app = Flask(__name__)

_state_lock = threading.Lock()
_latest_estado = {
    "num_manos": 0,
    "manos": [],
    "pulgares_extendidos": [],
    "timestamp": None,
    "conectado": False,
}
_latest_jpeg = None
_frame_version = 0

# El lector reemplaza este único frame de entrada. Si la cámara produce más
# rápido de lo que MediaPipe procesa, se conserva el más reciente y se saltan
# los anteriores; no se construye una cola.
_raw_condition = threading.Condition()
_latest_raw_frame = None
_raw_frame_version = 0

_workers_lock = threading.Lock()
_workers_started = False
_stop_event = threading.Event()


def _make_waiting_jpeg():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:] = (24, 32, 44)
    message = "Esperando conexion del ESP32-CAM..."
    text_size = cv2.getTextSize(
        message, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    )[0]
    x = max(12, (image.shape[1] - text_size[0]) // 2)
    y = image.shape[0] // 2
    cv2.putText(
        image,
        message,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (235, 242, 248),
        2,
        cv2.LINE_AA,
    )
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("No se pudo crear el frame de espera.")
    return encoded.tobytes()


_latest_jpeg = _make_waiting_jpeg()


def _classify_hand(landmarks):
    """Clasifica cuatro dedos por altura y evalúa el pulgar en el eje X."""
    finger_pairs = ((8, 6), (12, 10), (16, 14), (20, 18))
    fingers_up = sum(
        landmarks[tip].y < landmarks[joint].y
        for tip, joint in finger_pairs
    )
    state = "abierta" if fingers_up >= 3 else "cerrada"

    wrist_x = landmarks[0].x
    middle_mcp_x = landmarks[9].x
    palm_direction = 1 if middle_mcp_x >= wrist_x else -1
    thumb_ip_from_wrist = (landmarks[3].x - wrist_x) * -palm_direction
    thumb_tip_from_wrist = (landmarks[4].x - wrist_x) * -palm_direction
    thumb_extended = thumb_tip_from_wrist > thumb_ip_from_wrist

    return state, thumb_extended


def _draw_summary(image, states):
    if states:
        names = ", ".join(
            "Abierta" if state == "abierta" else "Cerrada"
            for state in states
        )
    else:
        names = "Sin manos"

    summary = "Manos: {} | {}".format(len(states), names)
    cv2.rectangle(image, (0, 0), (image.shape[1], 38), (24, 32, 44), -1)
    cv2.putText(
        image,
        summary,
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _set_camera_connected(connected):
    global _latest_jpeg, _frame_version

    with _state_lock:
        if _latest_estado["conectado"] == connected:
            return

        _latest_estado["conectado"] = connected
        if not connected:
            _latest_estado["num_manos"] = 0
            _latest_estado["manos"] = []
            _latest_estado["pulgares_extendidos"] = []
            _latest_jpeg = _make_waiting_jpeg()
            _frame_version += 1


def _camera_reader_loop():
    global _latest_raw_frame, _raw_frame_version

    if not CAMERA_STREAM_URL:
        logger.error(
            "Configura CAMERA_STREAM_URL con la URL del stream del ESP32-CAM."
        )
        return

    logger.info("Conectando al stream de cámara: %s", CAMERA_STREAM_URL)

    while not _stop_event.is_set():
        try:
            capture = cv2.VideoCapture(
                CAMERA_STREAM_URL,
                cv2.CAP_FFMPEG,
                [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                    5000,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                    5000,
                ],
            )
        except Exception:
            logger.exception("No se pudo abrir el stream de la cámara.")
            _set_camera_connected(False)
            _stop_event.wait(RECONNECT_DELAY_SECONDS)
            continue

        if not capture.isOpened():
            logger.warning(
                "No se pudo abrir %s; se reintentará en %s segundos.",
                CAMERA_STREAM_URL,
                RECONNECT_DELAY_SECONDS,
            )
            capture.release()
            _set_camera_connected(False)
            _stop_event.wait(RECONNECT_DELAY_SECONDS)
            continue

        # Es una preferencia del backend de video; si no está soportada,
        # VideoCapture simplemente la ignora.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        logger.info("Stream ESP32-CAM conectado.")
        consecutive_failures = 0

        while not _stop_event.is_set():
            ok, frame = capture.read()
            if not ok or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= MAX_READ_FAILURES:
                    logger.warning("El stream dejó de entregar frames.")
                    _set_camera_connected(False)
                    break
                time.sleep(0.05)
                continue

            consecutive_failures = 0
            _set_camera_connected(True)

            with _raw_condition:
                _latest_raw_frame = frame
                _raw_frame_version += 1
                _raw_condition.notify()

        capture.release()
        if not _stop_event.is_set():
            _stop_event.wait(RECONNECT_DELAY_SECONDS)


def _frame_processor_loop():
    global _latest_jpeg, _frame_version

    last_processed_version = 0

    while not _stop_event.is_set():
        with _raw_condition:
            _raw_condition.wait_for(
                lambda: (
                    _raw_frame_version != last_processed_version
                    or _stop_event.is_set()
                ),
                timeout=1.0,
            )
            if _stop_event.is_set():
                return
            if _raw_frame_version == last_processed_version:
                continue

            frame = _latest_raw_frame.copy()
            last_processed_version = _raw_frame_version

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = _hand_detector.process(rgb_frame)
            states = []
            thumbs_extended = []
            annotated = frame.copy()

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    state, thumb_extended = _classify_hand(
                        hand_landmarks.landmark
                    )
                    states.append(state)
                    thumbs_extended.append(thumb_extended)

                    mp_drawing.draw_landmarks(
                        annotated,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                    )

                    wrist = hand_landmarks.landmark[0]
                    x = int(wrist.x * annotated.shape[1])
                    y = max(50, int(wrist.y * annotated.shape[0]) - 12)
                    hand_label = (
                        "ABIERTA" if state == "abierta" else "CERRADA"
                    )
                    thumb_label = (
                        "PULGAR +" if thumb_extended else "PULGAR -"
                    )
                    cv2.putText(
                        annotated,
                        hand_label + " | " + thumb_label,
                        (max(4, x - 40), y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (40, 220, 90)
                        if state == "abierta"
                        else (40, 170, 245),
                        2,
                        cv2.LINE_AA,
                    )

            _draw_summary(annotated, states)
            jpeg_ok, encoded_output = cv2.imencode(".jpg", annotated)
            if not jpeg_ok:
                logger.warning("No se pudo codificar un frame procesado.")
                continue

            with _state_lock:
                _latest_jpeg = encoded_output.tobytes()
                _frame_version += 1
                _latest_estado.update(
                    {
                        "num_manos": len(states),
                        "manos": states,
                        "pulgares_extendidos": thumbs_extended,
                        "timestamp": int(time.time()),
                    }
                )
        except Exception:
            logger.exception("Error procesando un frame del ESP32-CAM.")


def _start_background_workers():
    global _workers_started

    with _workers_lock:
        if _workers_started:
            return

        threading.Thread(
            target=_camera_reader_loop,
            name="esp32-camera-reader",
            daemon=True,
        ).start()
        threading.Thread(
            target=_frame_processor_loop,
            name="hand-frame-processor",
            daemon=True,
        ).start()
        _workers_started = True


@app.before_request
def start_workers_before_request():
    _start_background_workers()


def _mjpeg_stream():
    last_sent_version = -1

    while True:
        with _state_lock:
            jpeg = _latest_jpeg
            version = _frame_version

        if version != last_sent_version:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: "
                + str(len(jpeg)).encode("ascii")
                + b"\r\n\r\n"
                + jpeg
                + b"\r\n"
            )
            last_sent_version = version

        time.sleep(0.1)


INDEX_HTML = """<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#101a2c">
  <title>Camagro · Robot de bienvenida</title>
  <style>
    :root{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;background:#0b1220;color:#edf5ff;--muted:#9aacc2;--line:rgba(171,197,226,.13);--cyan:#5ee7df}
    *{box-sizing:border-box}body{min-height:100vh;margin:0;padding:28px;background:radial-gradient(ellipse at 80% 5%,rgba(39,111,155,.23),transparent 35%),radial-gradient(ellipse at 10% 90%,rgba(43,75,143,.18),transparent 38%),#0b1220}
    .shell{width:min(1180px,100%);margin:auto}header{display:flex;align-items:center;justify-content:space-between;gap:18px;margin-bottom:25px}.brand{display:flex;align-items:center;gap:13px}.brand-mark{display:grid;place-items:center;width:44px;height:44px;border:1px solid #5ee7df55;border-radius:14px;background:#5ee7df18;color:var(--cyan);font-size:21px}.brand-name{margin:0;font-size:1rem;letter-spacing:.17em;font-weight:800}.brand-sub{margin:4px 0 0;color:var(--muted);font-size:.76rem}.connection{display:inline-flex;align-items:center;gap:9px;padding:9px 13px;border:1px solid var(--line);border-radius:999px;background:#142032;color:#b6c5d7;font-size:.82rem}.dot{width:8px;height:8px;border-radius:50%;background:#68778a}.connection.online{color:#baf5d3}.connection.online .dot{background:#66e3a1;box-shadow:0 0 12px #66e3a188}
    .hero{margin-bottom:22px}.eyebrow{margin:0 0 9px;color:var(--cyan);font-size:.72rem;font-weight:800;letter-spacing:.19em;text-transform:uppercase}h1{margin:0;font-size:clamp(1.8rem,4vw,3rem);line-height:1.08;letter-spacing:-.045em}.lead{max-width:660px;margin:12px 0 0;color:var(--muted);line-height:1.65}
    .grid{display:grid;grid-template-columns:minmax(300px,.88fr) minmax(0,1.35fr);gap:18px}.panel{overflow:hidden;border:1px solid var(--line);border-radius:22px;background:linear-gradient(145deg,#19273cf0,#0f1929f0);box-shadow:0 22px 70px #0003}.robot-panel{display:flex;flex-direction:column;min-height:510px;padding:22px}.panel-head,.camera-head{display:flex;align-items:flex-start;justify-content:space-between;gap:12px}.panel-label{margin:0;color:var(--muted);font-size:.72rem;font-weight:750;letter-spacing:.13em;text-transform:uppercase}.robot-name,.camera-title{margin:5px 0 0;font-size:1.16rem}.unit-chip,.live-tag{padding:6px 9px;border:1px solid var(--line);border-radius:8px;color:#a9bed5;font:700 .67rem ui-monospace,monospace;letter-spacing:.08em}.robot-stage{position:relative;display:grid;flex:1;place-items:center;min-height:340px;margin:5px -10px -3px;background:radial-gradient(ellipse at 50% 56%,#4e9dc026,transparent 51%)}.robot-stage:after{position:absolute;bottom:18px;width:150px;height:18px;border-radius:50%;background:#0005;filter:blur(8px);content:""}.robot{z-index:1;width:min(100%,330px);height:380px;overflow:visible}.metal{fill:url(#metal);stroke:#9ab2cc;stroke-opacity:.56;stroke-width:2}.edge{fill:none;stroke:#a9bfd7;stroke-opacity:.54;stroke-width:2}.joint{fill:#1a2b42;stroke:#93acc7;stroke-opacity:.68;stroke-width:2}.glow{fill:#62e4dd;filter:url(#softGlow)}.screen{fill:#10253a;stroke:#62e4dd;stroke-opacity:.55;stroke-width:2}.robot-caption{margin:0;text-align:center;color:#91a7bd;font-size:.82rem;line-height:1.5}.is-greeting .robot-caption{color:#aaf1e7}
    .camera-panel{display:flex;flex-direction:column;padding:22px}.camera-head{margin-bottom:16px}.live-tag{display:inline-flex;align-items:center;gap:7px;color:#91e8df;background:#5ee7df0f}.live-tag:before{width:6px;height:6px;border-radius:50%;background:currentColor;content:""}.video-frame{display:grid;place-items:center;min-height:260px;flex:1;overflow:hidden;border:1px solid #b0c9e226;border-radius:14px;background:#080e17;color:#9db0c6;line-height:0}.video-frame img{display:block;width:100%;height:100%;min-height:260px;max-height:420px;object-fit:contain}.metrics{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:14px}.metric{padding:13px 14px;border:1px solid var(--line);border-radius:13px;background:#0a132085}.metric-label{margin:0;color:var(--muted);font-size:.73rem}.metric-value{margin:6px 0 0;font-size:1.05rem;font-weight:750}.greeting-card{display:flex;align-items:center;gap:13px;margin-top:14px;padding:14px 15px;border:1px solid #72a7ff29;border-radius:14px;background:linear-gradient(100deg,#3d699f26,#5ee7df0f)}.greeting-icon{display:grid;flex:0 0 40px;place-items:center;width:40px;height:40px;border-radius:12px;background:#72a7ff1f;font-size:1.2rem}.greeting-text{margin:0;font-weight:700;line-height:1.4}.greeting-note{margin:3px 0 0;color:var(--muted);font-size:.77rem}footer{margin:18px 2px 0;color:#70839a;font-size:.76rem;line-height:1.5;text-align:center}
    @media(max-width:760px){body{padding:18px 14px}.grid{grid-template-columns:1fr}.robot-panel{min-height:440px}.robot-stage{min-height:290px}.robot{height:330px}.camera-panel{min-height:490px;padding:18px}}@media(prefers-reduced-motion:reduce){*,*:before,*:after{animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important}}
  </style>
</head>
<body>
  <main class="shell">
    <header><div class="brand"><div class="brand-mark" aria-hidden="true">✳</div><div><p class="brand-name">CAMAGRO</p><p class="brand-sub">Interacción humanoide · visión artificial</p></div></div><div id="connection" class="connection"><span class="dot"></span><span id="connectionText">Conectando cámara</span></div></header>
    <section class="hero"><p class="eyebrow">Asistente de gestos · unidad 01</p><h1>Un saludo, con un gesto.</h1><p class="lead">Milo observa la cámara y responde cuando detecta una mano abierta. El movimiento se activa únicamente ante ese gesto.</p></section>
    <section class="grid" aria-label="Panel de interacción Camagro">
      <article id="robotPanel" class="panel robot-panel">
        <div class="panel-head"><div><p class="panel-label">Robot humanoide</p><h2 class="robot-name">Milo</h2></div><span class="unit-chip">CAM-01</span></div>
        <div class="robot-stage">
          <svg class="robot" viewBox="0 0 320 390" role="img" aria-labelledby="robotTitle robotDesc" xmlns="http://www.w3.org/2000/svg">
            <title id="robotTitle">Robot humanoide Milo</title><desc id="robotDesc">Robot con ojos luminosos y brazo animado al reconocer una sola mano abierta.</desc>
            <defs><linearGradient id="metal" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#e1edf8"/><stop offset=".42" stop-color="#91a9c2"/><stop offset="1" stop-color="#536b86"/></linearGradient><linearGradient id="visor" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#163b53"/><stop offset="1" stop-color="#0d1b2d"/></linearGradient><filter id="softGlow" x="-100%" y="-100%" width="300%" height="300%"><feGaussianBlur stdDeviation="3" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>
            <path d="M160 38V25" class="edge"/><circle cx="160" cy="19" r="6" class="glow"/><rect x="105" y="39" width="110" height="82" rx="28" class="metal"/><rect x="116" y="56" width="88" height="43" rx="17" fill="url(#visor)" stroke="#b5c9dd" stroke-opacity=".3"/><ellipse cx="139" cy="77" rx="7" ry="9" class="glow"/><ellipse cx="181" cy="77" rx="7" ry="9" class="glow"/><path d="M146 105q14 8 28 0" class="edge" stroke-linecap="round"/><circle cx="98" cy="80" r="8" class="joint"/><circle cx="222" cy="80" r="8" class="joint"/><path d="M160 121v18" stroke="#7992ad" stroke-width="12" stroke-linecap="round"/>
            <path d="M119 143q0-8 10-8h62q10 0 10 8l13 89q2 13-11 13h-76q-13 0-11-13z" class="metal"/><rect x="132" y="158" width="56" height="43" rx="12" class="screen"/><path d="M143 180h8l6-10 8 17 7-10h8" fill="none" stroke="#63e4dd" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/><circle cx="144" cy="218" r="4" class="glow"/><circle cx="160" cy="218" r="4" fill="#718aa5"/><circle cx="176" cy="218" r="4" fill="#718aa5"/>
            <path d="M119 153L91 184L79 220" fill="none" stroke="#7891aa" stroke-width="19" stroke-linecap="round" stroke-linejoin="round"/><path d="M119 153L91 184M91 184L79 220" fill="none" stroke="url(#metal)" stroke-width="14" stroke-linecap="round"/><circle cx="119" cy="153" r="12" class="joint"/><circle cx="91" cy="184" r="9" class="joint"/><path d="M79 218l-8 12m8-12 2 15m-2-15 10 11" class="edge" stroke-linecap="round"/>
            <g id="waveArm"><path d="M221 153L246 183L253 222" fill="none" stroke="#7189a2" stroke-width="20" stroke-linecap="round" stroke-linejoin="round"/><path d="M221 153L246 183M246 183L253 222" fill="none" stroke="url(#metal)" stroke-width="14" stroke-linecap="round"/><circle cx="221" cy="153" r="12" class="joint"/><circle cx="246" cy="183" r="9" class="joint"/><circle cx="253" cy="222" r="8" class="joint"/><g id="waveHand"><path d="M246 225q0-8 7-8t8 8v10q4-11 8-8t1 11l-3 14q-2 10-13 10h-8q-11-1-12-12l-1-13q0-6 4-6t6 7z" class="metal"/><path d="M250 230v9m8-8v8m8-5-2 7" class="edge" stroke-linecap="round"/></g></g>
            <path d="M126 246h68l-4 20h-60z" fill="#142238" stroke="#829ab3" stroke-opacity=".5" stroke-width="2"/><path d="M139 264l-7 56m49-56 7 56" fill="none" stroke="url(#metal)" stroke-width="20" stroke-linecap="round"/><circle cx="132" cy="293" r="8" class="joint"/><circle cx="188" cy="293" r="8" class="joint"/><path d="M132 317l-5 28q-1 8 8 8h26q7 0 6-7l-8-25m22-4 5 28q1 8-8 8h-26q-7 0-6-7l8-25" class="metal"/><path d="M131 342h31m16 0h31" stroke="#62e4dd" stroke-opacity=".75" stroke-width="3" stroke-linecap="round"/><path d="M124 368h72" stroke="#5b718b" stroke-opacity=".45" stroke-width="2" stroke-dasharray="4 7"/>
          </svg>
        </div>
        <p id="robotCaption" class="robot-caption">Estoy listo. Muéstrame una mano abierta para saludarte.</p>
      </article>
      <article class="panel camera-panel">
        <div class="camera-head"><div><p class="panel-label">Visión en tiempo real</p><h2 class="camera-title">Cámara ESP32-CAM</h2></div><span class="live-tag">LIVE</span></div>
        <div class="video-frame"><img src="/video" alt="Stream procesado de la cámara ESP32-CAM"></div>
        <div class="metrics"><div class="metric"><p class="metric-label">Manos detectadas</p><p id="handCount" class="metric-value">—</p></div><div class="metric"><p class="metric-label">Gestos</p><p id="handStates" class="metric-value">Esperando…</p></div></div>
        <div class="greeting-card" role="status" aria-live="polite"><div id="greetingIcon" class="greeting-icon" aria-hidden="true">◉</div><div><p id="saludo" class="greeting-text">Esperando conexión del ESP32-CAM…</p><p id="greetingNote" class="greeting-note">El saludo se activa solo con una mano abierta.</p></div></div>
      </article>
    </section>
    <footer>La cámara se procesa en el servidor · Estado actualizado cada 500 ms</footer>
  </main>
  <script>
    const connection=document.getElementById("connection"),connectionText=document.getElementById("connectionText"),robotPanel=document.getElementById("robotPanel"),robotCaption=document.getElementById("robotCaption"),saludo=document.getElementById("saludo"),greetingNote=document.getElementById("greetingNote"),greetingIcon=document.getElementById("greetingIcon"),handCount=document.getElementById("handCount"),handStates=document.getElementById("handStates");
    const waveArm=document.getElementById("waveArm"),waveHand=document.getElementById("waveHand");
    let waveAnimationFrame=null;

    function animateWave(timestamp){
      const armAngle=-94+Math.sin(timestamp/150)*12;
      const handAngle=Math.sin(timestamp/95)*14;
      waveArm.setAttribute("transform","rotate("+armAngle.toFixed(1)+" 221 153)");
      waveHand.setAttribute("transform","rotate("+handAngle.toFixed(1)+" 258 238)");
      waveAnimationFrame=window.requestAnimationFrame(animateWave);
    }

    function setWave(active){
      robotPanel.classList.toggle("is-greeting",active);
      if(active&&waveAnimationFrame===null){
        waveAnimationFrame=window.requestAnimationFrame(animateWave);
      }else if(!active&&waveAnimationFrame!==null){
        window.cancelAnimationFrame(waveAnimationFrame);
        waveAnimationFrame=null;
        waveArm.removeAttribute("transform");
        waveHand.removeAttribute("transform");
      }
    }
    function describirEstado(estado){
      const connected=Boolean(estado&&estado.conectado),manos=Array.isArray(estado&&estado.manos)?estado.manos:[],abiertas=manos.filter(m=>m==="abierta").length,cerradas=manos.filter(m=>m==="cerrada").length;
      const openOnly=connected&&manos.length===1&&manos[0]==="abierta";
      connection.classList.toggle("online",connected);connectionText.textContent=connected?"Cámara conectada":"Esperando cámara";setWave(openOnly);handCount.textContent=connected?String(manos.length):"—";handStates.textContent=!connected?"Sin conexión":manos.length===0?"Ninguna":abiertas+" abierta(s) · "+cerradas+" cerrada(s)";
      if(!connected){saludo.textContent="Esperando conexión del ESP32-CAM…";greetingNote.textContent="Comprueba la URL del stream y el acceso del servidor.";greetingIcon.textContent="◉";robotCaption.textContent="Estoy en reposo hasta que llegue la señal de cámara."}
      else if(openOnly){saludo.textContent="¡Hola! 👋 Mano abierta detectada";greetingNote.textContent="Milo te saluda. Mantén la palma visible para continuar.";greetingIcon.textContent="👋";robotCaption.textContent="¡Qué gusto verte! Milo está saludando."}
      else if(manos.length===0){saludo.textContent="Esperando una mano…";greetingNote.textContent="Acerca tu mano al campo de visión de la cámara.";greetingIcon.textContent="⌖";robotCaption.textContent="Estoy listo. Muéstrame una mano abierta para saludarte."}
      else if(manos.length===1&&manos[0]==="cerrada"){saludo.textContent="✊ Puño detectado";greetingNote.textContent="El robot permanece en reposo con la mano cerrada.";greetingIcon.textContent="✊";robotCaption.textContent="He reconocido el gesto; espero una palma abierta para saludar."}
      else{saludo.textContent="Se detectaron "+manos.length+" manos";greetingNote.textContent="Para activar el saludo, muestra únicamente una mano abierta.";greetingIcon.textContent="◎";robotCaption.textContent="El saludo está reservado para una sola mano abierta."}
    }
    async function actualizarEstado(){try{const response=await fetch("/estado",{cache:"no-store"});if(!response.ok)throw new Error("HTTP "+response.status);describirEstado(await response.json())}catch(error){describirEstado({conectado:false,manos:[]});saludo.textContent="No se pudo consultar el servidor.";greetingNote.textContent="Reintentando automáticamente…"}}
    actualizarEstado();window.setInterval(actualizarEstado,500);
  </script>
</body>
</html>"""


@app.get("/")
def index():
    return render_template_string(INDEX_HTML)


@app.get("/video")
def video():
    return Response(
        _mjpeg_stream(),
        content_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
        },
    )


@app.get("/estado")
def estado():
    with _state_lock:
        current = {
            **_latest_estado,
            "manos": list(_latest_estado["manos"]),
            "pulgares_extendidos": list(
                _latest_estado["pulgares_extendidos"]
            ),
        }
    return jsonify(current)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True,
        use_reloader=False,
    )
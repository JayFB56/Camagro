import os
import time
import threading
import logging

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

# Un solo frame se procesa a la vez. Si llega otro mientras el detector está
# ocupado, se descarta sin ponerlo en una cola.
_processing_lock = threading.Lock()

# Este lock protege tanto el JPEG actual como el estado que consultan /video
# y /estado.
_state_lock = threading.Lock()
_frame_version = 0
_latest_estado = {
    "num_manos": 0,
    "manos": [],
    "pulgares_extendidos": [],
    "timestamp": None,
    "conectado": False,
}


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
_hand_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=4,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def _make_waiting_jpeg():
    """Crea un frame inicial para que /video funcione antes del primer POST."""
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
    """Devuelve el estado de los cuatro dedos y si el pulgar está extendido."""
    finger_pairs = ((8, 6), (12, 10), (16, 14), (20, 18))
    fingers_up = sum(
        landmarks[tip].y < landmarks[joint].y
        for tip, joint in finger_pairs
    )
    state = "abierta" if fingers_up >= 3 else "cerrada"

    # Se toma la dirección horizontal de la palma desde la muñeca (0) hasta
    # el nudillo del dedo medio (9). El pulgar extendido se aleja de la palma.
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


def _mjpeg_stream():
    last_sent_version = -1
    while True:
        with _state_lock:
            jpeg = _latest_jpeg
            version = _frame_version

        # Solo se transmite un nuevo JPEG cuando cambia el frame más reciente.
        # Los frames intermedios no se acumulan ni se envían en cola.
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
  <title>Camagro · Detector de manos</title>
  <style>
    :root {
      color-scheme: light;
      font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
      background: #eef3f7;
      color: #172b3a;
    }
    * { box-sizing: border-box; }
    body {
      min-height: 100vh;
      margin: 0;
      display: grid;
      place-items: center;
      padding: 24px;
    }
    main {
      width: min(900px, 100%);
      padding: 28px;
      background: #fff;
      border: 1px solid #dce5eb;
      border-radius: 18px;
      box-shadow: 0 16px 42px rgba(27, 54, 74, .10);
    }
    h1 { margin: 0 0 8px; font-size: clamp(1.5rem, 4vw, 2.1rem); }
    .intro { margin: 0 0 22px; color: #526777; }
    .video-frame {
      overflow: hidden;
      background: #18202c;
      border-radius: 12px;
      line-height: 0;
    }
    .video-frame img { display: block; width: 100%; height: auto; }
    .status {
      margin-top: 18px;
      padding: 16px 18px;
      border-radius: 12px;
      background: #f0f6fa;
      border-left: 5px solid #2c7a9b;
      font-size: 1.1rem;
      line-height: 1.5;
    }
    .details { margin: 12px 0 0; color: #526777; font-size: .95rem; }
    @media (max-width: 560px) {
      main { padding: 20px; }
      body { padding: 14px; }
    }
  </style>
</head>
<body>
  <main>
    <h1>Camagro · Detector de manos</h1>
    <p class="intro">Video en vivo y estado detectado por el ESP32-CAM.</p>
    <div class="video-frame">
      <img src="/video" alt="Video en vivo del ESP32-CAM">
    </div>
    <div id="saludo" class="status" role="status" aria-live="polite">
      Esperando conexión del ESP32-CAM...
    </div>
    <p id="detalle" class="details">El estado se actualiza cada 500 ms.</p>
  </main>

  <script>
    const saludo = document.getElementById("saludo");
    const detalle = document.getElementById("detalle");

    function describirEstado(estado) {
      if (!estado.conectado) {
        saludo.textContent = "Esperando conexión del ESP32-CAM...";
        detalle.textContent = "Aún no se ha recibido ningún frame.";
        return;
      }

      const manos = Array.isArray(estado.manos) ? estado.manos : [];
      const abiertas = manos.filter((mano) => mano === "abierta").length;
      const cerradas = manos.filter((mano) => mano === "cerrada").length;

      if (manos.length === 0) {
        saludo.textContent = "Esperando una mano...";
      } else if (manos.length === 1 && abiertas === 1) {
        saludo.textContent = "¡Hola! 👋 Mano abierta detectada";
      } else if (manos.length === 1 && cerradas === 1) {
        saludo.textContent = "✊ Puño detectado";
      } else {
        saludo.textContent = "Se detectaron " + manos.length + " manos.";
      }

      detalle.textContent =
        abiertas + " abierta(s) · " + cerradas + " cerrada(s)";
    }

    async function actualizarEstado() {
      try {
        const response = await fetch("/estado", { cache: "no-store" });
        if (!response.ok) {
          throw new Error("HTTP " + response.status);
        }
        describirEstado(await response.json());
      } catch (error) {
        saludo.textContent = "No se pudo consultar el estado del servidor.";
        detalle.textContent = "Reintentando automáticamente...";
      }
    }

    actualizarEstado();
    window.setInterval(actualizarEstado, 500);
  </script>
</body>
</html>
"""


@app.get("/")
def index():
    return render_template_string(INDEX_HTML)


@app.post("/upload")
def upload():
    global _latest_jpeg, _latest_estado, _frame_version

    if request.mimetype != "image/jpeg":
        return jsonify({"error": "El Content-Type debe ser image/jpeg."}), 415

    # No esperar por el detector: un frame que llega mientras otro se procesa
    # se descarta inmediatamente, evitando que se forme una cola.
    if not _processing_lock.acquire(blocking=False):
        return jsonify({"estado": "omitido", "motivo": "detector ocupado"}), 202

    try:
        payload = request.get_data(cache=False)
        if not payload:
            return jsonify({"error": "El cuerpo de la petición está vacío."}), 400

        encoded_input = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(encoded_input, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "El cuerpo no contiene un JPEG válido."}), 400

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = _hand_detector.process(rgb_frame)

        states = []
        thumbs_extended = []
        annotated = frame.copy()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                state, thumb_extended = _classify_hand(hand_landmarks.landmark)
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
                hand_label = "ABIERTA" if state == "abierta" else "CERRADA"
                thumb_label = "PULGAR +" if thumb_extended else "PULGAR -"
                cv2.putText(
                    annotated,
                    hand_label + " | " + thumb_label,
                    (max(4, x - 40), y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (40, 220, 90) if state == "abierta" else (40, 170, 245),
                    2,
                    cv2.LINE_AA,
                )

        _draw_summary(annotated, states)
        jpeg_ok, encoded_output = cv2.imencode(".jpg", annotated)
        if not jpeg_ok:
            return jsonify({"error": "No se pudo codificar el frame procesado."}), 500

        new_estado = {
            "num_manos": len(states),
            "manos": states,
            "pulgares_extendidos": thumbs_extended,
            "timestamp": int(time.time()),
            "conectado": True,
        }

        with _state_lock:
            _latest_jpeg = encoded_output.tobytes()
            _latest_estado = new_estado
            _frame_version += 1

        return jsonify({"estado": "procesado", **new_estado}), 200
    except Exception:
        logger.exception("Error procesando un frame recibido en /upload")
        return jsonify({"error": "No se pudo procesar el frame."}), 500
    finally:
        _processing_lock.release()


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


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "El frame supera el límite de 8 MB."}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
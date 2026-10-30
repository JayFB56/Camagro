# Camagro — backend para ESP32-CAM

Backend Flask que recibe imágenes JPEG de un ESP32-CAM, detecta hasta cuatro manos con MediaPipe Hands, clasifica cada una como abierta o cerrada y publica el frame procesado mediante streaming MJPEG.

## Qué hace

- **POST /upload**: recibe un JPEG con **Content-Type: image/jpeg**, lo decodifica con OpenCV, procesa los landmarks y reemplaza el último frame y estado guardados en memoria.
- **GET /video**: transmite el frame procesado más reciente en formato MJPEG.
- **GET /estado**: devuelve el número de manos, sus estados, el estado del pulgar, la hora Unix y si ya llegó un frame.
- **GET /**: muestra el video y actualiza el saludo consultando **/estado** cada 500 ms.

La mano se considera abierta cuando al menos tres de los dedos índice, medio, anular y meñique tienen la punta más arriba que su articulación media. El pulgar se evalúa aparte usando la coordenada X respecto a la muñeca y la dirección de la palma.

Si llega un POST mientras MediaPipe está procesando otro frame, el servidor lo descarta con HTTP 202 para no acumular trabajo. Solo se conserva el JPEG más reciente. El estado y el frame se comparten en memoria entre hilos del mismo proceso; al reiniciar el servicio se limpian.

## Requisitos

Se recomienda Python 3.11. Las versiones están fijadas en **requirements.txt**; MediaPipe 0.10.21 conserva la API Legacy **mp.solutions.hands** y publica ruedas para Python 3.10/3.11.

## Ejecución local

Desde PowerShell, dentro de la carpeta del proyecto:

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    python app.py

La aplicación escucha en http://localhost:5000. Si existe la variable de entorno **PORT**, la usa en lugar de 5000.

Para enviar un frame manualmente, el cuerpo de la petición debe ser el contenido binario de un JPEG y el encabezado debe ser **Content-Type: image/jpeg**.

## Despliegue en Render

1. Crea un repositorio de GitHub y sube la rama **main**.
2. En Render, crea un Web Service conectado a ese repositorio.
3. Render detectará **render.yaml** y usará el plan gratuito, el comando de instalación y el comando de inicio declarados allí.
4. El Blueprint fija Python 3.11.11 para que coincida con las ruedas de MediaPipe y ejecuta Gunicorn con un worker y varios hilos. El endpoint **/video** mantiene abierta la conexión MJPEG; los hilos permiten atender a la vez las subidas y el polling, mientras un único worker mantiene el estado en memoria compartido.

No se necesitan variables secretas para este backend. Render proporciona **PORT** automáticamente.

## Configuración del ESP32-CAM

En el código del ESP32-CAM, configura la variable **servidorURL** con la URL pública de Render seguida de **/upload**:

    const char* servidorURL = "https://TU-APP.onrender.com/upload";

Cada petición debe ser un HTTP POST cuyo cuerpo sea el JPEG y cuyo **Content-Type** sea **image/jpeg**.

## Respuesta de /estado

    {
      "num_manos": 2,
      "manos": ["abierta", "cerrada"],
      "pulgares_extendidos": [true, false],
      "timestamp": 1793385600,
      "conectado": true
    }

Antes del primer frame, **conectado** es **false**, **manos** está vacío y la página muestra “Esperando conexión del ESP32-CAM...”.
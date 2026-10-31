# Camagro — detección de manos desde ESP32-CAM

Backend Flask que lee directamente el stream MJPEG del ejemplo **CameraWebServer** del Arduino IDE. El ESP32-CAM solo captura y publica video; Python obtiene los frames desde la URL de la cámara, detecta hasta cuatro manos con MediaPipe Hands y publica el video anotado.

## Flujo

- El ESP32-CAM ejecuta el ejemplo CameraWebServer y se conecta a la Wi-Fi.
- Python abre la URL del stream mediante OpenCV y procesa el frame más reciente.
- Si la cámara entrega frames más rápido de lo que MediaPipe los procesa, se sobrescriben en un búfer de un solo frame; no se crea una cola.
- La página web consume el resultado del backend en **/video** y consulta **/estado** cada 500 ms.

La mano se considera abierta cuando al menos tres de los dedos índice, medio, anular y meñique tienen la punta más arriba que su articulación media. El pulgar se evalúa aparte usando la coordenada X respecto a la muñeca y la dirección de la palma.

## Endpoints

- **GET /**: página web con video y saludo.
- **GET /video**: stream MJPEG con los landmarks y el estado dibujados.
- **GET /estado**: estado JSON actual.
- **POST /upload** ya no se utiliza; Python obtiene los frames desde la cámara.

## Requisitos

Se recomienda Python 3.11. Las versiones están fijadas en **requirements.txt**. MediaPipe 0.10.21 incluye la API Legacy **mp.solutions.hands** y publica ruedas para Python 3.10 y 3.11. Las versiones posteriores a 0.10.30 retiraron esa API.

## Configurar el ejemplo del ESP32-CAM

1. En Arduino IDE abre **Archivo > Ejemplos > ESP32 > Camera > CameraWebServer**.
2. Selecciona el modelo de cámara correcto en el sketch y configura el nombre y la contraseña de la Wi-Fi.
3. Carga el ejemplo y abre el Monitor Serie.
4. Copia la IP que muestra la placa y comprueba desde un navegador en la misma Wi-Fi que abre la página de la cámara.

En el ejemplo oficial CameraWebServer, la página de control usa el puerto 80 y el stream se sirve en el puerto 81 por la ruta **/stream**. La URL suele tener este formato:

    http://192.168.1.50:81/stream

Cambia la IP por la que muestra el Monitor Serie. Si tu ejemplo usa otro puerto, ruta o firmware, configura su URL real.

## Ejecución local en la misma Wi-Fi

En PowerShell, desde la carpeta del proyecto:

    py -3.11 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    $env:CAMERA_STREAM_URL = "http://192.168.1.50:81/stream"
    python app.py

El código incluye como URL predeterminada el túnel actual `https://46ec-190-107-238-11.ngrok-free.app/stream`. Para usar una cámara local, configura `CAMERA_STREAM_URL` como en el ejemplo anterior. Las URL gratuitas de ngrok pueden cambiar cuando se reinicia el túnel; si eso ocurre, actualiza la URL en Render o en `app.py`. El backend escucha en http://localhost:5000. Si existe **PORT**, usa ese puerto.

Si cambiaste versiones de dependencias en **requirements.txt**, recrea el entorno virtual para que no queden paquetes antiguos:

    deactivate
    Remove-Item -Recurse -Force .venv
    py -3.11 -m venv .venv

## Comprobación local

Con el entorno virtual activo, comprueba los imports y la versión de MediaPipe:

    python -c "from importlib.metadata import version; import mediapipe as mp; print(version('mediapipe')); print(mp.solutions.hands)"

Comprueba que la cámara sea accesible desde la misma PC:

    Invoke-WebRequest -Method Head $env:CAMERA_STREAM_URL

El stream es continuo, así que una petición HEAD puede no estar admitida por algunas versiones del ejemplo. También puedes pegar la URL en el navegador. Luego abre http://localhost:5000 para ver el video procesado. El endpoint **/estado** muestra la conexión y el estado detectado.

## Despliegue en Render

1. Publica el proyecto en GitHub y crea un Web Service conectado al repositorio.
2. Render usa **render.yaml**, que fija Python 3.11.11 y arranca Gunicorn con un worker y varios hilos.
3. En la configuración Environment del servicio, configura **CAMERA_STREAM_URL** con `https://46ec-190-107-238-11.ngrok-free.app/stream` (o la URL vigente del túnel). Esta variable reemplaza la URL predeterminada del código.
4. Despliega nuevamente y confirma en los logs que OpenCV conecta con el stream.

**Importante:** una dirección privada como 192.168.x.x solo funciona desde dispositivos de esa misma red. Render no puede conectarse directamente a una IP privada de tu casa o laboratorio. Para que Render lea la cámara, el stream debe ser alcanzable desde Internet mediante una conexión/túnel configurado en la red de la cámara. Si no quieres exponer la cámara, ejecuta este backend en una PC conectada a la misma Wi-Fi que el ESP32-CAM.

Render proporciona **PORT** automáticamente. No se necesita otra variable secreta para el backend.

import requests
from loguru import logger
import os
import sys
import json

# Configuración de Logging
LOG_FILE = "ndvi_polygon_high_resolution.log"
logger.add(LOG_FILE, rotation="1 MB", retention="7 days", level="DEBUG")

# Configuración de la API de Sentinel Hub desde variables de entorno
CLIENT_ID = "3e4dc531-29fc-4a94-815e-a7b50660c813"
CLIENT_SECRET = "l2nFRqtcPOg1MYvdQkVbDFO7328uvyFK"
TOKEN_URL = "https://services.sentinel-hub.com/oauth/token"
PROCESS_URL = "https://services.sentinel-hub.com/api/v1/process"

# Verificar que las credenciales estén establecidas
if not CLIENT_ID or not CLIENT_SECRET:
    logger.error("Las variables de entorno SENTINELHUB_CLIENT_ID y SENTINELHUB_CLIENT_SECRET deben estar configuradas.")
    sys.exit(1)

# Configuración del Área de Interés (AOI) y Fechas
# Coordenadas del polígono en formato [longitud, latitud]
POLYGON_COORDINATES = [
    [
        [-72.476854, -45.805818],  # Oeste, Sur
        [-72.476854, -45.282606],  # Oeste, Norte
        [-71.581419, -45.282606],  # Este, Norte
        [-71.581419, -45.805818],  # Este, Sur
        [-72.476854, -45.805818]   # Cierre del polígono
    ]
]
TIME_FROM = "2024-02-01"
TIME_TO = "2024-10-31"

# Ruta para almacenar la imagen resultante
OUTPUT_IMAGE_PATH = "ndvi_polygon_high_res_result.jpeg"

def obtener_token(client_id, client_secret, token_url):
    """
    Obtiene el token de acceso de la API de Sentinel Hub.
    """
    token_data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret
    }
    token_headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }

    try:
        logger.info("Solicitando token de acceso...")
        response = requests.post(token_url, data=token_data, headers=token_headers)
        if response.status_code == 200:
            access_token = response.json().get("access_token")
            logger.info("Token de acceso obtenido correctamente.")
            return access_token
        else:
            logger.error(f"Error en la solicitud del token: {response.status_code} {response.text}")
            sys.exit(1)
    except requests.exceptions.RequestException as e:
        logger.error(f"Excepción al solicitar el token: {e}")
        sys.exit(1)

def generar_evalscript():
    """
    Genera el evalscript para calcular y visualizar NDVI con una rampa de colores.
    """
    evalscript = """
    //VERSION=3
    function setup() {
       return {
          input: ["B04", "B08", "dataMask"],
          output: { bands: 4 }
       };
    }
    
    const ramp = [
       [-0.5, 0x0c0c0c],
       [-0.2, 0xbfbfbf],
       [-0.1, 0xdbdbdb],
       [0, 0xeaeaea],
       [0.025, 0xfff9cc],
       [0.05, 0xede8b5],
       [0.075, 0xddd89b],
       [0.1, 0xccc682],
       [0.125, 0xbcb76b],
       [0.15, 0xafc160],
       [0.175, 0xa3cc59],
       [0.2, 0x91bf51],
       [0.25, 0x7fb247],
       [0.3, 0x70a33f],
       [0.35, 0x609635],
       [0.4, 0x4f892d],
       [0.45, 0x3f7c23],
       [0.5, 0x306d1c],
       [0.55, 0x216011],
       [0.6, 0x0f540a],
       [1, 0x004400],
    ];
    
    const visualizer = new ColorRampVisualizer(ramp);
    
    function evaluatePixel(samples) {
       let ndvi = (samples.B08 - samples.B04) / (samples.B08 + samples.B04);
       let imgVals = visualizer.process(ndvi);
       return imgVals.concat(samples.dataMask)
    }
    """
    return evalscript

def solicitar_ndvi_jpeg_polygon(access_token, process_url, polygon_coords, time_from, time_to, evalscript, output_format="image/jpeg", quality=80, width=1024, height=1024, resx=None, resy=None):
    """
    Solicita una imagen NDVI en formato JPEG con límites definidos por un polígono.
    Permite aumentar la resolución especificando width/height o resx/resy.
    """
    request_payload = {
        "input": {
            "bounds": {
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": polygon_coords  # Corregido: Sin envolver en una lista adicional
                }
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": f"{time_from}T00:00:00Z",
                            "to": f"{time_to}T23:59:59Z"
                        }
                    },
                    "processing": {
                        "harmonizeValues": "true"
                    }
                }
            ]
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [
                {
                    "identifier": "default",
                    "format": {
                        "type": output_format,
                        "quality": quality
                    }
                }
            ]
        }
    }

    if resx and resy:
        request_payload["output"]["resx"] = resx
        request_payload["output"]["resy"] = resy

    files = {
        'request': (None, json.dumps(request_payload), 'application/json'),
        'evalscript': (None, evalscript, 'application/javascript')
    }

    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    try:
        logger.info("Realizando solicitud de NDVI como imagen JPEG con polígono...")
        response = requests.post(process_url, headers=headers, files=files)
        if response.status_code == 200:
            content_type = response.headers.get("Content-Type", "")
            if content_type.lower() == "image/jpeg":
                # Guardar la imagen
                with open(OUTPUT_IMAGE_PATH, "wb") as f:
                    f.write(response.content)
                logger.info(f"Imagen NDVI guardada como {OUTPUT_IMAGE_PATH}")
                return True
            else:
                logger.error(f"Error en la solicitud de NDVI: Content-Type recibido es {content_type}, se esperaba 'image/jpeg'.")
                logger.error(f"Mensaje de Error: {response.text}")
                return False
        else:
            logger.error(f"Error en la solicitud de NDVI: {response.status_code} {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Excepción al solicitar NDVI: {e}")
        return False

def main():
    access_token = obtener_token(CLIENT_ID, CLIENT_SECRET, TOKEN_URL)
    if not access_token:
        logger.error("No se pudo obtener el token de acceso. Terminando el script.")
        sys.exit(1)
    
    evalscript = generar_evalscript()
    
    exito = solicitar_ndvi_jpeg_polygon(
        access_token=access_token,
        process_url=PROCESS_URL,
        polygon_coords=POLYGON_COORDINATES,
        time_from=TIME_FROM,
        time_to=TIME_TO,
        evalscript=evalscript,
        output_format="image/jpeg",
        quality=80,          
        width=1024,          
        height=1024
    )
    
    if exito:
        logger.info("Proceso completado exitosamente.")
    else:
        logger.error("No se pudo obtener la imagen NDVI en el formato solicitado.")

if __name__ == "__main__":
    main()

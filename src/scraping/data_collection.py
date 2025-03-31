"""
Funciones para obtener las transcipciones de los videos recolectados
"""
import os
import yt_dlp
import whisper
import pandas as pd
from constants.constants_tiktok import (TIKTOK_DOWNLOAD_VIDEO_DIR,
                                     TIKTOK_TRANSCRIBED_VIDEOS_PATH,
                                     YDL_OPTS)
from constants.constants_nlp import WHISPER_DEVICE, WHISPER_MODEL_VERSION

model = whisper.load_model(WHISPER_MODEL_VERSION, WHISPER_DEVICE)

def descargar_video(url: str):
    """Devuelve el nombre del video descargado y 
    
    Devuelve None si el video no fue descargado
    """

    filename = None # nombre del video descargado
    with yt_dlp.YoutubeDL(YDL_OPTS) as ydl:
        try:
            # Descargar y obtener información del video
            info = ydl.extract_info(url, download=True)
            # Generar nombre del archivo
            filename = f"{info.get('id')}.{info.get('ext')}"
        except yt_dlp.utils.DownloadError as e: # Posiblemente el video ya no existe
            print("Posiblemente el video no existe: ", url)
        except Exception as e: # Capturar otras posibles excepciones
            print("Error al descargar el video: ", url)
        return filename
    
def download_and_trancribe(row) -> bool:
    """Descargar y transcribir videos.
    Los videos descargados se guardan de forma persistente.
    
    Si una url no existe, se llenan los campos con valores vacios.
    
    El número de elementos guardados es igual a end-start si no hay
    interrupciones.

    :param row: Fila de un DataFrame
    """

    url = row['url']
    # Descargar y guardar video
    filename = descargar_video(url)
    df = None
    text = ''
    downloaded = False
    lang = ''
    if filename: # Descarga exitosa
        path = os.path.join(TIKTOK_DOWNLOAD_VIDEO_DIR, filename)
        # Transcribir el video
        try:
            print("Transcribiendo: ", filename)
            result = model.transcribe(path)
            text = result['text']
            lang = result['language']
            downloaded = True
        except RuntimeError as e:
            print("Runtime error al transcribir el video")
        except Exception as e:
            print("Error al transcribir el video")
        # Eliminar el video descargado
        try:
            os.remove(path)
        except FileNotFoundError:
            print("No se encontró el archivo ", url)
        except PermissionError:
            print("Sin permisos para eliminar el archivo ", url)
    # Guardar transcripcion
    df = pd.DataFrame({
        "date": [row['date']],
        "fecha_recuperacion": [row['fecha_recuperacion']],
        "termino": [row["termino"]],
        "vistas": [row["views"]],
        "url": [url],
        "titulo": [row["title"]],
        "hashtags": [row["hashtags"]],
        'descargado': [downloaded],
        'text': [text],
        'idioma': [lang]
    })
    df.to_csv(TIKTOK_TRANSCRIBED_VIDEOS_PATH, mode='a', index=False, header=None)
    return True
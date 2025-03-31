import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -- RUTAS DE SCRAPING
TIKTOK_SCRAPED_DIR = os.path.join(ROOT_DIR, "data", "tiktok", "scraped")
TIKTOK_MERGED_SCRAPED_PATH = os.path.join(ROOT_DIR, "data", "tiktok", "raw","tiktok_merged_scraped.csv")

# .. RUTAS DE DESCARGA Y TRANSCRIPCION
# Directorio donde serán descargados los videos
TIKTOK_DOWNLOAD_VIDEO_DIR = os.path.join(ROOT_DIR, "data", "tiktok", "transcribed")
# Ruta del archivo con las transcripciones
TIKTOK_TRANSCRIBED_VIDEOS_PATH = os.path.join(ROOT_DIR, "data", "tiktok", "transcribed", "tiktok_transcribed.csv")

# yt_dlp config
YDL_OPTS = {
    'format': 'bestaudio/bestvideo/best',  # Orden de preferencia separado por '/'
    'outtmpl': f'{TIKTOK_DOWNLOAD_VIDEO_DIR}/%(id)s.%(ext)s' # Formato de archivos descargados 
}
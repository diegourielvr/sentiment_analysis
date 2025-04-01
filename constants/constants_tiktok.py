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


# -- RUTAS DE LIMPIEZA
TIKTOK_PRE_TRANSLATED_SENTENCES = os.path.join(ROOT_DIR, 'data','tiktok', 'raw','tiktok_pre_translated_sentences.csv')
TIKTOK_PRE_TRANSLATED_ONLY_SENTENCES = os.path.join(ROOT_DIR, 'data','tiktok', 'raw','tiktok_pre_translated_only_sentences')
TIKTOK_TRANSLATED_ONLY_SENTENCES = os.path.join(ROOT_DIR, "data","tiktok","raw","tiktok_translated_only_sentences.txt")

# Rutas de archivos para obtener poalridad
TIKTOK_PRE_SENTIMENT_SENTENCES = os.path.join(ROOT_DIR, "data","tiktok","raw","tiktok_pre_sentiment_sentences.csv")
TIKTOK_PRE_SENTIMENT_TEXT = os.path.join(ROOT_DIR, "data","tiktok","raw","tiktok_pre_sentiment_text.csv")

TIKTOK_SENTIMENT_SENTENCES = os.path.join(ROOT_DIR, "data","tiktok","raw","tiktok_sentiment_sentences.csv")
TIKTOK_SENTIMENT_TEXT = os.path.join(ROOT_DIR, "data","tiktok","raw","tiktok_sentiment_text.csv")

# Datasets para entrenamiento
TIKTOK_DATASET_SENTENCES = os.path.join(ROOT_DIR, "data","tiktok","clean","tiktok_dataset_sentences.csv")
TIKTOK_DATASET_TEXT = os.path.join(ROOT_DIR, "data","tiktok","clean","tiktok_dataset_text.csv")

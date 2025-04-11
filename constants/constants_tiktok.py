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

# Modelos de W2V
EMBEDDING_W2V_TIKTOK_SENTENCES_PATH = os.path.join(ROOT_DIR, "models", "tiktok", "embeddings", "w2v_embeddings_tiktok_sentences")

# Models path
SVM_PIPELINE_PATH = os.path.join(ROOT_DIR, "models","tiktok","classifiers","svm_pipeline.pkl")
NB_PIPELINE_PATH = os.path.join(ROOT_DIR, "models","tiktok","classifiers","nb_pipeline.pkl")
LR_PIPELINE_PATH = os.path.join(ROOT_DIR, "models","tiktok","classifiers", "lr_pipeline.pkl")
SWEM_MODEL_PATH = os.path.join(ROOT_DIR, "models","tiktok","classifiers", "swem_model.pkl")
MLP_SWEM_MODEL_PATH = os.path.join(ROOT_DIR, "models","tiktok","classifiers", "mlp_swem_model.pth")
RNN_MODEL_PATH = os.path.join(ROOT_DIR, "models","tiktok","classifiers", "rnn_model.pth")
LSTM_MODEL_PATH = os.path.join(ROOT_DIR, "models","tiktok","classifiers", "lstm_model.pth")

# Metrics Path
TIKTOK_SVM_METRICS_PATH = os.path.join(ROOT_DIR, "results","tiktok","metrics","tiktok_svm_metrics.csv")
TIKTOK_NB_METRICS_PATH = os.path.join(ROOT_DIR, "results","tiktok","metrics","tiktok_nb_metrics.csv")
TIKTOK_LR_METRICS_PATH = os.path.join(ROOT_DIR, "results","tiktok","metrics","tiktok_lr_metrics.csv")
TIKTOK_SWEM_METRICS_PATH = os.path.join(ROOT_DIR, "results","tiktok","metrics","tiktok_swem_metrics.csv")
TIKTOK_MLP_SWEM_METRICS_PATH = os.path.join(ROOT_DIR, "results","tiktok","metrics","tiktok_mlp_swem_metrics.csv")
TIKTOK_RNN_METRICS_PATH = os.path.join(ROOT_DIR, "results","tiktok","metrics","tiktok_rnn_metrics.csv")
TIKTOK_LSTM_METRICS_PATH = os.path.join(ROOT_DIR, "results","tiktok","metrics","tiktok_lstm_metrics.csv")

# Loss curves
MLP_SWEM_LOSS_CURVES_DIR = os.path.join(ROOT_DIR, "results","tiktok","loss_curves", "mlp_swem")
RNN_LOSS_CURVES_DIR = os.path.join(ROOT_DIR, "results","tiktok","loss_curves", "rnn")
LSTM_LOSS_CURVES_DIR = os.path.join(ROOT_DIR, "results","tiktok","loss_curves", "lstm")
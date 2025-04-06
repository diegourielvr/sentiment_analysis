import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# SYMSPELL
SYMSPELL_DICTIONARIES_PATHS = {
    "en": os.path.join(ROOT_DIR, "data", "dictionaries", "en-80k.txt"),
    "es": os.path.join(ROOT_DIR, "data", "dictionaries", "es-100l.txt")
}

# -- SPACY
SPACY_NAME_MODELS = {
    "en": "en_core_web_sm",
    "es": "es_core_news_sm"
}

# -- WHISPER
WHISPER_DEVICE = "cpu" # "cpu" | "cuda"
WHISPER_MODEL_VERSION = "small" # tiny | base | small | medium | large


POLARITY_MAP = {
    "NEG": 0,
    "NEU": 1,
    "POS": 2
}

INDEX_TO_POLARITY = {
    0: "NEG",
    1: "NEU",
    2: "POS"
}

SEED = 42
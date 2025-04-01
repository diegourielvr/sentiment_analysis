import re
import unicodedata
import emoji
import pandas as pd 

from src.preprocesamiento.spell import spell

# from src.preprocesamiento.spell import spell
# from src.nlp.nlp_spacy import get_no_stopwords

def replace_nbsp(texto: str):
    """Reemplazar el símbolo &nbsp; por un espacio regular
    """
    
    return re.sub(r'[\xa0]', ' ', str(texto))

def replace_whitespaces(text: str):
    """Reemplazar espacios consecutivos, al inicio, final o intermedios por uno solo
    """
    
    return re.sub(r'\s+', ' ', text).strip()

def replace_quotes(text:str):
    """Reemplazar comillas dobles por comillas simples
    """

    texto = re.sub(r'"', "'", text)
    texto = re.sub(r'[“”]', "'", texto)  # Para comillas curvas
    texto = re.sub(r'&quot;', "'", texto)  # Para la entidad HTML
    return texto

def replace_url(text:str, value='url'):
    """Reemplazar una url por otro valor
    """
    
    return re.sub(r'https?://\S+|www\.\S+', value, text)

def reduce_repeated_punct(text):
    """Reducir signos de puntuación repetidos
    """
    
    # Reducir signos de puntuación repetidos
    return re.sub(r'([^\w\s])\1+', r'\1', text)

def remove_zwj(text):
    """Eliminar simbolo ZWJ (Zero Width Joiner)
    """
    
    return re.sub(r'\u200D', '', text)

def emoji_to_text(text: str, lang='es'):
    """Convertir emojis visibles a texto
    """

    new_text =  emoji.demojize(text, language=lang)
    return " ".join(new_text.split("_"))

def normalize_text(text: str):
    """Convertir texto en negritas, cursivas a su forma normal y
    Unír unir acentos
    """

    # Eliminar negritas, cursivas, separar acentos, etc.
    text = unicodedata.normalize("NFKD", str(text))

    # Unir acentos para evitar problemas
    text = unicodedata.normalize("NFC", text)
    return text

def drop_non_letters_only_rows(df, col):
    """Elimina filas donde la columna especificada solo contenga valores diferentes
    a letras, números y guines bajos.
    """

    return df[~df[col].str.fullmatch(r'\W+', na=False)]

def drop_blank_or_nan_or_duplicated(df, col):
    """Eliminar filas con cadenas vacías, espacios, tabulaciones, NA o valores duplicados
    """

    new_df = df[~df[col].str.match(r'^\s*$') & df[col].notna()]
    return new_df.drop_duplicates(subset=col, keep='first').reset_index(drop=True)

def remove_html(text: str):
    clean_text = re.sub(r'<[^>]+>', '', text)
    return clean_text

def remove_punctuation(text: str):
    return re.sub(r'[!"#&\'*+,-./<=>?@:;_%&¿<>()[\\]^`{|}~]', '', text)

def filter_text(text: str):
    pattern = re.compile(r'([a-zA-ZáéíóúüÁÉÍÓÚÜñÑ0-9]+)', re.UNICODE)
    return ' '.join(pattern.findall(text))

def clean_text(text: str, lang="es"):
    """Mantienen letras en español, acentos y algunos signos de puntuación.
    Elimina emojis
    """
    # Convertir a minúsculas
    text = text.lower()
    # Normalizar texto en negritas, italicas, etc
    text = normalize_text(text)
    # Reemplazar urls
    text = replace_url(text, " ")
    # Eliminar etiquetas html
    text = remove_html(text)
    # Eliminar signos de puntuación
    text = remove_punctuation(text)
    # Filtrar palabras con letras del español (elimina, emojis, símbolos espaciales, etc)
    text = filter_text(text)
    # Eliminar espacios, tabulaciones, etc consecutivos
    text = replace_whitespaces(text)
    # Corrección ortográfica
    text = spell(text, lang)
    # Remover stopwords
    # text = get_no_stopwords(text, lang)
    return text
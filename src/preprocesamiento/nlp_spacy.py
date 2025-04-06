from matplotlib.pyplot import get
import spacy
from tqdm import tqdm

from constants.constants_nlp import SPACY_NAME_MODELS
from src.preprocesamiento.nlp_nltk import stemming_pipe

# nlp = spacy.load(SPACY_NAME_MODELS[SPACY_DEFAULT_LANG_MODEL])
models = {
    "es": spacy.load(SPACY_NAME_MODELS["es"]),
    "en": spacy.load(SPACY_NAME_MODELS["en"])
}

def get_model(lang: str="es"):
    """Cambiar el idioma del modelo de spacy
    :param lang: 'es' o 'en'
    """

    if lang not in SPACY_NAME_MODELS.keys():
        print(f"Idioma no soportado: {lang}")
        return None
    print(f"Modelo cargado: {SPACY_NAME_MODELS[lang]}")
    # return spacy.load(SPACY_NAME_MODELS[lang])
    return models[lang]

def get_sentences_with_ids(documentos: list[str], lang="es", max_chars: int=4900):
    """ Devuelve una lista de oraciones para cada documento
    Ej. [['oracion 1 doc 1', 'oracion 2 doc 1'], ['oracion 1 doc 2']]
    """
    
    nlp = get_model(lang)
    docs = nlp.pipe(documentos)
    sentence_data = []
    for doc_id, doc in tqdm(enumerate(docs)):
        sentence_id = 0
        for sentence in doc.sents:
            sentence_text = sentence.text.strip()
            
            while len(sentence_text) > max_chars:
                # Obtener la posición del último espacio antes de max_chars
                split_pos = sentence_text.rfind(" ", 0, max_chars)
                # Si no se encuentra un espacio, tomar el maximo de caracteres
                if split_pos == -1: split_pos = max_chars
                
                sentence_data.append({
                    'doc_id': doc_id,
                    'sentence_id': sentence_id,
                    'sentence': sentence_text[:split_pos].strip()
                })
                # Seguir con el resto del texto
                sentence_text = sentence_text[split_pos:].strip() 
                sentence_id += 1 # Aumentar para la siguiente oración
                
            if sentence_text:
                sentence_data.append({
                    'doc_id': doc_id,
                    'sentence_id': sentence_id,
                    'sentence': sentence_text
                })
                sentence_id += 1 # Automar para la siguiente oración
    return sentence_data

def reconstruct_text_from_df(sentences_df,col_doc_id='doc_id', col_sentence_id='sentence_id',col_sentence='sentence'):
    # Ordenar por doc_id y sentence_id
    sentences_df_sorted = sentences_df.sort_values(by=[col_doc_id, col_sentence_id])
    
    # Agrupar las oraciones por doc_id y reconstruir el texto
    reconstructed_texts = sentences_df_sorted.groupby(col_doc_id)[col_sentence].apply(lambda x: ' '.join(x)).reset_index()
    
    return reconstructed_texts

def get_no_stopwords_pipe(documentos: list[str], sep = ' ', lang: str="es"):
    """Devuelve las palabras de un documento que no son stopwords"""
    nlp = get_model(lang)
    docs = nlp.pipe(documentos)
    docs_no_stopwords = []
    for doc in tqdm(docs):
        no_stopwords = []
        for token in doc:
            if not token.is_stop:
                no_stopwords.append(token.text)
        docs_no_stopwords.append(sep.join(no_stopwords))
    return docs_no_stopwords

def preprocesamiento(documentos: list[str], stemming=True, lang="es", progress=False) -> list[str]:
    """Procesamiento de lenguaje natural del texto.
    Tokenizar, eliminar signos de puntuacion, stopwords y lematizar.

    :param documentos: Lista de textos
    :type documentos: list[str]
    :param lang: idioma de los textos
    :type lang: str
    :return: Lista de lemas de cada documento
    :rtype: list[list[str]]
    """
    nlp = get_model(lang)

    documentos_preprocesados = []
    
    if progress: documentos = tqdm(documentos, leave=True, desc="nlp.pipe")
    docs = nlp.pipe(documentos)
    
    for doc in docs: # procesamiento de documentos
        doc_preprocesado = []
        for token in doc:
            # filtrar lemas de palabras que no sean stopwords o signos de puntuacion
            if not token.is_stop and not token.is_punct:
                doc_preprocesado.append(token.lemma_)
        documentos_preprocesados.append(doc_preprocesado)

    if stemming:
        print("Aplicando stemming...")
        documentos_preprocesados = stemming_pipe(documentos_preprocesados, lang)
    
    # devolver stems separados por espacios
    documentos_preprocesados = [" ".join(docs) for docs in documentos_preprocesados]
    
    print(f"Total de documentos preprocesados: {len(documentos_preprocesados)}")
    return documentos_preprocesados

class Tokenizer:
    def __init__(self, lang: str="es"):
        self.nlp = get_model(lang)
        
    def tokenize(self, text_list: list[str], progress=False):
        if progress:
            text_list = tqdm(text_list)
        docs = self.nlp.pipe(text_list)
        return list(
            map(
                lambda doc: [token.text for token in doc], 
                docs
            )
        )
        
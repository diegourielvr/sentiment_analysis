from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from src.trainers.utils import get_metrics

import time

def train_nb(
    dataset_train, dataset_val,
    alpha: float = 1, vec: str = "tfidf"):
    """
    alpha: un valor alto indica que las características no conocidas toman una probabilidad alta.
    por el contrario, un valor pequeño asigna probabilidades pequeñas a caracteristicas no conocidas
    """

    vectorizer = None
    if vec.lower() == "tfidf": vectorizer = TfidfVectorizer()
    else: vectorizer = CountVectorizer()
        
    pipeline = Pipeline([
        ("vectorizer", vectorizer),
        ("nb", MultinomialNB(alpha=alpha))
    ])
    
    # Dividir datos
    x_train, y_train = dataset_train['text'], dataset_train['polarity']

    # Entrenar modelo
    start = time.time()
    pipeline.fit(x_train, y_train)
    end = time.time()

    # Dividir datos
    x_val, y_val = dataset_val['text'], dataset_val['polarity']

    # Evaluar modelo
    y_pred = pipeline.predict(x_val)
    metrics = get_metrics(y_val, y_pred)
    
    metrics['model'] = "NB"
    metrics['vectorizer'] = vec
    metrics['alpha'] = alpha
    metrics['vocab_size'] = len(pipeline.named_steps['vectorizer'].vocabulary_)
    metrics['train_time'] = end - start

    return pipeline, metrics
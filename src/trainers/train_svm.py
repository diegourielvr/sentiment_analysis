from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC

from src.trainers.utils import get_metrics

import time

def train_svm(
    dataset_train, dataset_val,
    C: float = 1.0, kernel: str = "linear",
    vec: str = "tfidf"):

    vectorizer = None
    if vec.lower() == "tfidf": vectorizer = TfidfVectorizer()
    else: vectorizer = CountVectorizer()
        
    pipeline = Pipeline([
        ("vectorizer", vectorizer),
        ("svm", SVC(kernel=kernel, C=C))
    ])
    
    # Dividir datos
    x_train, y_train = dataset_train['text'], dataset_train['polarity']
    # Entrenar modelo
    start = time.perf_counter()
    pipeline.fit(x_train, y_train)
    end = time.perf_counter()

    # Dividir datos
    x_val, y_val = dataset_val['text'], dataset_val['polarity']
    # Evaluar modelo
    y_pred = pipeline.predict(x_val)
    metrics, _ = get_metrics(y_val, y_pred)
    
    metrics['model'] = "SVM"
    metrics['vectorizer'] = vec.upper()
    metrics['dataset'] = "VAL" # validation dataset
    metrics['regularization'] = C
    metrics['kernel'] = kernel.upper()
    metrics['vocab_size'] = len(pipeline.named_steps['vectorizer'].vocabulary_)
    metrics['train_time'] = end - start

    return pipeline, metrics
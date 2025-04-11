from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from src.trainers.utils import get_metrics

import time

def train_svm(
    dataset_train, dataset_val,
    C: float = 1.0, kernel: str = "linear",
    vec: str = "tfidf", class_weight=None):
    """
    Valores pequeños de C penalizan poco el error y da un mayor margen
    Valores grandes de C dan mayor penalización al error y da un menor margen

    Escoger kernel 'linear' si los datos son linealmente separables,
    si no lo son, entonces usar 'rbf'.
    
    class_weight=None si todas las clases tiene le mismo peso
    class_weight='balanced' para compenzar las clases con pocos ejemplos
    """

    vectorizer = None
    if vec.lower() == "tfidf": vectorizer = TfidfVectorizer()
    else: vectorizer = CountVectorizer()
        
    pipeline = Pipeline([
        ("vectorizer", vectorizer),
        ("svm", SVC(kernel=kernel, C=C, class_weight=class_weight))
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
    
    metrics['model'] = "SVM"
    metrics['vectorizer'] = vec.upper()
    metrics['regularization'] = C
    metrics['kernel'] = kernel.upper()
    metrics['vocab_size'] = len(pipeline.named_steps['vectorizer'].vocabulary_)
    metrics['train_time'] = end - start

    return pipeline, metrics
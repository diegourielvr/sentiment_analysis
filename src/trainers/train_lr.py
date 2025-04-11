from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression

from src.trainers.utils import get_metrics

import time

def train_lr(
    dataset_train, dataset_val, vec = "tfidf",
    penalty = "l2", C = 1, solver = "lbfgs",
    max_iter=100, classs_weight=None):

    """
    C >= 0
    solver 'lbfgs' | 'saga'
    """

    vectorizer = None
    if vec.lower() == "tfidf": vectorizer = TfidfVectorizer()
    else: vectorizer = CountVectorizer()
        
    pipeline = Pipeline([
        ("vectorizer", vectorizer),
        ("lr", LogisticRegression(
            penalty=penalty, C=C,
            max_iter=max_iter,
            multi_class="multinomial", # n_classes >= 3 
            solver=solver,
            random_state=42,
            class_weight=classs_weight
        ))
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
    
    metrics['model'] = "LR"
    metrics['vectorizer'] = vec.upper()
    metrics['penalty'] = penalty
    metrics['regularization'] = C
    metrics['max_iter'] = max_iter
    metrics['solver'] = solver
    metrics['vocab_size'] = len(pipeline.named_steps['vectorizer'].vocabulary_)
    metrics['train_time'] = end - start

    return pipeline, metrics
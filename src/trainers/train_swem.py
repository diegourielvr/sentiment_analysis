import time
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from src.trainers.utils import EmbeddingLoader, get_metrics, show_confusion_matrix, apply_pooling

def train_swem(
    dataset_train, dataset_val, 
    embeddings_path, pooling="aver",
    classifier="svm", model_args=None):
    
    # Dividir Información
    x_train_tokenized, y_train = dataset_train['tokens'], dataset_train['polarity']
    x_val_tokenized, y_val = dataset_val['tokens'], dataset_val['polarity']

    # Cargar modelo y obtener embeddings
    embedding_model = EmbeddingLoader(f"{embeddings_path}.bin")
    x_train_embeddings = embedding_model.get_embeddings(x_train_tokenized)
    x_val_embeddings = embedding_model.get_embeddings(x_val_tokenized)

    # Operación de agrupación
    x_train_pooling = apply_pooling(x_train_embeddings, embedding_model.vector_size(), pooling)
    x_val_pooling = apply_pooling(x_val_embeddings, embedding_model.vector_size(), pooling)

    # Seleccionar modelo de clasificación
    classifier_model = None
    if classifier.lower() == "svm":
        classifier_model =  SVC(
            kernel=model_args.kernel,
            C=model_args.C
        )
    elif classifier.lower() == "lr":
        classifier_model = LogisticRegression(
            penalty=model_args.penalty,
            C=model_args.C,
            max_iter=model_args.max_iter,
            multi_class="multinomial", # n_classes >= 3 
            solver=model_args.solver,
            random_state=42
        )

    # Entrenar modelo
    start = time.time()
    classifier_model.fit(x_train_pooling, y_train)
    end = time.time()

    # Evaluar modelo
    y_pred = classifier_model.predict(x_val_pooling)
    metrics = get_metrics(y_val, y_pred)
    
    metrics['model'] = "SWEM"
    metrics['classifier'] = classifier
    metrics['pooling'] = pooling
    metrics['penalty'] = model_args.penalty if hasattr(model_args,"penalty") else None
    metrics['kernel'] = model_args.kernel if hasattr(model_args,"kernel") else None
    metrics['regularization'] = model_args.C
    metrics['max_iter'] = model_args.max_iter if hasattr(model_args, "max_iter") else None
    metrics['solver'] = model_args.solver if hasattr(model_args, "solver") else None
    metrics['embedding_dim'] = embedding_model.vector_size
    metrics['train_time'] = end - start

    return classifier_model, metrics

def evaluate_model(model, dataset, title, embeddings_path, pooling):
    x_tokenized, y_true = dataset['tokens'], dataset['polarity']
    embedding_model = EmbeddingLoader(f"{embeddings_path}.bin")
    x_embeddings = embedding_model.get_embeddings(x_tokenized)
    x_pooling = apply_pooling(x_embeddings, embedding_model.vector_size(), pooling)

    y_pred = model.predict(x_pooling)
    metrics = get_metrics(y_true, y_pred)
    show_confusion_matrix(y_pred, y_true, title)
    return metrics
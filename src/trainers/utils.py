import gensim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from constants.constants_nlp import POLARITY_MAP

import matplotlib.pyplot as plt

import joblib
import os

def get_metrics(y_true, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, average="macro"),
        "precision": precision_score(y_true, y_pred, average="macro"),
        "f1_score": f1_score(y_true, y_pred, average="macro")
    }
    
    return metrics

def show_confusion_matrix(y_true, y_pred, title="", path=None, dpi=300):
    print(f"\n{title}")
    cr = classification_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("Reporte de clasificacion")
    print(cr)
    
    print("Matriz de confusión")
    labels = None
    if cm.shape[0] == 3:
        labels = list(POLARITY_MAP.keys())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Oranges")
    plt.title(title)
    plt.show()
    if path:
        plt.savefig(path, dpi)
        print(f"Imagen guardada en: {path}")

def build_datasets(
    data_path: str, text = 'text', label = 'polarity',
    test_size:float = 0.3, val_size:float = 0.0,
    random_state: int = 42):
    
    """
    Si val_dataset=True, se toma el porcentaje especificado val_size
    del conjunto de datos de prueba.
    """
    
    df = pd.read_csv(data_path, encoding="utf-8")
    df = df[[text, label]]

    # Separar los datos
    labels = df[label].unique()
    df_clases = []
    for label_name in labels:
        df_clases.append(df[df[label] == label_name])
    
    min_size = min([len(df_clase) for df_clase in df_clases])
    
    # Tomar muestras balanceadas
    df_sampled = []
    for df_clase in df_clases:
        df_sampled.append(df_clase.sample(min_size, random_state=random_state))
    
    # Unir los datos
    df_combined = pd.concat(df_sampled)

    # Mezclar datos
    df_combined = df_combined.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Dividr datos
    train_df, test_df, val_df = None, None, None
    train_df, test_df = train_test_split(
        df_combined,
        test_size=test_size,
        random_state=random_state,
        stratify=df_combined[label])
    
    if val_size:
        val_df, test_df = train_test_split(
            test_df,
            test_size=val_size,
            random_state=random_state,
            stratify=test_df[label])
    
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), val_df.reset_index(drop=True)

def save_model(model, path):
    joblib.dump(model, path)
    print(f"Modelo guardado en: {path}")

def load_model(path):
    print(f"Cargando modelo: {path}")
    return joblib.load(path)

def save_metrics(metrics, path: str):
    row = pd.DataFrame([metrics])
    row.to_csv(path, index=False, mode="a", header=not os.path.exists(path))

def evaluate_model(model, dataset, title):
    x, y_true = dataset['text'], dataset['polarity']
    y_pred = model.predict(x)
    metrics = get_metrics(y_true, y_pred)
    show_confusion_matrix(y_pred, y_true, title)
    return metrics

def show_losses(train_losses, val_losses):
    x = range(1, len(train_losses)+1)
    plt.plot(x, train_losses, label="Pérdida Entrenamiento")
    plt.plot(x, val_losses, label="Pérdida Validación", linestyle="dashed")
    plt.xlabel("Épocas")
    plt.ylabel("Log Loss")
    plt.title("Evolución de la Pérdida")
    plt.legend()
    plt.show()

class EmbeddingLoader:
    def __init__(self, embeddings_path: str, type: str='w2v'):
        self.word_vectors = None
        if type == 'w2v':
            self.word_vectors = gensim.models.KeyedVectors.load_word2vec_format(embeddings_path, binary=True)
    
    def get_word_vectors(self):
        return self.word_vectors
            
    def vector_size(self):
        return self.word_vectors.vector_size
    
    def get_embedding(self, word: str):
        if word in self.word_vectors:
            return self.word_vectors[word] # numpy.ndarray
        return np.zeros(self.word_vectors.vector_size)

    def get_embeddings(self, tokens_list: list[list[str]]):
        embeddings = []
        for tokens in tokens_list:
            embeddings.append(np.array([self.get_embedding(token) for token in tokens]))
        return embeddings

class LRModelArgs:
    penalty: str
    C: float
    max_iter: int
    solver: str

class SVMModelArgs:
    kernel: str
    C: float

def order_dataset(data: pd.DataFrame, text_col="text", ascending=False):
    # Ordenar por número de palabras (de mayor a menor)
    data['num_words'] = data[text_col].apply(lambda x: len(str(x).split()))
    data = data.sort_values(by='num_words', ascending=ascending).drop(columns=['num_words'])
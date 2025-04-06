import gensim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from constants.constants_nlp import POLARITY_MAP, SEED

import matplotlib.pyplot as plt

import torch
import joblib
import os

class CustomDataset:
    def __init__(self, embeddings: list[np.ndarray], labels: list[int]):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        x = self.embeddings[idx]
        y = self.labels[idx]
        return x, y

def aver_pooling(embeddings: np.array, embeddings_dim):
    if not embeddings is None:
        return np.mean(embeddings, axis=0)
    return np.zeros(embeddings_dim)

def max_pooling(embeddings: np.array, embeddings_dim):
    if not embeddings is None:
        return np.max(embeddings, axis=0)
    return np.zeros(embeddings_dim)

def apply_pooling(list_embeddings: list[np.ndarray], embeddings_dim, pooling="aver"):
    x_pooling = None
    if pooling== "aver":
        x_pooling = np.array(
            [aver_pooling(embeddings, embeddings_dim) for embeddings in list_embeddings]
        )
    else: # pooling='max'
        x_pooling = np.array(
            [max_pooling(embeddings, embeddings_dim) for embeddings in list_embeddings]
        )
    return x_pooling

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
        plt.savefig(path, dpi=dpi)
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

def save_model_torch(model, path):
    torch.save(model.state_dict(), path)
    print(f"Modelo guardado en: {path}")

def load_model_torch(model, path):
    """
    model = MLPModelCustom()
    """
    print(f"Cargando modelo: {path}")
    model.load_state_dict(torch.load(path))
    model.eval()  # Inferencia
    return model

def evaluate_model(model, dataset, title):
    x, y_true = dataset['text'], dataset['polarity']
    y_pred = model.predict(x)
    metrics = get_metrics(y_true, y_pred)
    show_confusion_matrix(y_pred, y_true, title)
    return metrics

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
        return embeddings # list[ndarray[ndarray[float]]]

class LRModelArgs:
    penalty: str
    C: float
    max_iter: int
    solver: str

class SVMModelArgs:
    kernel: str
    C: float

class ModelArgs:
    input_size: int
    hidden_size: int # rnn
    num_layers: int = 1 # rnn
    hidden_layers: list # mlp
    output_size: int
    nonlinearity: str = 'tanh' # 'tanh' | 'relu'
    dropout:int = 0

def show_loss_val_curves(train_losses, val_losses, epochs, path:str="", dpi:int=300):
    x = range(1, epochs + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(x, train_losses, label="Entrenamiento")
    plt.plot(x, val_losses, label="Validación")
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.title("Evolución de la Pérdida en Entrenamiento y Validación")
    plt.legend()
    plt.grid()
    if path:
        plt.savefig(path, dpi=dpi)
    else:
        plt.show()

class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0
        
    def get_patience(self):
        return self.patience

    def get_min_delta(self):
        return self.min_delta
    
    def __call__(self, val_loss):
        if self.patience == None: return False # Ignorar early stopping
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0 # Reiniciar contador de paciencia
        
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def create_dataloder_from_embeddings(embeddings, labels, batch_size, custom_collate_fn):
    dataset = CustomDataset(embeddings, labels)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        generator=torch.Generator().manual_seed(SEED)
    )
    return dataloader

import random

def set_seed():
    seed = SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # Hace que cuDNN use algoritmos deterministas
    torch.backends.cudnn.benchmark = False     # Desactiva la búsqueda automática de algoritmos

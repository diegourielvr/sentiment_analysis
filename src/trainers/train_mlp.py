import torch
import time
import numpy as np
from tqdm import tqdm
from constants.constants_nlp import POLARITY_MAP
from sklearn.utils.class_weight import compute_class_weight
from src.trainers.utils import EmbeddingLoader, ModelArgs, create_dataloder_from_embeddings, get_metrics, show_confusion_matrix, apply_pooling

def collate_fn(batch):
    x_batch, y_batch = zip(*batch)  # Separar x e y
    x_batch = torch.tensor(np.array(x_batch), dtype=torch.float32)
    y_batch = torch.tensor(np.array(y_batch), dtype=torch.long)
    return x_batch, y_batch

class MLPModelCustom(torch.nn.Module):
    def __init__ (self, model_args: ModelArgs):
        super().__init__()
        layers = []
        previous_dim = model_args.input_size
        for hidden_dim in model_args.hidden_layers:
            layers.append(torch.nn.Linear(previous_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(model_args.dropout))
            previous_dim = hidden_dim
        layers.append(torch.nn.Linear(previous_dim, model_args.output_size))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
class Trainer:
    def __init__(self, model, lr=0.001, optim="adam", class_weights=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        
        # self.model = MLPModel(model_args.input_size, model_args.hidden_size, model_args.output_size)
        self.model = model
        if not class_weights is None:
            class_weights = class_weights.to(self.device)
        self.cost_function = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = None
        if optim.lower() == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        if optim.lower() == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
    
    def fit(self, dataloader_train, dataloader_val, early_stopping, epochs=50):
        # Mover el modelo al device disponible
        self.model.to(self.device)

        train_losses = []
        val_losses = []
        
        progress_bar = tqdm(range(epochs), leave=True)
        for epoch in progress_bar:
            self.model.train() # Modo de entrenamiento

            # progress_bar = tqdm(dataloader_train, desc=f"Época {epoch+1}/{epochs}", leave=True)
            train_loss = 0
            num_train_samples = 0
            
            for batch_x, batch_y in dataloader_train:
                # Mover los datos al dispositivo disopnible
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward propagation
                output = self.model(batch_x)
                loss = self.cost_function(output, batch_y)

                # Backward propagation
                self.optimizer.zero_grad() # Reiniciar gradientes en cero
                loss.backward() # Calcular gradientes
                self.optimizer.step()# Actualizar parámetros
                
                local_batch_size = batch_x.size(0)
                train_loss += loss.item() * local_batch_size # Promedio ponderado
                num_train_samples += local_batch_size
                
            train_loss /= num_train_samples
            train_losses.append(train_loss)

            # Validation
            self.model.eval()
            val_loss = 0
            num_val_samples = 0

            with torch.torch.no_grad():
                for batch_x, batch_y in dataloader_val:
                    # Mover al device disponible
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    # Forward propagation
                    output = self.model(batch_x) # (batch_size, output_size)
                    loss = self.cost_function(output, batch_y)

                    local_batch_size = batch_x.size(0)
                    val_loss += loss.item() * local_batch_size # Promedio ponderado
                    num_val_samples += local_batch_size
            val_loss /= num_val_samples
            val_losses.append(val_loss)

            progress_bar.set_postfix(loss=val_loss)
            
            if early_stopping(val_loss):
                return train_losses, val_losses

        return train_losses, val_losses

    def predict(self, dataloader):
        self.model.eval() # Desactivar dropout
        with torch.torch.no_grad(): # no calcular gradiente
            predictions = []
            for batch_x, batch_y in dataloader:
                # Mover al device disponible
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward propagation
                output = self.model(batch_x) # (batch_size, output_size)
                preds = torch.argmax(output, dim=1)
                predictions.extend(preds.cpu().numpy())
        return np.array(predictions)   

    def get_model(self):
        return self.model

def train_mlp(
    dataset_train, dataset_val, embeddings_path,
    model_args, early_stopping, batch_size=64, lr=0.001,
    epochs=50, optim="adam", pooling="aver", use_class_weights=False):

    # Dividir información
    x_train_tokenized, y_train = dataset_train['tokens'], dataset_train['polarity']
    x_val_tokenized, y_val = dataset_val['tokens'], dataset_val['polarity']

    # Cargar modelo de embeddings
    embedding_model = EmbeddingLoader(f"{embeddings_path}.bin")
    model_args.input_size = embedding_model.vector_size()
    # Obtener embeddings
    train_embeddings = embedding_model.get_embeddings(x_train_tokenized) # list[ndarray[ndarray[float]]]
    val_embeddings = embedding_model.get_embeddings(x_val_tokenized)
    
    # Operación de agrupación
    x_train_pooling = apply_pooling(train_embeddings, embedding_model.vector_size(), pooling)
    x_val_pooling = apply_pooling(val_embeddings, embedding_model.vector_size(), pooling)
    
    # Crear datasets y dataloaders
    dataloader_train = create_dataloder_from_embeddings(
        x_train_pooling, y_train,batch_size, collate_fn, shuffle=True
    )
    dataloader_val = create_dataloder_from_embeddings(
        x_val_pooling,y_val, batch_size, collate_fn, shuffle=False
    )

    model = MLPModelCustom(model_args)
    class_weights = None
    if use_class_weights is True:
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float)

    trainer = Trainer(model, lr, optim, class_weights)

    start = time.time()
    train_losses, val_losses = trainer.fit(dataloader_train, dataloader_val, early_stopping, epochs)
    end = time.time()
    print(f"Pérdida Entrenamiento = {train_losses[-1]:.4f}, Pérdida Validación = {val_losses[-1]:.4f}")
    
    # Evaluar modelo
    y_pred = trainer.predict(dataloader_val)
    metrics = get_metrics(y_val, y_pred)

    metrics['model'] = "MLP SWEM"
    metrics['pooling'] = pooling
    metrics['optim'] = optim
    metrics['lr'] = lr 
    metrics['patience'] = early_stopping.get_patience() 
    metrics['min_delta'] = early_stopping.get_min_delta() 
    metrics['hidden_layers'] = model_args.hidden_layers
    metrics['output_size'] = model_args.output_size
    metrics['dropout'] = model_args.dropout
    metrics['epochs'] = epochs
    metrics['batch_size'] = batch_size
    metrics['embedding_dim'] = embedding_model.vector_size()
    metrics['train_time'] = end - start

    return trainer, metrics, train_losses, val_losses

def evaluate_model(model, dataset, title, embeddings_path, pooling="aver", batch_size=64):
    x_tokenized, y_true = dataset['tokens'], dataset['polarity']

    embedding_model = EmbeddingLoader(f"{embeddings_path}.bin")
    embeddings = embedding_model.get_embeddings(x_tokenized)
    x_pooling = apply_pooling(embeddings, embedding_model.vector_size(), pooling)

    dataloader = create_dataloder_from_embeddings(
        x_pooling, y_true, batch_size, collate_fn, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() # Desactivar dropout
    with torch.torch.no_grad(): # no calcular gradiente
        predictions = []
        for batch_x, batch_y in dataloader:
            # Mover al device disponible
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Forward propagation
            output = model(batch_x) # (batch_size, output_size)
            preds = torch.argmax(output, dim=1)
            predictions.extend(preds.cpu().numpy())
    y_pred = np.array(predictions)   
    
    metrics = get_metrics(y_true, y_pred)
    show_confusion_matrix(y_pred, y_true, title)
    return metrics

class SentimentAnalysis:
    def __init__(self, base_model, embeddings_path, tokenizer, device=None, pooling="aver"):
        self.model = base_model # MLPModelCustom()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval() # Desactiva Dropout
        self.tokenizer = tokenizer
        self.embedding_model = EmbeddingLoader(f"{embeddings_path}.bin")
        self.pooling=pooling
    
    def predict(self, x: str):
        x_tokenized = self.tokenizer.tokenize([x]) # list[list[str]
        x_embeddings = self.embedding_model.get_embeddings(x_tokenized)
        x_pooling = apply_pooling(x_embeddings, self.embedding_model.vector_size(), self.pooling)
        x_tensor = torch.tensor(np.array(x_pooling), dtype=torch.float32)
        
        with torch.torch.no_grad(): # no calcular gradiente
            x_tensor = x_tensor.to(self.device)
            output = self.model(x_tensor) # (batch_size, output_size)
            pred = torch.nn.functional.softmax(output, dim=1).squeeze()
            res = list(zip(list(POLARITY_MAP.keys()), pred.tolist()))
            
            return sorted(res, key=lambda x: x[1], reverse=True)
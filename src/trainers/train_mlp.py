import torch
import time
import numpy as np
from tqdm import tqdm
from src.trainers.utils import EmbeddingLoader, create_dataloder_from_embeddings, get_metrics, show_confusion_matrix, apply_pooling

def collate_fn(batch):
    x_batch, y_batch = zip(*batch)  # Separar x e y
    x_batch = torch.tensor(np.array(x_batch), dtype=torch.float32)
    y_batch = torch.tensor(np.array(y_batch), dtype=torch.long)
    return x_batch, y_batch

def collate_fn_rnn(batch):
    x_batch, y_batch = zip(*batch)  # Separar x e y

    # Convertir a tensores
    x_batch_tensor = torch.tensor(np.array(x_batch), dtype=torch.float32)
    y_batch_tensor = torch.tensor(np.array(y_batch), dtype=torch.long)
    
    # Obtener longitudes de cada ejemplo
    lengths = torch.tensor([len(seq) for seq in x_batch_tensor])

    # Agregar padding
    x_batch_padded = torch.nn.utils.rnn.pad_sequence(
        x_batch_tensor,
        batch_first=True,
        padding_value=0
    ) # (batch_size, sequence_length, embedding_dim)
    
    return x_batch_padded, lengths, y_batch_tensor
        
class MLPModel(torch.nn.Module):
    def __init__ (self, input_size, hidden_size, output_size):
        super().__init__()
        self.h1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.h2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = self.h1(x)
        a = self.relu(h)
        return self.h2(a)

class MLPModelCustom(torch.nn.Module):
    def __init__ (self, input_size, hidden_layers, output_size, dropout=0.5):
        super().__init__()
        layers = []
        previous_dim = input_size
        for hidden_dim in hidden_layers:
            layers.append(torch.nn.Linear(previous_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))
            previous_dim = hidden_dim
        layers.append(torch.nn.Linear(previous_dim, output_size))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
class MLP:
    def __init__(self, model_args, lr=0.001, optim="adam"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        
        # self.model = MLPModel(model_args.input_size, model_args.hidden_size, model_args.output_size)
        self.model = MLPModelCustom(
            model_args.input_size,
            model_args.hidden_layers,
            model_args.output_size,
            model_args.dropout
        )
        self.cost_function = torch.nn.CrossEntropyLoss()
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

                # Backward propaghation
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

            # print(f"Época {epoch+1}: Pérdida Entrenamiento = {train_loss:.4f}, Pérdida Validación = {val_loss:.4f}")
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

def train_swem_mlp(
    dataset_train, dataset_val, embeddings_path,
    model_args, early_stopping, batch_size=64, lr=0.001,
    epochs=50, optim="adam", pooling="aver", num_workers=0, pin_memory=False):

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
        x_train_pooling, y_train,batch_size,
        collate_fn, num_workers, pin_memory
    )
    dataloader_val = create_dataloder_from_embeddings(
        x_val_pooling,y_val, batch_size,
        collate_fn, num_workers, pin_memory
    )

    classifier_model = MLP(model_args, lr, optim)
    start = time.time()
    train_losses, val_losses = classifier_model.fit(dataloader_train, dataloader_val,early_stopping, epochs)
    end = time.time()
    
    # Evaluar modelo
    y_pred = classifier_model.predict(dataloader_val)
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

    return classifier_model, metrics, train_losses, val_losses

def evaluate_model(model, dataset, title, embeddings_path, pooling="aver", batch_size=64, num_workers=0, pin_memory=False):
    x_tokenized, y_true = dataset['tokens'], dataset['polarity']

    embedding_model = EmbeddingLoader(f"{embeddings_path}.bin")
    embeddings = embedding_model.get_embeddings(x_tokenized)
    x_pooling = apply_pooling(embeddings, embedding_model.vector_size(), pooling)

    dataloader= create_dataloder_from_embeddings(
        x_pooling, y_true, batch_size, 
        collate_fn, num_workers, pin_memory
    )

    y_pred = model.predict(dataloader)
    
    metrics = get_metrics(y_true, y_pred)
    show_confusion_matrix(y_pred, y_true, title)
    return metrics
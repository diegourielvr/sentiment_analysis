import torch
import numpy as np
from tqdm import tqdm
from src.trainers.utils import EmbeddingLoader, create_dataloder_from_embeddings, get_metrics, show_confusion_matrix

class TrainerRNN:
    def __init__(self, model, lr, optim="adam", class_weights=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        
        self.model =  model
        if not class_weights is None:
            class_weights = class_weights.to(self.device)
        self.cost_function = torch.nn.CrossEntropyLoss(weight=class_weights)
        if optim.lower() == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        if optim.lower() == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
    
    def fit(self, dataloader_train, dataloader_val, early_stopping, epochs=50):
        self.model.to(self.device)

        train_losses = []
        val_losses = []
        
        progress_bar = tqdm(range(epochs), leave=True)
        for epoch in progress_bar:
            self.model.train() # Modo de entrenamiento

            # progress_bar = tqdm(dataloader_train, desc=f"Época {epoch+1}/{epochs}", leave=True)
            train_loss = 0
            num_train_samples = 0
            
            for batch_x, lengths, batch_y in dataloader_train:
                # Mover los datos al dispositivo disopnible
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                # lengths = lengths.to(self.device)
                
                # Forward propagation
                output = self.model(batch_x, lengths)
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
                for batch_x, lengths, batch_y in dataloader_val:
                    # Mover al device disponible
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    # lengths = lengths.to(self.device)

                    # Forward propagation
                    output = self.model(batch_x, lengths) # (batch_size, output_size)
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
            for batch_x, lengths, batch_y in dataloader:
                # Mover al device disponible
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward propagation
                output = self.model(batch_x, lengths) # (batch_size, output_size)

                preds = torch.argmax(output, dim=1)
                predictions.extend(preds.cpu().numpy())
        return np.array(predictions)   
    
    def get_model(self):
        return self.model

def evaluate_model(model, dataset, title, embeddings_path, batch_size=64):
    x_tokenized, y_true = dataset['tokens'], dataset['polarity']

    embedding_model = EmbeddingLoader(f"{embeddings_path}.bin")
    embeddings = embedding_model.get_embeddings(x_tokenized)

    dataloader = create_dataloder_from_embeddings(
        embeddings, y_true, batch_size, collate_fn_rnn, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() # Desactivar dropout
    with torch.torch.no_grad(): # no calcular gradiente
        predictions = []
        for batch_x, lengths, batch_y in dataloader:
            # Mover al device disponible
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            # lengths = lengths.to(device)

            # Forward propagation
            output = model(batch_x, lengths) # (batch_size, output_size)
            preds = torch.argmax(output, dim=1)
            predictions.extend(preds.cpu().numpy())
    y_pred = np.array(predictions)   
    
    metrics = get_metrics(y_true, y_pred)
    show_confusion_matrix(y_pred, y_true, title)
    return metrics

def collate_fn_rnn(batch):
    x_batch, y_batch = zip(*batch)  # Separar x e y

    # Obtener longitudes de cada ejemplo
    lengths = torch.tensor([len(seqs) for seqs in x_batch])

    # Convertir a tensores
    x_batch_tensor = [torch.tensor(x, dtype=torch.float32) for x in x_batch]
    y_batch_tensor = torch.tensor(y_batch, dtype=torch.long)

    # Agregar padding
    x_batch_padded = torch.nn.utils.rnn.pad_sequence(
        x_batch_tensor, batch_first=True, padding_value=0
    ) # (batch_size, sequence_length, embedding_dim)

    return x_batch_padded, lengths, y_batch_tensor
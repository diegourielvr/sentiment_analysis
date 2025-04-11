import torch
import time
import numpy as np
from constants.constants_nlp import POLARITY_MAP
from sklearn.utils.class_weight import compute_class_weight
from src.trainers.trainer_rnn import TrainerRNN, collate_fn_rnn
from src.trainers.utils import ModelArgs, EmbeddingLoader, create_dataloder_from_embeddings, get_metrics

class RNNModel(torch.nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.rnn = torch.nn.RNN(
            input_size = model_args.input_size, # Número de características por palabra | Tamaño de los embeddings
            hidden_size = model_args.hidden_size, # Número de características/neuronas de la capa recurrente
            num_layers = model_args.num_layers, # Núumero de capas recurrentes
            nonlinearity = model_args.nonlinearity, # 'relu' deafult: 'tanh'
            dropout = model_args.dropout, # Porcentaje de neuronas desactivadas durante el entrenamiento
            batch_first = True # Los tensores de entrada y salida tienen las dimensiones (batch, seq, feature)
        )
        self.fc = torch.nn.Linear(
            model_args.hidden_size, # Número de características de entrada
            model_args.output_size # Número de clases
        )
    
    def forward(self, x, lengths):
        packed_x = torch.nn.utils.rnn.pack_padded_sequence(
            x,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )
        
        # x: (batch_size, sequence_length, input_size)
        # lengths: (batch_size)
        # output: (batch_size, sequence_length, hidden_size)
        # hidden: (num_layers, batch_size, hidden_size)
        packed_output, hidden = self.rnn(packed_x)
        
        # No es necesario desempacar la salida,
        # porque hidden es el último estado oculto real de cada capa sin padding

        h_n = hidden[-1] # tomar la salida de la última capa rnn
        # h_n: (batch_size, hidden_size)
        # output: (batch_size, output_size)
        return self.fc(h_n)

def train_rnn(dataset_train, dataset_val, embeddings_path,
                    model_args, early_stopping, batch_size=64,
                    lr=1e-3, epochs=50, optim="adam", use_class_weights=False):
    # Dividir información
    x_train_tokenized, y_train = dataset_train['tokens'], dataset_train['polarity']
    x_val_tokenized, y_val = dataset_val['tokens'], dataset_val['polarity']

    # Cargar modelo de embeddings
    embedding_model = EmbeddingLoader(f"{embeddings_path}.bin")
    model_args.input_size = embedding_model.vector_size()
    # Obtener embeddings
    x_train_embeddings = embedding_model.get_embeddings(x_train_tokenized) # list[ndarray[ndarray[float]]]
    x_val_embeddings = embedding_model.get_embeddings(x_val_tokenized)

    dataloader_train = create_dataloder_from_embeddings(
        x_train_embeddings, y_train, batch_size, collate_fn_rnn, shuffle=True
    )
    dataloader_val = create_dataloder_from_embeddings(
        x_val_embeddings, y_val, batch_size, collate_fn_rnn, shuffle=False
    )

    model = RNNModel(model_args)
    class_weights = None
    if use_class_weights is True:
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float)
    trainer = TrainerRNN(model, lr, optim, class_weights)

    # Enrtenar modelo
    start = time.time()
    train_losses, val_losses = trainer.fit(
        dataloader_train, dataloader_val,
        early_stopping, epochs
    )
    end = time.time()
    print(f"Pérdida Entrenamiento = {train_losses[-1]:.4f}, Pérdida Validación = {val_losses[-1]:.4f}")

    # Evaluar modelo
    y_pred = trainer.predict(dataloader_val)
    metrics = get_metrics(y_val, y_pred)

    metrics['model'] = "RNN"
    metrics['optim'] = optim
    metrics['lr'] = lr 
    metrics['patience'] = early_stopping.get_patience() 
    metrics['min_delta'] = early_stopping.get_min_delta() 
    metrics['num_layers'] = model_args.num_layers
    metrics['hidden_size'] = model_args.hidden_size
    metrics['dropout'] = model_args.dropout
    metrics['epochs'] = epochs
    metrics['batch_size'] = batch_size
    metrics['embedding_dim'] = embedding_model.vector_size()
    metrics['train_time'] = end - start

    return trainer, metrics, train_losses, val_losses

class SentimentAnalysis:
    def __init__(self, base_model, tokenizer, embeddings_path, device=None):
        self.model = base_model # MLPModelCustom()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval() # Desactiva Dropout
        self.tokenizer = tokenizer
        self.embedding_model = EmbeddingLoader(f"{embeddings_path}.bin")
    
    def predict(self, x: str):
        x_tokenized = self.tokenizer.tokenize([x]) # list[list[str]
        x_embeddings = self.embedding_model.get_embeddings(x_tokenized)

        x_tensor = torch.tensor(np.array(x_embeddings), dtype=torch.float32)
        length = torch.tensor([len(x_tensor)])
        
        with torch.torch.no_grad(): # no calcular gradiente
            x_tensor = x_tensor.to(self.device)
            length = length.to(self.device)
            output = self.model(x_tensor, length) # (batch_size, output_size)
            pred = torch.nn.functional.softmax(output, dim=1).squeeze()
            res = list(zip(list(POLARITY_MAP.keys()), pred.tolist()))
            
            return sorted(res, key=lambda x: x[1], reverse=True)

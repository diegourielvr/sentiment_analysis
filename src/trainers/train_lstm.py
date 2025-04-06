import torch
import time
from src.trainers.utils import EmbeddingLoader, ModelArgs, create_dataloder_from_embeddings, get_metrics
from src.trainers.trainer_rnn import TrainerRNN, collate_fn_rnn

class LSTMMOdel(torch.nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=model_args.input_size,
            hidden_size=model_args.hidden_size,
            num_layers=model_args.num_layers,
            dropout=model_args.dropout,
            batch_first=True
        )
        self.fc = torch.nn.Linear(
            in_features=model_args.hidden_size,
            out_features=model_args.output_size
        )

    def forward(self, x, lengths):
        """
        x: (batch_size, sequence_length, input_size)
        lengths: (batch_size)
        """
        
        # Empacar las secuencias para ignorar el padding.
        packed_input_sequences = torch.nn.utils.rnn.pack_padded_sequence(
            x,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )
        
       # packed_output: (batch_size, sequence_length, hidden_size) Contiene el estado oculto de todas las secuencias de la ultima capa LSTM
       # hidden_n (num_layers, batch_size, hidden_size) Contiene el estado oculto de la última secuencia de las capas LSTM
       # cell_n (num_layers, batch_size, hidden_size) Contiene el estado de la celda de la ultima secuencia de las capas LSTM
        packed_output_sequences, (hidden, cell) = self.lstm(packed_input_sequences)

        h_n = hidden[-1] # Tomar el estado oculto final de la última capa
        return self.fc(h_n) # (batch_size, output_size)

def train_lstm(dataset_train, dataset_val, embeddings_path,
                    model_args, early_stopping, batch_size=64,
                    lr=1e-3, epochs=50, optim="adam"):
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
        x_train_embeddings, y_train, batch_size, collate_fn_rnn
    )
    dataloader_val = create_dataloder_from_embeddings(
        x_val_embeddings, y_val, batch_size, collate_fn_rnn,
    )

    model = LSTMMOdel(model_args)
    trainer = TrainerRNN(model, lr, optim)

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

    metrics['model'] = "LSTM"
    metrics['optim'] = optim
    metrics['lr'] = lr 
    metrics['patience'] = early_stopping.get_patience() 
    metrics['min_delta'] = early_stopping.get_min_delta() 
    metrics['rnn_layers'] = model_args.num_layers
    metrics['hidden_size'] = model_args.num_layers
    metrics['dropout'] = model_args.dropout
    metrics['epochs'] = epochs
    metrics['batch_size'] = batch_size
    metrics['embedding_dim'] = embedding_model.vector_size()
    metrics['train_time'] = end - start

    return trainer, metrics, train_losses, val_losses

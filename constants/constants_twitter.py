
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TWITTER_TRAIN_RAW_PATH = os.path.join(ROOT_DIR, "data", "twitter", "raw", "twitter_training.csv")
TWITTER_VAL_RAW_PATH = os.path.join(ROOT_DIR, "data", "twitter", "raw", "twitter_validation.csv")

TWITTER_DATASET_TRAIN_PATH = os.path.join(ROOT_DIR, "data","twitter","clean","twitter_dataset_train.csv")
TWITTER_DATASET_VAL_PATH = os.path.join(ROOT_DIR, "data","twitter","clean","twitter_dataset_val.csv")

# Modelos de W2V
EMBEDDING_W2V_TWITTER_PATH = os.path.join(ROOT_DIR, "models", "twitter", "embeddings", "w2v_embeddings_twitter")

# Models path
MODELS_DIR = os.path.join(ROOT_DIR, "models", "twitter", "classifiers")
SVM_MODEL_DIR = os.path.join(MODELS_DIR, "svm")
NB_MODEL_DIR = os.path.join(MODELS_DIR, "nb")
LR_MODEL_DIR = os.path.join(MODELS_DIR, "lr")
SWEM_MODEL_DIR = os.path.join(MODELS_DIR, "swem")
MLP_MODEL_DIR = os.path.join(MODELS_DIR, "mlp")
RNN_MODEL_DIR = os.path.join(MODELS_DIR, "rnn")
LSTM_MODEL_DIR = os.path.join(MODELS_DIR, "lstm")

# Metrics Path
METRICS_DIR = os.path.join(ROOT_DIR, "results", "twitter", "metrics")
SVM_METRICS_PATH = os.path.join(METRICS_DIR, "twitter_svm_metrics.csv")
NB_METRICS_PATH = os.path.join(METRICS_DIR, "twitter_nb_metrics.csv")
LR_METRICS_PATH = os.path.join(METRICS_DIR, "twitter_lr_metrics.csv")
SWEM_METRICS_PATH = os.path.join(METRICS_DIR, "twitter_swem_metrics.csv")
MLP_METRICS_PATH = os.path.join(METRICS_DIR, "twitter_mlp_metrics.csv")
RNN_METRICS_PATH = os.path.join(METRICS_DIR, "twitter_rnn_metrics.csv")
LSTM_METRICS_PATH = os.path.join(METRICS_DIR, "twitter_lstm_metrics.csv")

# Loss curves
MLP_SWEM_LOSS_CURVES_DIR = os.path.join(ROOT_DIR, "results","twitter","loss_curves", "mlp_swem")
RNN_LOSS_CURVES_DIR = os.path.join(ROOT_DIR, "results","twitter","loss_curves", "rnn")
LSTM_LOSS_CURVES_DIR = os.path.join(ROOT_DIR, "results","twitter","loss_curves", "lstm")
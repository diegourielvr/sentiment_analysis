
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TWITTER_TRAIN_RAW_PATH = os.path.join(ROOT_DIR, "data", "twitter", "raw", "twitter_training.csv")
TWITTER_VAL_RAW_PATH = os.path.join(ROOT_DIR, "data", "twitter", "raw", "twitter_validation.csv")

TWITTER_DATASET_TRAIN_PATH = os.path.join(ROOT_DIR, "data","twitter","clean","twitter_dataset_train.csv")
TWITTER_DATASET_VAL_PATH = os.path.join(ROOT_DIR, "data","twitter","clean","twitter_dataset_val.csv")

# Modelos de W2V
EMBEDDING_W2V_TWITTER_PATH = os.path.join(ROOT_DIR, "models", "twitter", "embeddings", "w2v_embeddings_twitter")

# Models path
SVM_PIPELINE_PATH = os.path.join(ROOT_DIR, "models","twitter","classifiers","svm_pipeline.pkl")
NB_PIPELINE_PATH = os.path.join(ROOT_DIR, "models","twitter","classifiers","nb_pipeline.pkl")
LR_PIPELINE_PATH = os.path.join(ROOT_DIR, "models","twitter","classifiers", "lr_pipeline.pkl")
SWEM_MODEL_PATH = os.path.join(ROOT_DIR, "models","twitter","classifiers", "swem_model.pkl")
MLP_SWEM_MODEL_PATH = os.path.join(ROOT_DIR, "models","twitter","classifiers", "mlp_swem_model.pth")
RNN_MODEL_PATH = os.path.join(ROOT_DIR, "models","twitter","classifiers", "rnn_model.pth")
LSTM_MODEL_PATH = os.path.join(ROOT_DIR, "models","twitter","classifiers", "lstm_model.pth")

# Metrics Path
TWITTER_SVM_METRICS_PATH = os.path.join(ROOT_DIR, "results","twitter","metrics","twitter_svm_metrics.csv")
TWITTER_NB_METRICS_PATH = os.path.join(ROOT_DIR, "results","twitter","metrics","twitter_nb_metrics.csv")
TWITTER_LR_METRICS_PATH = os.path.join(ROOT_DIR, "results","twitter","metrics","twitter_lr_metrics.csv")
TWITTER_SWEM_METRICS_PATH = os.path.join(ROOT_DIR, "results","twitter","metrics","twitter_swem_metrics.csv")
TWITTER_MLP_SWEM_METRICS_PATH = os.path.join(ROOT_DIR, "results","twitter","metrics","twitter_mlp_swem_metrics.csv")
TWITTER_RNN_METRICS_PATH = os.path.join(ROOT_DIR, "results","twitter","metrics","twitter_rnn_metrics.csv")
TWITTER_LSTM_METRICS_PATH = os.path.join(ROOT_DIR, "results","twitter","metrics","twitter_lstm_metrics.csv")

# Loss curves
MLP_SWEM_LOSS_CURVES_DIR = os.path.join(ROOT_DIR, "results","twitter","loss_curves", "mlp_swem")
RNN_LOSS_CURVES_DIR = os.path.join(ROOT_DIR, "results","twitter","loss_curves", "rnn")
LSTM_LOSS_CURVES_DIR = os.path.join(ROOT_DIR, "results","twitter","loss_curves", "lstm")
# Análisis de sentimientos

## Estructura

```bash
/analisis_sentimientos/
│
├── /data/
│   ├── dictionaries/               # Diccionarios para corrección ortografica
│   └── tiktok/               
│       ├── clean/                  # Datos preprocesados (limpios y listos para el modelo)
│       ├── raw/                    # Datos intermedios (divididos en oraciones, traducidos)
│       ├── scraped/                # Datos crudos (obtenidos mediante web scraping)
│       └── transcribed/            # Datos transcritos
│
├── /notebooks/                     # EDA y entrenamiento de modelos
│   ├── train_nn.ipynb              # Entrenamiento de modelos MLP, RNN y LSTM en Google Colab
│   ├── tiktok/                     # Entrenamiento de modelos en LOCAL
│   │    ├── 01_eda_tiktok.ipynb    # Análisis exploratorio de datos (EDA)
│   │    ├── 02_SVM_tiktok.ipynb 
│   │    ├── 03_NB_tiktok.ipynb  
│   │    ├── 04_LR_tiktok.ipynb  
│   │    ├── 05_S2V_tiktok.ipynb 
│   │    ├── 06_SWEM_tiktok.ipynb 
│   │    ├── 07_MLP_tiktok.ipynb  
│   │    ├── 08_RNN_tiktok.ipynb  
│   │    └── 09_LSTM_tiktok.ipynb 
│   └── utils/                      # Fusionamiento de datos scrapeados y procesamiento de métricas
│
├── /src/                           # Código fuente del proyecto
│   ├── preprocesamiento/           # Funciones de limpieza, pln y spelling
│   ├── scraping/
│   │   └── data_collection.py      # Funciones para descargar y transcribir videos
│   ├── statistics/                 # Funciones para calcular estadísticas y graficar datos
│   └── trainers/                   # Clases para entrenar modelos
│
├── /models/                        # Modelos guardados
│   └── tiktok/
│       ├── classifiers/            # Modelos entrenados
│       └── embeddings/             # Embeddings entrenados
│
├── /results/                       # Resultados del análisis (gráficos, métricas, etc.)
│   └── tiktok/                     # Resultados de tiktok
│       ├── metrics/                # Métricas de rendimiento de los modelos
│       └── loss_curves/            # Curva de aprendizaje de los modelos
├
├── /constants/                     # Constantes utilizadas en todo el proyecto 
│
├── requirements.txt                # Dependencias del proyecto
└── README.md                       # Documento del proyecto
```

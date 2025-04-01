"""Funciones para estadísticas de los datos
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

## Estadisticas univariables
def estadisticas_numericas(data, col):
    print(f"+ Promedio           : {data[col].mean():.2f}")
    print(f"+ Mediana            : {data[col].median()}")
    print(f"+ Moda               : {data[col].mode()[0]}")
    print(f"+ Varianza           : {data[col].var():.2f}")
    print(f"+ Desviación estándar: {data[col].std():.2f}")
    print(f"+ Mínimo             : {data[col].min()}")
    print(f"+ Máximo             : {data[col].max()}")
    print(f"+ Rango              : {data[col].max()-data[col].min()}")

def estadisticas_categoricas(data, col,min_num_categories_to_show=40):
    """Muestra la frecuencia de cada categoría y su porcentaje con respecto del total
    Si el número de categorías supera el mínimo número de categorías, entonces se muestra
    la cantidad de valores únicos (incluyendo nulos) y la cantidad de valores repetidos (incluyendo nulos)
    """
    total_elementos = data.shape[0]
    categorias = data[col].value_counts()
    print(f"Total de elementos de la columna {col}: {total_elementos}")
    if len(categorias) < min_num_categories_to_show:
        ancho_maximo = max(len(cat) for cat in categorias.index)
        for categoria, count in categorias.items():
            porcentaje = float(count) / total_elementos * 100
            print(f"+ {categoria.ljust(ancho_maximo)}: {str(count).rjust(4)} - {porcentaje:.2f}%")
    else:
        print(f"+ Valores únicos   :\t{len(data[col].unique())}")
        # repetidos = len(df[df[col].duplicated(False) == True].value_counts())
        repetidos = len(data[data[col].duplicated(False) == True][col].unique())
        print(f"+ Valores repetidos:\t{repetidos}")

def estadisticas_fechas(data, col):
    total_elementos = data.shape[0]
    print(f"+ Mínimo: {data[col].min()}")
    print(f"+ Máximo: {data[col].max()}")
    df_year = data.groupby(data[col].dt.year.rename("year")).size().reset_index(name="count") # Obtener numero de comentarios por año
    df_year["porcentaje"] = df_year["count"].apply(lambda x: x / total_elementos * 100)
    for idx, row in df_year.iterrows():
        print(f"+ {int(row['year'])}: {str(int(row['count'])).rjust(4)} - {row['porcentaje']:.2f}%")
    print(f"+ Rango : {data[col].max()-data[col].min()}")

def estadisticas_booleanas(data, col):
    value_counts = data[col].value_counts()
    true = 0
    false = 0
    if True in value_counts:
        true = value_counts[True]
    if False in value_counts:
        false = value_counts[False]
    print(f"+ True : {true}")     
    print(f"+ False: {false}")     

def estadisticas_nulos(data, col):
    """Número de valores núlos
    Las cadenas vacias no son contadas como valores nulos
    """
    total_elementos = data.shape[0]
    nulos = data[col].isnull().sum()
    porcentaje_nulos = nulos / total_elementos * 100
    print(f"+ Valores nulos: {nulos} - {porcentaje_nulos:.2f}%")   

def estadisticas_basicas(data):
    for col in data.columns:
        print(f"--- Estadísticas para la columna: {col} ---")
        if data[col].dtype in ["int64", "float64"]: # Columna numérica
            estadisticas_numericas(data, col)
        elif data[col].dtype == "datetime64[ns]":
            estadisticas_fechas(data, col)
        elif data[col].dtype == "bool":
            estadisticas_booleanas(data, col)
        elif data[col].dtype == "object":
            estadisticas_categoricas(data, col)
        else:
            print(f"Tipo de dato no conocido: {data[col].dtype}")
        estadisticas_nulos(data, col)
        print()       

## Estadisticas bivariables 

# Talvez no es necesaria
def estadisticas_idioma(data, col, lang):
    print(f"--- Estadisticas de la columna {col} del idioma {lang}")
    estadisticas_categoricas(data, col)
    estadisticas_nulos(data, col)
    comentarios_vacios = data[data[col].str.strip() == ""]
    print(f"Comentarios con espacios vacios: {len(comentarios_vacios)}")

    print("\nValores duplicados:")
    repetidos = data[data[col].duplicated(False) == True].reset_index(drop=True) # Marcar todos los duplicados como True
    # repetidos_str = repetidos[col].astype(str)
    repetidos_str = repetidos[col].astype('string')
    print(repetidos_str.value_counts())

def estadisticas_vocabulario(data, col, ascendente=True, archivo=None):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data[col].astype(str))
    print("Palabras únicas: ", len(vectorizer.get_feature_names_out()))
    # print(vectorizer.get_feature_names_out())
    
    # Obtener el vocabulario (palabras únicas) y su frecuencia total
    frecuencia_palabras = X.toarray().sum(axis=0)  # Suma la frecuencia de cada palabra en todas las filas
    vocabulario = vectorizer.get_feature_names_out()

    # Convertir a DataFrame para visualizar mejor
    df_frecuencia = pd.DataFrame({'Palabra': vocabulario, 'Frecuencia': frecuencia_palabras})

    # Ordenar por frecuencia descendente
    df_frecuencia = df_frecuencia.sort_values(by="Frecuencia", ascending=ascendente)
    if archivo is not None and archivo.endswith(".csv"):
        df_frecuencia.to_csv(archivo, index=None)
        print(f"Reporte de frecuencias de palabras generado: {archivo}")
    return df_frecuencia.reset_index(drop=True)
    # print(df_frecuencia[:20])
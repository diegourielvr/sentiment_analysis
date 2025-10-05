"""Funciones para visualizar datos
"""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import seaborn as sns
import matplotlib.pyplot as plt

meses_map = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
    5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
    9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
}

def grafica_numerica(data, col):
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(data, x=col,
        color='black', fill=False,
        flierprops={"marker": "x"}, medianprops={"color": 'red', "linewidth": 2})
    ax.ticklabel_format(axis='x', style='plain')
    plt.show()

def grafica_fechas(data, col):
    df_anio = data.groupby(data[col].dt.year).size().reset_index(name="count")
    # df_anio = df_anio.sort_values(col)
    fig_line_anio = px.line(df_anio, x=col,y="count",markers=True)
    df_mes = data.groupby(data[col].dt.month).size().reset_index(name="count")
    df_mes[col] = df_mes[col].apply(lambda x: meses_map[x])
    fig_line_mes= px.line(df_mes, x=col,y="count",markers=True)


    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Cantidad de datos por año", "Cantidad de datos por mes"))
    fig.add_trace(fig_line_anio.data[0], row=1, col=1)
    fig.add_trace(fig_line_mes.data[0], row=1, col=2)
    fig.update_layout(title_text=f"Cantidad de datos por {col}")
    fig.show()

def grafica_booleana(data, col):
    df_temp = data[col].value_counts().reset_index()
    plt.figure(figsize=(10,6))
    ax = sns.barplot(df_temp, x=col, y='count', hue=col)
    ax.set_xlabel(col)
    ax.set_ylabel("Cantidad")

    # Cambiqr rotación y posición de las categorías
    plt.setp(ax.get_xticklabels(),
            rotation=-45, 
            ha='left',
            va='top')
    plt.show()

def grafica_categorica(data, col, min_num_categories_to_whow = 40):
    categorias = data[col].value_counts().reset_index()
    if len(categorias) < min_num_categories_to_whow:
        plt.figure(figsize=(10,6))
        ax = sns.barplot(categorias, x=col, y='count', hue=col)
        ax.set_xlabel(col)
        ax.set_ylabel("Cantidad")

        # Cambiqr rotación y posición de las categorías
        plt.setp(ax.get_xticklabels(),
             rotation=-45, 
             ha='left',
             va='top')
        plt.show()
    else:
        print(f"Gráficos de la variable {col} omitidos")

def visualizacion_datos(data):
    for col in data.columns:
        if data[col].dtype in ["int64", "float64"]:
            grafica_numerica(data, col)
        elif data[col].dtype == "datetime64[ns]":
            grafica_fechas(data, col)
        elif data[col].dtype == "bool":
            grafica_booleana(data, col)
        elif data[col].dtype == "object":
            grafica_categorica(data, col)

## Visualización UNIVARIABLE
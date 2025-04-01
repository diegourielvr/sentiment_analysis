"""Funciones para visualizar datos
"""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

bg_color="plotly_dark" # https://plotly.com/python/templates/
pio.templates.default = bg_color # Establecer el tema oscuro por defecto
color_seq = px.colors.qualitative.Plotly # https://plotly.com/python/builtin-colorscales/ or https://plotly.com/python/discrete-color/#discrete-vs-continuous-color

meses_map = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
    5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
    9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
}

## Visualización UNIVARIABLE

def grafica_numerica(data, col):
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Frecuencia de {col}", f"Distribución de {col}"))
    fig_hist = px.histogram(data, x=col)
    fig_box = px.box(data, x=col, notched=True)
    fig.add_trace(fig_hist.data[0], row=1, col=1)
    fig.add_trace(fig_box.data[0], row=1, col=2)
    # Actualizar etiquetas de los ejes
    fig.update_xaxes(title_text=col, row=1, col=1)  # Eje X del histograma
    fig.update_xaxes(title_text=col, row=1, col=2)  # Eje X del boxplot
    fig.update_yaxes(title_text="count", row=1, col=1)  # Eje Y del histograma
    # fig.update_yaxes(title_text="Valor", row=1, col=2)  # Eje Y del boxplot
    
    fig.update_layout(title_text=f"Gráficos de la variable {col}")
    fig.show()

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
    total_elementos = data.shape[0]
    df_temp = data[col].value_counts().reset_index()
    df_temp["porcentaje"] = df_temp["count"].apply(lambda x: (x / total_elementos) * 100)
    df_temp["porcentaje"] = df_temp["porcentaje"].apply(lambda x: f"{x:.2f}%")
    fig_bar = px.bar(df_temp, x=col,y="count",text=df_temp["porcentaje"],color=col,color_discrete_sequence=color_seq)
    fig_bar.update_layout(title_text=f"Cantidad de datos por {col}")
    fig_bar.update_traces(textposition="outside")
    fig_bar.show()

def grafica_categorica(data, col, min_num_categories_to_whow = 40):
    total_elementos = data.shape[0]
    categorias = data[col].value_counts().reset_index()
    if len(categorias) < min_num_categories_to_whow:
        categorias["porcentaje"] = categorias["count"].apply(lambda x: (x / total_elementos) * 100)
        categorias["porcentaje"] = categorias["porcentaje"].apply(lambda x: f"{x:.2f}%")
        fig_bar = px.bar(categorias, x=col,y="count", text=categorias["porcentaje"],color=col)
        fig_bar.update_layout(title_text=f"Cantidad de datos por {col}")
        fig_bar.update_traces(textposition="outside")
        fig_bar.show()
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
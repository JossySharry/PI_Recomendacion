from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.metrics.pairwise import cosine_similarity
import operator
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix


app = FastAPI()


# Cargamos los datos
df_movies = pd.read_parquet('D:/2024/HenryData/Py_Individual/PI_Recomendacion/Datasets/df_limpio.parquet')

"""
Deben crear 6 funciones para los endpoints que se consumirán en la API, recuerden que deben tener un decorador por cada una (@app.get(‘/’)).

def cantidad_filmaciones_mes( Mes ): Se ingresa un mes en idioma Español. Debe devolver la cantidad de películas que fueron estrenadas en el mes consultado en la totalidad del dataset.
                    Ejemplo de retorno: X cantidad de películas fueron estrenadas en el mes de X

def cantidad_filmaciones_dia( Dia ): Se ingresa un día en idioma Español. Debe devolver la cantidad de películas que fueron estrenadas en día consultado en la totalidad del dataset.
                    Ejemplo de retorno: X cantidad de películas fueron estrenadas en los días X

def score_titulo( titulo_de_la_filmación ): Se ingresa el título de una filmación esperando como respuesta el título, el año de estreno y el score.
                    Ejemplo de retorno: La película X fue estrenada en el año X con un score/popularidad de X

def votos_titulo( titulo_de_la_filmación ): Se ingresa el título de una filmación esperando como respuesta el título, la cantidad de votos y el valor promedio de las votaciones. La misma variable deberá de contar con al menos 2000 valoraciones, caso contrario, debemos contar con un mensaje avisando que no cumple esta condición y que por ende, no se devuelve ningun valor.
                    Ejemplo de retorno: La película X fue estrenada en el año X. La misma cuenta con un total de X valoraciones, con un promedio de X

def get_actor( nombre_actor ): Se ingresa el nombre de un actor que se encuentre dentro de un dataset debiendo devolver el éxito del mismo medido a través del retorno. Además, la cantidad de películas que en las que ha participado y el promedio de retorno. La definición no deberá considerar directores.
                    Ejemplo de retorno: El actor X ha participado de X cantidad de filmaciones, el mismo ha conseguido un retorno de X con un promedio de X por filmación

def get_director( nombre_director ): Se ingresa el nombre de un director que se encuentre dentro de un dataset debiendo devolver el éxito del mismo medido a través del retorno. Además, deberá devolver el nombre de cada película con la fecha de lanzamiento, retorno individual, costo y ganancia de la misma.

adicional 
def recomendacion( titulo ): Se ingresa el nombre de una película y te recomienda las similares en una lista de 5 valores.

"""
# Función auxiliar para traducir meses y días a español
meses_esp = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
             "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12}

dias_esp = {"lunes": 0, "martes": 1, "miércoles": 2, "jueves": 3, "viernes": 4, "sábado": 5, "domingo": 6}

# 1. Función para la cantidad de filmaciones en un mes específico
def cantidad_filmaciones_mes(mes: str):
    mes = mes.lower()
    if mes in meses_esp:
        mes_num = meses_esp[mes]
        count = df_movies[df_movies['release_date'].dt.month == mes_num].shape[0]
        return {"message": f"{count} películas fueron estrenadas en el mes de {mes.capitalize()}"}
    else:
        return {"error": "Mes no válido. Use un mes en español"}

# ruta para devolver la cantidad de filmaciones por mes desde el archivo parquet

@app.get('/cantidad_filmaciones_mes/{mes}')
async def get_cantidad_filmaciones_mes(mes: str):
    return cantidad_filmaciones_mes(mes)


# 2. Función para contar cantidad de filmaciones en un día específico

def cantidad_filmaciones_dia(dia: str):
    dia = dia.lower()
    if dia in dias_esp:
        dia_num = dias_esp[dia]
        count = df_movies[df_movies['release_date'].dt.weekday == dia_num].shape[0]
        return {"message": f"{count} películas fueron estrenadas en los días {dia.capitalize()}"}
    else:
        return {"error": "Día no válido. Use un día en español"}

# Ruta para devolcer la cantidad de filmaciones por día
@app.get('/cantidad_filmaciones_dia/{dia}')
async def get_cantidad_filmaciones_dia(dia:str):
    return cantidad_filmaciones_dia(dia)

"""
# 3. Función para obtener el score de una película por título
@app.get('/score_titulo/{titulo}')
def score_titulo(titulo: str):
    row = df_movies[df_movies['title'].str.lower() == titulo.lower()]
    if not row.empty:
        year = row.iloc[0]['release_year']
        score = row.iloc[0]['popularity']
        return {"message": f"La película {titulo} fue estrenada en el año {year} con un score de {score}"}
    else:
        return {"error": "Película no encontrada"}

# 4. Función para obtener votos de una película por título
@app.get('/votos_titulo/{titulo}')
def votos_titulo(titulo: str):
    row = df_movies[df_movies['title'].str.lower() == titulo.lower()]
    if not row.empty:
        votes = row.iloc[0]['vote_count']
        average_vote = row.iloc[0]['vote_average']
        year = row.iloc[0]['release_year']
        if votes >= 2000:
            return {"message": f"La película {titulo} fue estrenada en el año {year}. La misma cuenta con un total de {votes} valoraciones, con un promedio de {average_vote}"}
        else:
            return {"message": "La película no cumple con la condición de al menos 2000 valoraciones"}
    else:
        return {"error": "Película no encontrada"}

# 5. Función para obtener datos de un actor
@app.get('/get_actor/{nombre_actor}')
def get_actor(nombre_actor: str):
    # Filtrar el DataFrame para obtener películas en las que el actor ha participado
    df_actor = df_movies[df_movies['cast'].apply(lambda x: nombre_actor in x)]
    if not df_actor.empty:
        total_return = df_actor['revenue'].sum() - df_actor['budget'].sum()
        num_movies = df_actor.shape[0]
        avg_return = total_return / num_movies
        return {"message": f"El actor {nombre_actor} ha participado en {num_movies} filmaciones, consiguiendo un retorno total de {total_return} y un promedio de {avg_return:.2f} por filmación"}
    else:
        return {"error": "Actor no encontrado o sin participaciones en el dataset"}

# 6. Función para obtener datos de un director
@app.get('/get_director/{nombre_director}')
def get_director(nombre_director: str):
    # Filtrar el DataFrame para obtener películas dirigidas por el director especificado
    df_director = df_movies[df_movies['crew'].apply(lambda x: nombre_director in x)]
    if not df_director.empty:
        details = []
        total_return = 0
        for _, row in df_director.iterrows():
            return_individual = row['revenue'] - row['budget']
            total_return += return_individual
            details.append({"title": row['title'], "release_date": row['release_date'], "return": return_individual, "budget": row['budget'], "revenue": row['revenue']})
        return {"message": f"El director {nombre_director} tiene un retorno total de {total_return} con el siguiente detalle:", "details": details}
    else:
        return {"error": "Director no encontrado o sin filmaciones en el dataset"}

# Modelo
@app.get('/recommend/{title}')
def recommend_movies(title: str):
    try:
        recommendations = get_recommendations(title)
        return {"recommendations": recommendations}
    except IndexError:
        return {"error": "Title not found"}
        """
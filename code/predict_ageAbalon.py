# Para prediccion, correr en consola: uvicorn code.predict_ageAbalon:app --reload
# Correr en browser de internet:  http://localhost:8000/docs


import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pydantic.types import confloat
from code.models.predictAbalonRings import predict_abalonRings_withUserData



# Leer el archivo JSON
with open('predict_vars.json', 'r') as file:
    data = json.load(file)

# Recuperar el diccionario y el string
info_entry = data["info_entry"]
flg = data["flg"]



# Preparacion de la aplicacion que llamará a FastAPI
app = FastAPI()                                     # Se crea instancia

# Clase con las variables de entrada (atributos)
class AbalonInput(BaseModel):
    Shell_weight: confloat (ge=0.001, le=1)         # Solo recibirá valores en rango [0.001, 1]
    Diameter: confloat (ge=0.001, le=1.5)           # Solo recibirá valores en rango [0.001, 1.5]


# Clase con la variable de salida (objetivo) y una segunda variable derivada de la primera
class AbalonOutput(BaseModel):
    predicted_rings: int
    predicted_age: float


# ------------------------------------------------------
@app.post("/predict_abalon")
async def predict_abalon (input_data: AbalonInput) :
# ------------------------------------------------------
    Shell_weight = input_data.Shell_weight
    Diameter = input_data.Diameter

    X_new = [[Shell_weight, Diameter]]              # Inicializar la matriz de datos nuevos de entrada

    # Llamar a la función predict_abalonRings_withUserData
    y_pred_new_rounded = predict_abalonRings_withUserData(X_new, info_entry, flg)
    abalon_age = y_pred_new_rounded + 1.5           # Se suma el factor 1.5 para obtener la edad del abalón

    # Devolver los resultados
    return AbalonOutput(predicted_rings=y_pred_new_rounded, predicted_age=abalon_age)


# ------------------------------------------------------
@app.get("/")
async def read_root () :
# ------------------------------------------------------
    return {"message": "Bienvenido a la API de predicción de edad de abulón."}


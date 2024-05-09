# PROYECTO WIZELINE MLOPS 
Proyecto de 'Machine Learning Operations' para la predicción de la edad de abulones en base a sus características físicas

## Pasos para probar el proyecto

Crear entorno virtual:

```cmd
python -m venv venv
```

Activar entorno virtual:

```cmd
activate
```

Instalar dependencias:

```cmd
pip install -r requirements.txt
```


Para correr modelo, ejecutar en consola el script main.py 
enviando el parametro .dvc del archivo que se desea procesar y la ruta donde se encuentra el repositorio dvc: 

```cmd
python code\main.py data\raw\abalone.data.dvc .\
```

Para testear el modelo con pruebas unitarias y de integración: correr en consola: 

```cmd
pytest -v
```


Para predecir con el modelo

```cmd
uvicorn code.predict_ageAbalon:app --reload
```
y en el navegador de internet revisar la documentación de FastAPI

http://127.0.0.1:8000/docs



Para revisar los experimentos

```cmd
mlflow ui
```
y en el navegador de internet revisar

http://127.0.0.1:5000

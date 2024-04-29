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

Ejecutar runAll.bat ubicado en PROYECTOWIZELINE para ejecutar pruebas unitarias y de integración con pytest y enseguida la corrida del modelo con main.py

```cmd
.\runAll.bat
```


Para predecir con el modelo

```cmd
uvicorn code.predict_ageAbalon:app --reload
```

y en el navegador de internet revisar la documentación de FastAPI

http://127.0.0.1:8000/docs



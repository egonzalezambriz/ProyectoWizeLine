# Importar librerías
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow


from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from utils.visualization import histplots_figure, matrixCorrCoef_figure, boxplot_figure, dispersion_figure
from utils.mseMetric import find_minMseMetric, print_mse_tests, metrics_report

from data.preprocessData import remove_outliers, preprocess_data
from data.loadData import load_data
from data.splitData import split_dataset

from models.experiments import do_experiment
from models.validate import validateModel_withValuationDataset
from models.predictAgeAbalon import predict_abalonRings_withUserData
from models.experiments import gen_gridSearchHiperParam
from models.adminWithMLFlow import get_metadataOfBestModel






# ================================================================================================
def predictAbalonAge (printImages=0) :
# ================================================================================================

    # -------------------------------------------------------------------------------------------------------------------
    #       P   R   E   P   R   O   C   E   S   O          D  A   T   O   S
    # -------------------------------------------------------------------------------------------------------------------

    # "Rings" es una variable categórica discreta que representa el número de anillos
    # Es posible predecirla mediante clasificación pero también mediante regresión  
    # si se considera como una variable numérica continua
  
    # Obtencion de datos y generacion de la informacion que contiene
    data_Mtrx, names_Arr = load_data()

    # Conversiones e imputaciones de datos por datos faltantes
    data_Mtrx, cols_names_Arr, cols_ImputerNames_Arr, data_Fr = preprocess_data (data_Mtrx)

    # Graficas de exploracion de datos
    if ( printImages == 1 ) :
        
        # Desplegar histogramas de los principales atributos
        # Se notará que prácticamente todos son aproximaciones de distribuciones normales con excepcion de 'Height' que está muy cargado 2 dos rangos
        histplots_figure (data_Fr, cols_ImputerNames_Arr, nrows=2, ncols=4)


        # Se deducirá que las variables de entrada mas correlacionadas con la variable de salida son:
        # 'Shell_weight', 'Diameter','Length' y 'Height', de las que se elegirán para las predicciones las 2 primeras por cumplir con los criterios 
        matrixCorrCoef_figure (data_Fr, cols_names_Arr)


        # Graficar diagramas de caja para variables de entrada de manera que se visualicen los valores atípicos
        # De las dos características seleccionadas anteriormente, se notará que :
        # 'Shell_weight' tiene valores atípicos en la parte alta de la gráfica
        # 'Diameter' tiene valores atípicos en la parte baja de la gráfica            
        boxplot_figure (data_Fr, ['Length','Diameter','Height','Whole_weight','Shucked_weight','Viscera_weight','Shell_weight'])

    
    
    # Remover valores atípicos de las variables de atributos 'Shell_weight' y 'Diameter'
    data_Fr = remove_outliers(data_Fr, column='Shell_weight', lower_percentile=0.0, upper_percentile=0.90)
    data_Fr = remove_outliers(data_Fr, column='Diameter', lower_percentile=0.075, upper_percentile=1)
    print ('data_Fr.shape: ', data_Fr.shape)

    if ( printImages == 1 ) :

        # Diagramas de dispersión para pares de variables
        # Se confirma que las caracteristicas 'Shell_weight' y 'Diameter' están muy relacionadas con las demás variables de entrada
        
        dispersion_figure (data_Fr, cols_names_Arr)

    
    


    # ---------------------------------------------------------------------------------------------------
    #   S   E   P   A   R   A   C   I   O   N       D   A   T   O   S
    # ---------------------------------------------------------------------------------------------------
    test_size = 0.15
    print ('test_size: ', test_size)
    X, y, X_train, y_train, X_test, y_test, X_val, y_val = split_dataset (test_size, data_Fr [['Shell_weight','Diameter']], data_Fr ['Rings'])




    # --------------------------------------------------------
    #       E    N   T   R   E   N   A   M   I   E   N   T   O
    # --------------------------------------------------------
    print ('Inicia entrenamiento ...')
    
    # Se entrenan 3 modelos con algoritmos diferentes para ajustarlos a los datos de entrenamiento
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    rf_model = RandomForestRegressor(n_estimators=100, max_features=2, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train)

    en_model = ElasticNet(alpha=0.1, l1_ratio=0.1, max_iter=1000, tol=0.001, random_state=42)
    en_model.fit(X_train, y_train)




    # --------------------------------------------------------
    #       P    R   U   E   B   A   S
    # --------------------------------------------------------
    
    # Calcular las desviaciones entre los datos reales del conjunto de prueba contra los datos predichos por los modelos 

    print ('Inician pruebas ...')

    
    y_pred_linear_test = np.round(linear_model.predict(X_test))         # Predicciones con Regresion Lineal en conjunto de datos de prueba
    mse_linear_test = mean_squared_error(y_test, y_pred_linear_test)    # Metrica error cuadrático medio para regresion lineal
    
    y_pred_en_test = np.round(en_model.predict(X_test))                 # Predicciones con Elastic net en conjunto de datos de prueba    
    mse_en_test = mean_squared_error(y_test, y_pred_en_test)            # Metrica error cuadrático medio para regesion elastic net
    
    
    y_pred_rf_test = np.round(rf_model.predict(X_test))                 # Predicciones con Random Forest en conjunto de datos de prueba       
    mse_rf_test = mean_squared_error(y_test, y_pred_rf_test)            # Metrica error cuadrático medio para Random forest
        
    # Se elije el modelo con la mejor métrica
    models = ['LinearRegression','ElasticNet','RandomForestRegressor']
    test_min_mse, min_idx = find_minMseMetric (mse_linear_test, mse_en_test, mse_rf_test)
    test_best_model = models [min_idx]
    
    print_mse_tests (mse_linear_test, mse_en_test, mse_rf_test)
    
    



    # --------------------------------------------------------------------------------------------------------
    #           G   E   N   E   R   A   R          E   X   P   E   R   I   M   E   N   T   O   S
    # --------------------------------------------------------------------------------------------------------

    # Generar rejilla de hiper parámetros que se utilizarán en cada experimento con el modelo seleccionado en las pruebas
    param_grid = gen_gridSearchHiperParam (test_best_model) 

    # Para guardar los rendimientos de cada corrida de cada experimentos
    performanceRunningLst = []


    print ('Inicia experimento 1 ...')
    # Iniciar experimento version 1
    experiment_name = "Abalon/V1"
    mlflow.set_experiment(experiment_name)
    test_size = 0.10
    if 'LinearRegression' in test_best_model :
        X_val, y_val, y_pred_val, mse_val1 = do_experiment (test_best_model, test_min_mse, param_grid, test_size, performanceRunningLst, X_train, y_train, X_val, y_val)
    elif 'ElasticNet' in test_best_model or 'RandomForestRegressor' in test_best_model :
        X_val, y_val, y_pred_val, mse_val1 = do_experiment (test_best_model, test_min_mse, param_grid[0], test_size, performanceRunningLst, X_train, y_train, X_val, y_val)
 

    print ('Inicia experimento 2 ...')
    # Iniciar experimento version 2
    experiment_name = "Abalon/V2"
    mlflow.set_experiment(experiment_name)
    test_size = 0.20
    if test_best_model == 'LinearRegression' :
        X_val, y_val, y_pred_val, mse_val2 = do_experiment (test_best_model, test_min_mse, param_grid, test_size, performanceRunningLst, X_train, y_train, X_val, y_val)
    elif test_best_model == 'ElasticNet' or test_best_model == 'RandomForestRegressor' :
        X_val, y_val, y_pred_val, mse_val2 = do_experiment (test_best_model, test_min_mse, param_grid[1], test_size, performanceRunningLst, X_train, y_train, X_val, y_val)


    


    # ---------------------------------------------------------------------------------------------------------------------
    #   V  A   L   I   D   A   C   I   O   N        M   O   D   E   L   O
    # ---------------------------------------------------------------------------------------------------------------------
    
    # Predecir numero de anillos de los abulones con el conjunto de datos de validacion con el modelo con el mejor rendimiento  

    print ('Inicia validación del modelo ...')

    # Se obtienen algunos metadatos del modelo con el mejor rendimiento
    info_entry = get_metadataOfBestModel (performanceRunningLst)
     
    flg = validateModel_withValuationDataset (X_val, y_pred_val, y_val, info_entry['model'])



    # Se escribe el reporte de las métricas
    metrics_report (mse_linear_test, mse_en_test, mse_rf_test, mse_val1, mse_val2, test_best_model)




    print ('Predicción de la edad de un abulón ...')

    # Se cargará desde MLFLow el modelo con el mejor rendimiento y con los valores de atributos o variables de entrada 
    # se predecirá el numero de anillos y por tanto la edad de un abulon
    X_new = [[0,0]]                                      # Inicializamos la matriz de datos nuevos
    X_new [0][0]  = 0.114                                # peso caparazón
    X_new [0][1] =  0.295                                # diametro 

    y_pred_new_rounded = predict_abalonRings_withUserData (X_new, info_entry, flg) 
    abalon_age = y_pred_new_rounded + 1.5                # Se suma el factor 1.5 para obtener la edad del abalón
    
    # Impresion de resultados
    print ('Peso caparazon: ', X_new [0][0], ' Diámetro: ', X_new [0][1], ' Número anillos predecidos abulón: ', y_pred_new_rounded, ' edad predecida: ', abalon_age)

    





# ================================================================================================
#                          ENTRADA PRINCIPAL AL CODIGO
# ================================================================================================
if __name__ == "__main__":
    predictAbalonAge(1)
    print ('F    I   N   A   L   I   Z   A   C   I   O   N')

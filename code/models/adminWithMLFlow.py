

import os
import pickle
import mlflow.sklearn
import numpy as np
import datetime

from sklearn.metrics import mean_squared_error
from code.utils.mseMetric import print_mseMetrics



# ========================================================================================================================================================
def do_runningsModels (validation_best_model, test_best_model, test_min_mse, stats_X_test, X_train, X_val, y_train, y_val, test_size, performanceRunningsLst, data_file) :
# ========================================================================================================================================================

    ''' 
    Se realizan las corridas del modelo seleccionado como el mejor de las pruebas
    '''
    
    print ('validation_best_model: ', validation_best_model)

    if (test_best_model == 'LinearRegression') :
        i = 0
    
    elif (test_best_model == 'ElasticNet') :
        i = 1

    elif (test_best_model == 'RandomForestRegressor') :
        i = 2


    # Realizar corrida del modelo del experimento
    
    print("Corrida " + str(i + 1) + " model " +  validation_best_model.__class__.__name__)
    with mlflow.start_run(run_name="Corrida " + str(i + 1) + " model " + validation_best_model.__class__.__name__):
                    
        # guarda en MLFlow
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("modelo", validation_best_model.__class__.__name__)
       
        if test_best_model == 'LinearRegression':
            mlflow.log_param("fit_intercept", validation_best_model.fit_intercept)
            mlflow.log_param("n_jobs", validation_best_model.n_jobs)
            mlflow.log_param("positive", validation_best_model.positive)
                    
        elif test_best_model == 'ElasticNet':
            mlflow.log_param("alpha", validation_best_model.alpha)
            mlflow.log_param("l1_ratio", validation_best_model.l1_ratio)
            mlflow.log_param("fit_intercept", validation_best_model.fit_intercept)
            mlflow.log_param("precompute", validation_best_model.precompute)
            mlflow.log_param("max_iter", validation_best_model.max_iter)
            mlflow.log_param("tolerance", validation_best_model.tol)
            mlflow.log_param("random_state", validation_best_model.random_state)
        
        elif test_best_model == 'RandomForestRegressor':
            mlflow.log_param("n_estimators", validation_best_model.n_estimators)
            mlflow.log_param("max_depth", validation_best_model.max_depth)
            mlflow.log_param("max_features", validation_best_model.max_features)



        # Predicciones con Regresion Lineal en conjunto de datos de validación
        y_pred_val = np.round(validation_best_model.predict(X_val))

        # Metrica error cuadrático medio (mse)
        mlflow.log_metric("mse_test", test_min_mse)    
        mse_val = mean_squared_error(y_val, y_pred_val)
        mlflow.log_metric("mse_validation", mse_val)

        # se guarda el modelo      
        mlflow.sklearn.log_model(validation_best_model, f"modelo_regresion_{i + 1}")
            
        # Obtener el experiment_id y run_id dentro del contexto del run actual
        experiment_id = mlflow.active_run().info.experiment_id
        run_id = mlflow.active_run().info.run_id
        mlflow.set_tag ("dataset", data_file)
        mlflow.set_tag ('Shell_weight avg', float("{:.6f}".format(stats_X_test['mean_X_test'][0])))
        mlflow.set_tag ('Shell_weight var', float("{:.6f}".format(stats_X_test['var_X_test'][0])))       
        mlflow.set_tag ('Diameter avg', float("{:.6f}".format(stats_X_test['mean_X_test'][1])))
        mlflow.set_tag ('Diameter var', float("{:.6f}".format(stats_X_test['var_X_test'][1])))       


        # Se agrega la corrida a la lista de experimentos  
        performanceRunningsLst.append ({'experiment_id': experiment_id, 'run_id': run_id, 'model': test_best_model, 'mse_val': mse_val} )

    
    # Finaliza la corrida
    mlflow.end_run()
    
    print_mseMetrics (test_best_model, test_min_mse, mse_val) 
 
    return y_pred_val, mse_val




# ----------------------------------------------------------------------------
def load_bestModelFromMLFlow (info_entry, flg) :
# ----------------------------------------------------------------------------
    '''
    Como los modelos están registrado en MLFLow, entonces para hacer predicciones se carga desde ahí el pickle del modelo óptimo     
    '''

    # Configurar la conexión a MLflow
    experiment_id = info_entry ['experiment_id']
    run_id = info_entry ['run_id']

    # Obtener el cliente de MLflow
    client = mlflow.tracking.MlflowClient()

    # Obtener la información del run
    run_info = client.get_run(run_id)

    # Ruta al directorio de artefactos y nombre del archivo .pickle buscado
    artifacts_dir = "C:\\Users\\52477\\Documents\\ProyectoWizeLine\\mlruns\\" + experiment_id + "\\" + run_id + "\\artifacts\\modelo_regresion_" + flg + "\\"
    pickle_file_name = "model.pkl"

    # Verificar si el directorio de artefactos existe
    if os.path.exists(artifacts_dir):
            
        files_in_artifacts = os.listdir(artifacts_dir)              # Obtener la lista de archivos en el directorio de artefactos
        if pickle_file_name in files_in_artifacts:                  # Buscar el archivo .pickle por su nombre
            model_path = artifacts_dir + pickle_file_name           # Construir la ruta completa al archivo .pickle
                
            try:
                with open(model_path, "rb") as f:
                    loaded_model = pickle.load(f)                   # Cargar el modelo desde el archivo .pickle

            except Exception as e:
                print ("Ocurrió un error al cargar el archivo pickle: ", e)
        else:
            print ("El archivo pickle no fue encontrado en el directorio de artefactos")
    else:
        print ("El directorio de artefactos no existe")

    return loaded_model








# ==========================================================
def get_metadataOfBestModel (performanceRunningLst) :
# ==========================================================
    '''
    Encontrar la entrada de las corridas de los experimentos con el mejor rendimiento
    '''
    min_mse = 10000000                                  # Establecer un valor inicial muy grande
    min_entry = None                                    # Inicializar la entrada mínima como None
      
    for entry in performanceRunningLst :
        mse_actual = entry.get('mse_val', 10000000)     # Obtener el valor de 'mse_val', si no está presente, usar un valor muy grande
        if mse_actual < min_mse:
            min_mse = mse_actual
            min_entry = entry

    print ("experiment_id                     run_id                    model             me_val")
    print ("------------------------------------------------------------------------------")
    print (min_entry['experiment_id'], ' ', min_entry['run_id'], ' ', min_entry['model'], ' ', min_entry['mse_val'])
    print ("")    
       
    return min_entry

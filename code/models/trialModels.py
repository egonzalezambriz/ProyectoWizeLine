
# Importar librerias
import numpy as np
from sklearn.metrics import mean_squared_error


# ---------------------------------------------------------------------------------------------------------------------
def testModels_gettingStatistics (X_test) :
# ---------------------------------------------------------------------------------------------------------------------
    """
    Calcula la media y la varianza de los atributos en X_test.
    
    Parámetros:
    - X_test: array-like, shape (n_samples, 2). Conjunto de datos de prueba con las variables atributos
        
    Retorna:
    - stats: dict
        Diccionario con las estadísticas de media y varianza de X_test
    """

    # Calcula la media y la varianza de X_test
    mean_X_test = np.mean (X_test, axis=0)
    var_X_test = np.var (X_test, axis=0)
    
    # Almacena las estadísticas en un diccionario
    stats_X_test = {'mean_X_test': mean_X_test, 'var_X_test': var_X_test}
   
    return stats_X_test




# ---------------------------------------------------------------------------------------------------------------------
def testModels_gettingMSE (linear_model, en_model, rf_model, X_test, y_test) :
# ---------------------------------------------------------------------------------------------------------------------
    '''
    Predicciones con 3 modelos diferentes y el mismo dataset de pruebas
    Regresa la métrica MSE obtenida con cada modelo 
    '''

    print ('Inician pruebas ...')

    y_pred_linear_test = np.round(linear_model.predict(X_test))         # Predicciones con Regresion Lineal en conjunto de datos de prueba
    mse_linear_test = mean_squared_error(y_test, y_pred_linear_test)    # Metrica error cuadrático medio para regresion lineal
    
    y_pred_en_test = np.round(en_model.predict(X_test))                 # Predicciones con Elastic net en conjunto de datos de prueba    
    mse_en_test = mean_squared_error(y_test, y_pred_en_test)            # Metrica error cuadrático medio para regesion elastic net
    
    
    y_pred_rf_test = np.round(rf_model.predict(X_test))                 # Predicciones con Random Forest en conjunto de datos de prueba       
    mse_rf_test = mean_squared_error(y_test, y_pred_rf_test)            # Metrica error cuadrático medio para Random forest

    return mse_linear_test, mse_en_test, mse_rf_test               



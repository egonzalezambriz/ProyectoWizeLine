
# Importar librerias
import numpy as np
from sklearn.metrics import mean_squared_error


# ---------------------------------------------------------------------------------------------------------------------
def testModels_gettingMSE (linear_model, en_model, rf_model, X_test, y_test) :
# ---------------------------------------------------------------------------------------------------------------------

    print ('Inician pruebas ...')

    y_pred_linear_test = np.round(linear_model.predict(X_test))         # Predicciones con Regresion Lineal en conjunto de datos de prueba
    mse_linear_test = mean_squared_error(y_test, y_pred_linear_test)    # Metrica error cuadrático medio para regresion lineal
    
    y_pred_en_test = np.round(en_model.predict(X_test))                 # Predicciones con Elastic net en conjunto de datos de prueba    
    mse_en_test = mean_squared_error(y_test, y_pred_en_test)            # Metrica error cuadrático medio para regesion elastic net
    
    
    y_pred_rf_test = np.round(rf_model.predict(X_test))                 # Predicciones con Random Forest en conjunto de datos de prueba       
    mse_rf_test = mean_squared_error(y_test, y_pred_rf_test)            # Metrica error cuadrático medio para Random forest

    return mse_linear_test, mse_en_test, mse_rf_test               

import pandas as pd
import numpy as np




# ============================================================================================================
def validateBestModel_withValuationDataset(X_val, y_pred_evaluation, y_val, info_entry_model) :
# ============================================================================================================
    '''
    Se predice la edad del abulon para todo el dataset de valuacion
    '''

    if info_entry_model == 'LinearRegression':
        flg = "1"
    elif info_entry_model == 'ElasticNet':
        flg = "2"
    elif info_entry_model == 'RandomForestRegressor':
        flg = "3"

    # Obtener los valores reales de la variable objetivo "rings" para los datos de validación
    y_val_real = y_val.reset_index(drop=True)  # Reiniciar los índices para que coincidan con las predicciones

    # Convertir y_pred_evaluation a un DataFrame de Pandas
    predictions_df = pd.DataFrame(y_pred_evaluation, columns=['Rings_Predicted'])

    # Concatenar las predicciones con X_val y y_val_real
    predictions_df = pd.concat([X_val.reset_index(drop=True), predictions_df, y_val_real], axis=1)

    return flg, predictions_df




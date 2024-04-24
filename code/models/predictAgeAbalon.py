
import pandas as pd
import numpy as np
from models.adminWithMLFlow import load_bestModelFromMLFlow



# ======================================================================================
def predict_abalonRings_withUserData (X_new, info_entry, flg) :    
# ======================================================================================
    '''
    Se realizará la predicción de la edad de un abulón dados el peso de su caparazón y su diámetro 
    '''
    # Cargar en memoria el mejor modelo guardado en MLFlow
    loaded_model = load_bestModelFromMLFlow (info_entry, flg)

    # Realizar prediccion final
    y_pred_new = loaded_model.predict(X_new)            # Se obtiene el número de anillos
    y_pred_new_rounded = round (y_pred_new [0])         # Se redondea el número de anillos

    return y_pred_new_rounded
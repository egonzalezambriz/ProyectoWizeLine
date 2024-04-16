
# Importar librerias
from sklearn.model_selection import train_test_split


# ================================================================================================
def split_dataset (test_size, data_inputAtributes, data_outputTarget) :
# ================================================================================================
    
    '''
    Separacion del dataset en datos para entrenamiento, prueba y validacion
    '''

    # Separar las dos características de entrada de la variable objetivo de salida 'Rings'
    X = data_inputAtributes
    y = data_outputTarget

    # Se repartirán los datos para entrenamiento, pruebas y validacion
    X_remain, X_test, y_remain, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_remain, y_remain, test_size=test_size, random_state=42)

    return X, y, X_train, y_train, X_test, y_test, X_val, y_val  

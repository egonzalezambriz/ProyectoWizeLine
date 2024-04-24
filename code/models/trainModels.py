
# Importar librerias
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor


# --------------------------------------------
def initialTrainModels (X_train, y_train) :
# --------------------------------------------

    '''
    Se entrenan los 3 modelos de regresion con los hiper parametros mas default posible 
    '''

    print ('Inicia entrenamiento ...')
    
    # Se entrenan 3 modelos con algoritmos diferentes para ajustarlos a los datos de entrenamiento
    
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    rf_model = RandomForestRegressor(n_estimators=100, max_features=2, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train)

    en_model = ElasticNet(alpha=0.1, l1_ratio=0.1, max_iter=1000, tol=0.001, random_state=42)
    en_model.fit(X_train, y_train)

    return linear_model, rf_model, en_model

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from data.splitData import split_dataset
from models.adminWithMLFlow import do_runningsModels




# ================================================================================================
def gen_gridSearchHiperParam (best_model) : 
# ================================================================================================

    ''' 
    Generar las rejillas de busqueda de los mejores hiper parametros  de cada algoritmo de aprendizaje
    '''

    if (best_model == 'LinearRegression') :
        param_grid = {
                            'fit_intercept': [True, False],
                            'n_jobs': [2, 4, 8],
                            'positive': [True]
                    }

    elif (best_model == 'ElasticNet') :
        param_grid = [
                            {
                                'alpha': [0.1, 0.5, 0.9],
                                'l1_ratio': [0.33, 0.66, 1],
                                'fit_intercept': [True, False],
                                'precompute': [False],
                                'max_iter' : [500, 750, 1000],
                                'tol' : [0.001],
                                'random_state' : [42]
                            },
                            {
                                'alpha': [0.00001, 0.0001, 0.001],
                                'l1_ratio': [0.001, 0.01, 0.1],
                                'fit_intercept': [True, False],
                                'precompute': [False],              
                                'max_iter' : [1500, 2500, 5000],
                                'tol' : [0.0001],
                                'random_state' : [42]
                            }
                        ]

    elif (best_model == 'RandomForestRegressor') :
        param_grid =    [   
                            {
                                'n_estimators': [100, 500, 1000],
                                'max_depth': [5, 10, 15],
                                'max_features' : [2],                     # Máximo dos caracteristicas
                                'random_state' : [42]
                            },
                            {
                                'n_estimators': [1500, 2500, 3500],
                                'max_depth': [2, 3, 5],
                                'max_features' : [2],                      # Máximo dos caracteristicas
                                'random_state' : [42]
                            }
                        ]

    return param_grid




 
# ================================================================================================
def get_optimizedValidationModel (param_grid, test_best_model, X_train, y_train) :
# ================================================================================================

    '''
    EL mejor modelo del testing, se optimizao utilizando rejillas de busqueda de mejores hiper parametros
    '''

    # Generar el grid search y usar 10 subconjuntos para realizar validaciones cruzadas  
    if ('LinearRegression' in test_best_model) :
        grid_search = GridSearchCV(LinearRegression(), param_grid, cv=10)
        
    elif ('ElasticNet' in test_best_model) :
        grid_search = GridSearchCV(ElasticNet(), param_grid, cv=10)

    elif ('RandomForestRegressor' in test_best_model) :
        grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=10)
        
    
    # Ajustar el mejor modelo obtenido con la rejilla de busqueda de hiper parametros con los datos de entrenamiento
    grid_search.fit(X_train, y_train)
        
    # Obtener los modelos óptimos de acuerdo a los hiper parámetros de la busqueda por rejilla
    validation_best_model = grid_search.best_estimator_
    

    # Imprimir los hiper parámetros de cada modelo
    print ('')
    print ('Mejores hiper parámetros por modelo')
    print ('-----------------------------------')
    print ('validation_best_model: ', validation_best_model)
    print ("\n")

    return validation_best_model





# ====================================================================================================================================================================
def do_experiment (test_best_model, test_min_mse, param_grid, test_size, performanceRunningsLst, X_train, y_train, X_val, y_val) :
# ====================================================================================================================================================================

    
    #       V    A   L   I   D   A   C   I   O   N

    # El mejor modelo en las pruebas pero con diferentes hiper parametros
    validation_best_model = get_optimizedValidationModel (param_grid, test_best_model, X_train, y_train) 
        
    # Corridas del experimento
    y_pred_val, mse_val = do_runningsModels (validation_best_model, test_best_model, test_min_mse, X_train, X_val, y_train, y_val, test_size, performanceRunningsLst) 

    return X_val, y_val, y_pred_val, mse_val







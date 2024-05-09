
from code.models.optimizeModel import get_optimizedValidationModel
from code.models.adminWithMLFlow import do_runningsModels




# ====================================================================================================================================================================
def do_experiment (test_best_model, test_min_mse, param_grid, test_size, performanceRunningsLst, X_train, y_train, X_val, y_val, data_file) :
# ====================================================================================================================================================================

    '''
    Se realiza un experimento con el mejor modelo de la fase de pruebas,  ahora con los datos de validación
    Se optimiza con una rejilla de hiper parametros
    '''    

    #       V    A   L   I   D   A   C   I   O   N

    # El mejor modelo en las pruebas pero con diferentes hiper parametros
    validation_best_model = get_optimizedValidationModel (param_grid, test_best_model, X_train, y_train) 
        
    # Corridas del experimento
    y_pred_val, mse_val = do_runningsModels (validation_best_model, test_best_model, test_min_mse, X_train, X_val, y_train, y_val, test_size, performanceRunningsLst, data_file) 

    return y_pred_val, mse_val






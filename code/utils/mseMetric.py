

# ==============================================================
def find_minMseMetric (mse_0, mse_1, mse_2) :
# ==============================================================
    """
    Función para determinar el menor de tres errores cuadráticos medios (MSE)
    """
    mse_list = [mse_0, mse_1, mse_2]
    min_val = min(mse_list)
    min_idx = mse_list.index(min_val)  
        
    return min_val, min_idx




# ==============================================================
def print_mse_tests (mse_linear_test, mse_en_test, mse_rf_test) :
# ==============================================================
    '''
    Se imprime la metrica mse para cada uno de los 3 algoritmos de regresion
    '''

    print ("\n")
    print ('                          Test mse')
    print ('-----------------------------------------------------------------')
    print('Linear Regression            ', '{:.6f}'.format(mse_linear_test))
    print('Elastic Net                  ', '{:.6f}'.format(mse_en_test))
    print('Random Forest                ', '{:.6f}'.format(mse_rf_test))
    print ("\n")

    






# =======================================================================================================
def print_mseMetrics (test_best_model, test_min_mse, mse_val) :
# =======================================================================================================
    '''
    # Imprimir resultados de los errores cuadráticos medios (mse) de cada modelo en cada conjunto de datos
    '''
 
    print ("\n")
    print ('                               Test mse     Val mse      ')
    print ('-----------------------------------------------------------------')
    print(test_best_model + "         ", "{:.6f}".format(test_min_mse), "   ", "{:.6f}".format(mse_val))
    print ("\n")
    





# =======================================================================================================
def metrics_report (mse_linear_test, mse_en_test, mse_rf_test, mse_val1, mse_val2, test_best_model) :
# =======================================================================================================

    '''
    Escribe en un archivo la evaluacion de los modelos con las métricas obtenidas
    '''

    ruta_archivo = "C:\\Users\\52477\\Documents\\ProyectoWizeLine\\reports\\evaluation_report.txt"
    
    with open(ruta_archivo, 'w') as f:
        f.write("Reporte de Validacion de Modelos\n")
        f.write("                          Test mse\n")
        f.write("-----------------------------------------------------------------\n")
        f.write("Linear Regression            " + str(mse_linear_test) + "\n")
        f.write("Elastic Net                  " + str(mse_en_test) + "\n")
        f.write("Random Forest                " + str(mse_rf_test) + "\n")
        f.write("-----------------------------------------------------------------\n\n")
        if (mse_val1 <= mse_val2) :
            f.write("El modelo optimizado para generalizar es: " + test_best_model + " con mse: " + str(mse_val1))
        else :
            f.write("El modelo optimizado para generalizar es: " + test_best_model + " con mse: " + str(mse_val2))



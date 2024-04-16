# Importar librerias
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ================================================================================================
def histplots_figure (data_Fr, cols, nrows, ncols):
# ================================================================================================

    '''
    Esta función dibuja varios histogramas en una sola gráfica utilizando 'subplots'
    Parámetros:
        data_Fr : dataframe con los datos a graficar 
        cols : nombre de las columnas del dataframe
        nrows : número de renglones
        ncols : numero de columnas
    '''
    # Crear un dataframe temporal para graficar los histogramas de las caracteristicas mas importantes
    data_temp_Fr = pd.DataFrame(data_Fr, columns=cols)

    # Crear una figura con varios 'subplots' dividos en "nrows" renglones y "ncols" columnas
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))

    # Iterar sobre las columnas y renglones dibujando un histograma en cada 'subplot'
    for i in range(nrows):
        for j in range(ncols):
            column_name = cols[i * ncols + j]
            axes[i, j].hist(data_temp_Fr[column_name])
            axes[i, j].set_title(f'Histogram {column_name}')
    
    plt.tight_layout()              # Ajustar automaticamente el espacio entre los 'subplots'
    plt.gcf().canvas.manager.set_window_title('Menu Opciones')
    #plt.show()                      # Desplegar
    plt.savefig("C:\\Users\\52477\\Documents\\ProyectoWizeLine\\reports\\figures\\histplots.png")







# ================================================================================================
def matrixCorrCoef_figure (data_Fr, cols_names_Arr) :
# ================================================================================================

    '''
    Esta función grafica un mapa de calor de los coeficientes de la matriz de correlaciones
    Parámetros:
        data_Fr : dataframe con los datos a graficar 
        cols_names_Arr : nombre de las columnas del dataframe
    '''
   
   
    # Visualizar la matriz de correlación como grafico 
    #
    # Criterios:
    # Buscar variables de entrada con una correlación más fuerte con la variable de salida 
    # Evitar variables de entrada con una correlación muy fuerte con otra variables de entrada para evitar posibles redundancias 
    
    matrixCorrCoef = np.corrcoef(data_Fr.T)
    plt.figure(figsize=(14, 7))
    sns.heatmap(matrixCorrCoef, cbar=True, annot=True, annot_kws={"size": 11}, yticklabels=cols_names_Arr, xticklabels=cols_names_Arr)
    plt.title('Mapa Calor Matriz Factores Correlacion', fontsize=14)
    sns.set(font_scale=1.5)
    
    # Dar nombre al icono de la parte superior izquierda
    plt.gcf().canvas.manager.set_window_title('Menu Opciones')
    
    #plt.show()
    plt.savefig("C:\\Users\\52477\\Documents\\ProyectoWizeLine\\reports\\figures\\matrizCorrCoef.png")






# ================================================================================================
def  boxplot_figure (data_Fr, names) :
# ================================================================================================

    ''' 
    Diagramas de caja de las variables numericas continuas de tipo atributo 
    Parámetros:
        data_Fr : dataframe con los datos a graficar 
        names : nombre de las columnas del dataframe
    '''

    plt.figure(figsize=(14, 7))
    sns.boxplot(data=data_Fr[names])
    plt.title('Diagrama Caja Variables Atributos', fontsize=14)
    plt.xlabel('Variables de entrada', fontsize=12)
    plt.ylabel('Valores', fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.gcf().canvas.manager.set_window_title('Menu Opciones')
    # plt.show()
    plt.savefig("C:\\Users\\52477\\Documents\\ProyectoWizeLine\\reports\\figures\\boxplot.png")




# ================================================================================================
def boxplot_figure2 (sets, labels) :
# ================================================================================================
    
    ''' 
    Diagramas de caja de los conjuntos de datos reales, de entrenamiento, prueba y validacion 
    Parámetros:
        sets : conjuntos de datos a graficar
        labes : etiquetas de los conjuntos de datos
    '''

    plt.figure(figsize=(14, 7))

    # En un mismo diagrama de caja se muestran los datos de las 4 categorias
    
    plt.boxplot(sets, labels=labels)

    # Agregar etiquetas y título
    plt.xlabel('Dataset')
    plt.ylabel('Rings')
    plt.title('Diagrama Caja Real vs Predicciones')
    plt.gcf().canvas.manager.set_window_title('Menu Opciones')
    #plt.show()                                                              # Mostrar la gráfica
    plt.savefig("C:\\Users\\52477\\Documents\\ProyectoWizeLine\\reports\\figures\\boxplot2.png")




# ================================================================================================
def twoSubplotsDensity_figure (   y, 
                                   y_pred_linear_test, y_pred_rf_test, y_pred_poly_test,
                                   y_pred_linear_val, y_pred_rf_val, y_pred_poly_val        ) :
# ================================================================================================

    
    ''' 
    Graficas de densidad a dos subplots
    Parámetros: 
        y                   : datos reales de la variable objetivo
        y_pred_linear_test  : datos predecidos de prueba algoritmo regresion lineal
        y_pred_rf_test      : datos predecidos de prueba algoritmo random forest
        y_pred_poly_test    : datos predecidos de prueba algoritmo regresion polinomial
        y_pred_linear_val   : datos predecidos de validacion algoritmo regresion lineal
        y_pred_rf_val       : datos predecidos de validacion algoritmo random forest
        y_pred_poly_val     : datos predecidos de validacion algoritmo regresion polinomial
    '''
    
    # Crear gráfica de densidad para comparar las predicciones de las pruebas y las validaciones contra los valores reales
    # Trazar la estimación de la KDE (Kernel Density Estimation) de una variable unidimensional 
    # Es una forma de estimar la distribución de probabilidad de una variable continua basada en una muestra de datos

    plt.figure(figsize=(20, 6))  # Definir el tamaño de la figura

    # Primer subgráfico: comparación de predicciones en las pruebas vs valores reales
    # Se colorean las áreas bajo las curvas
    plt.subplot(1, 2, 1)  # 1 fila, 2 columnas, primer subgráfico
    sns.kdeplot(y, color='blue', label='Valores Reales', fill=True)
    sns.kdeplot(y_pred_linear_test, color='red', label='Predicciones Lineales', fill=True)
    sns.kdeplot(y_pred_rf_test, color='green', label='Predicciones Random Forest', fill=True)
    sns.kdeplot(y_pred_poly_test, color='orange', label='Predicciones Polinómicas', fill=True)
    plt.title('Densidad Predicciones Prueba vs Reales')
    plt.xlabel('Valor')
    plt.ylabel('Densidad')
    plt.legend()
    plt.xlim(5, 15)

    # Segundo subgráfico: comparación de predicciones en las validaciones vs valores reales
    # Se colorean las áreas bajo las curvas
    plt.subplot(1, 2, 2)  # 1 fila, 2 columnas, segundo subgráfico
    sns.kdeplot(y, color='blue', label='Valores Reales', fill=True)
    sns.kdeplot(y_pred_linear_val, color='red', label='Predicciones Lineales', fill=True)
    sns.kdeplot(y_pred_rf_val, color='green', label='Predicciones Random Forest', fill=True)
    sns.kdeplot(y_pred_poly_val, color='orange', label='Predicciones Polinómicas', fill=True)
    plt.title('Densidad Predicciones Validación vs Reales')
    plt.xlabel('Valor')
    plt.ylabel('Densidad')
    plt.legend()
    plt.xlim(5, 15)

    plt.tight_layout()      # Ajustar automáticamente la disposición de los subgráficos para evitar superposiciones
    plt.gcf().canvas.manager.set_window_title('Menu Opciones')
    #plt.show()              # Mostrar la figura con los dos subgráficos
    plt.savefig("C:\\Users\\52477\\Documents\\ProyectoWizeLine\\reports\\figures\\twoSubplotsDensity.png")





# ================================================================================================
def dispersion_figure (data_Fr, cols_names_Arr) :
# ================================================================================================
    
    ''' 
    Diagramas de disperion por pares de variables tanto de atributos como objetivo
    Parámetros:
        data_Fr : dataframe con los datos a graficar 
        cols_names_Arr : nombre de las columnas del dataframe
    '''

    # Criterios :
    # Comparar la variable de salida vs las de entrada para tomar las de la mas clara relacion
    # Comparar entre sí variables de entrada para descartar las mas correlacionadas pues pueden ser redundantes 
    sns.set(style='whitegrid')
    sns.pairplot(data_Fr[cols_names_Arr], height=1)
    plt.gcf().canvas.manager.set_window_title('Menu Opciones')
    #plt.show()
    plt.savefig("C:\\Users\\52477\\Documents\\ProyectoWizeLine\\reports\\figures\\dispersion.png")




# ================================================================================================
def  twoSubplotsScatters_figure (predictions_df, Shell_weight, Rings_Real, Rings_Predicted, Rings, Diameter) :
# ================================================================================================

    ''' 
    Graficas de scatter a dos subprlots
    Parámetros:
        predictions_df  : 
        Shell_weight    :  
        Rings_Real      : 
        Rings_Predicted : 
        Rings           : 
        Diameter        : 
    '''

    # Crear la figura y los ejes
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Graficar la comparación de "Rings_Real" y "Rings_Predicted" en función de Shell_weight
    axs[0].scatter(predictions_df[Shell_weight], predictions_df[Rings_Real], color='blue', label=Rings_Real)
    axs[0].scatter(predictions_df[Shell_weight], predictions_df[Rings_Predicted], color='orange', label=Rings_Predicted)
    axs[0].set_title('Comparación Variable Objetivo Real vs Predicción')
    axs[0].set_xlabel(Shell_weight)
    axs[0].set_ylabel(Rings)
    axs[0].legend()

    # Graficar la comparación de "Rings_Real" y "Rings_Predicted" en función de Diameter
    axs[1].scatter(predictions_df[Diameter], predictions_df[Rings_Real], color='blue', label=Rings_Real)
    axs[1].scatter(predictions_df[Diameter], predictions_df[Rings_Predicted], color='orange', label=Rings_Predicted)
    axs[1].set_title('Comparación Variable Objetivo Real vs Predicción')
    axs[1].set_xlabel(Diameter)
    axs[1].set_ylabel(Rings)
    axs[1].legend()

    plt.tight_layout()                          # Ajustar automáticamente la disposición de los subgráficos para evitar superposiciones
    plt.gcf().canvas.manager.set_window_title('Menu Opciones')
    # plt.show()                                  # Mostrar la gráfica
    plt.savefig("C:\\Users\\52477\\Documents\\ProyectoWizeLine\\reports\\figures\\twoSubplotsScatters.png")



# ================================================================================================================================
def validation_figures (y, y_pred_linear_test, y_pred_poly_test, y_pred_rf_test, y_pred_linear_val, y_pred_poly_val, y_pred_rf_val) :
# ================================================================================================================================

    '''
    Graficas de validacion      
    '''
    
    # Crear gráfica de diagrama de caja 
    # Para comparar datos reales con las predicciones prueba y validacion de los modelos 
    # Se notará que todas las medias de las predicciones de las pruebas y validaciones de los 3 modelos modelos 
    # están un poco mas altas que la de los datos reales

    boxplot_figure2 (   [y, y_pred_linear_test, y_pred_poly_test, y_pred_rf_test, y_pred_linear_val, y_pred_poly_val, y_pred_rf_val], 
                        ['Reales', 'Pruebas Lineal', 'Pruebas Polinomial', 'Pruebas Random Forest', 'Validacion Lineal', 'Validacion Polinomial', 'Validacion Random Forest'] )
           

       
    # Se notará que la densidad de las predicciones polinomiales son las mas parecidas a los valores reales
    # pues las predicciones lineal y "random forest" tienen picos mas altos o mas de 1 pico

    twoSubplotsDensity_figure ( y,  y_pred_linear_test, y_pred_rf_test, y_pred_poly_test,
                                    y_pred_linear_val, y_pred_rf_val, y_pred_poly_val    ) 

    plt.savefig("C:\\Users\\52477\\Documents\\ProyectoWizeLine\\reports\\figures\\validation.png")        

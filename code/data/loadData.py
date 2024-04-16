
# Importar librerias
import numpy as np
import pandas as pd

    
    
def load_data () :   
    '''
    Obtencion de los datos desde un dataset en un archivo separado por comas 
    Impresion de la informacion y descripcion de los datos que contiene el dataset
    '''

    # Cargar los datos y los nombres de las columnas del dataset a analizar y sobre el que se realizarán predicciones
    data_Mtrx = pd.read_csv("C:\\Users\\52477\\Documents\\ProyectoWizeLine\\data\\row\\abalone.data")
    names_Arr = pd.array(['Sex','Length','Diameter','Height','Whole_weight','Shucked_weight','Viscera_weight','Shell_weight','Rings'])

    # Asignar nombres de columnas
    data_Mtrx.columns = names_Arr

    
   
    # Informacion y descripcion de los datos
    print ('Información y descripción de las características del dataset avalon.data')
    print ('------------------------------------------------------------------------')
    data_Mtrx.info()
    print ("\n")

    data_Mtrx.describe()
    print ("\n")
        
        
    return data_Mtrx, names_Arr

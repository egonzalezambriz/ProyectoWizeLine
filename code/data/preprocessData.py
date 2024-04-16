
# Importar librerias
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer





# ================================================================================================
def preprocess_data (data_Mtrx) :
# ================================================================================================

    ''' 
     Conversiones e imputaciones de datos
    '''

    # El atributo "Sex" que es categorico se convierte a 3 columnas de tipo "int"
    data_Mtrx = pd.get_dummies(data_Mtrx, columns=["Sex"], dtype=int)

    cols_names_Arr = ['Length','Diameter','Height','Whole_weight','Shucked_weight','Viscera_weight','Shell_weight','Rings','Sex_F','Sex_I','Sex_M']
    cols_ImputerNames_Arr = ['Length','Diameter','Height','Whole_weight','Shucked_weight','Viscera_weight','Shell_weight','Rings']

    # Se crea una instancia de un imputer para aplicar la media a los valores "nan" en la matriz con los datos y después guardarlos en un data frame
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    data_Fr = pd.DataFrame(imputer.fit_transform(data_Mtrx[cols_names_Arr]))
    data_Fr.columns = cols_names_Arr

    # Se aplica la media a los valores 0 en el data frame para todos los atributos excepto el de "Sex"
    column_means = data_Fr[cols_ImputerNames_Arr].mean()
    for col in cols_ImputerNames_Arr:
        for index, row in data_Fr.iterrows():
            if row[col] == 0:
                data_Fr.at[index, col] = column_means[col]


    return data_Mtrx, cols_names_Arr, cols_ImputerNames_Arr, data_Fr



# ================================================================================================
def remove_outliers(df, column, lower_percentile, upper_percentile):
# ================================================================================================
    """
    Esta función elimina los valores atípicos de un DataFrame en la columna especificada,
    utilizando percentiles como límites inferior y superior. Los percentiles son medidas estadísticas 
    utilizadas para dividir un conjunto de datos ordenados en cien partes iguales. 
    En otras palabras, un percentil indica el valor por debajo del cual cae un cierto porcentaje 
    de los datos en un conjunto de datos ordenado
    """

    # Calcula los percentiles definidos por "lower_percentile" y "upper_percentile" para la columna especificada
    # y devuelve los valores correspondientes para esos percentiles
    lower_value = df[column].quantile(lower_percentile)
    upper_value = df[column].quantile(upper_percentile)

    # Filtra el DataFrame para mantener solo las filas donde los valores estén dentro de los límites
    filtered_df = df[(df[column] >= lower_value) & (df[column] <= upper_value)]
        
    # Devuelve el DataFrame filtrado
    return filtered_df




import unittest
import pandas as pd
import sys
import os
from code.data.loadData import load_data 

class TestDataLoading(unittest.TestCase):

    # Prueba para verificar si se cargan los datos correctamente
    # -------------------------------------------------------------------
    def test_data_loading(self):
    # -------------------------------------------------------------------
        '''
        La prueba asegura de que la función load_data() cargue correctamente los datos y que devuelva un DataFrame de Pandas con las columnas correctamente etiquetadas
        '''
        
        # Obtener la ruta relativa al archivo 'abalone.data'
        file_name = os.path.join("data", "raw", "abalone.data")
        
        print ('file_name', file_name)
        data, column_names = load_data(file_name)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), len(pd.read_csv(file_name)))  # Asumiendo que los datos de prueba tienen la misma longitud
        



    # Prueba para verificar si los nombres de las columnas se asignan correctamente
    # -------------------------------------------------------------------
    def test_column_names(self):
    # -------------------------------------------------------------------

        '''
        La prueba verifica si los nombres de las columnas obtenidos al cargar los datos coinciden exactamente con los nombres de columnas esperados 
        '''

        if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
            file_name = sys.argv[1]
        else:
            file_name = ".\\data\\raw\\abalone.data"
            

        print ('file_name', file_name)
        data, column_names = load_data(file_name)
        self.assertEqual(list(data.columns), column_names)



    # Prueba para verificar si la columna 'Rings' se convierte a tipo int correctamente
    # -------------------------------------------------------------------
    def test_column_conversion(self):
    # -------------------------------------------------------------------

        '''
        Prueba verifica que la columna variable de salida (objetivo) tiene el tipo correcto
        '''

        if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
            file_name = sys.argv[1]
        else:
            file_name = ".\\data\\raw\\abalone.data"
            

        print ('file_name', file_name)            
        data, _ = load_data(file_name)
        self.assertEqual(data['Rings'].dtype, int)



    # -----------------------------------------------------      
    def test_data_types(self):
    # -----------------------------------------------------      
        '''
        La prueba asegura que las columnas en el DataFrame tengan los tipos de datos esperados según lo definido en el diccionario 'expected_types'
        Es decir se verifica si los datos se cargaron correctamente y si tienen los tipos de datos correctos
        '''
        
        if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
            file_name = sys.argv[1]
        else:
            file_name = ".\\data\\raw\\abalone.data"
            

        print ('file_name', file_name)
        data, _ = load_data(file_name)
        expected_types = {'Sex': object, 'Length': float, 'Diameter': float, 'Height': float,
                          'Whole_weight': float, 'Shucked_weight': float, 'Viscera_weight': float,
                          'Shell_weight': float, 'Rings': int}
        
        for column, expected_type in expected_types.items():
            self.assertEqual(data[column].dtype, expected_type)
    


if __name__ == '__main__':
    unittest.main()
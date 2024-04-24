
import unittest
import pandas as pd
from ProyectoWizeLine.code.data.loadData import load_data 


# ----------------------------------------------------------------------
class TestDataLoading (unittest.TestCase) :
# ----------------------------------------------------------------------

    # -----------------------------------------------------
    def test_data_loading (self) :
    # -----------------------------------------------------
        '''
        La prueba asegura de que la función load_data() cargue correctamente los datos y que devuelva un DataFrame de Pandas con las columnas correctamente etiquetadas
        '''
        data, names = load_data()
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(names), len(data.columns))

        

    # -----------------------------------------------------
    def test_column_names (self) :
    # -----------------------------------------------------
        '''
        La prueba verifica si los nombres de las columnas obtenidos al cargar los datos coinciden exactamente con los nombres de columnas esperados 
        '''
        data, names = load_data()
        self.assertListEqual(list(data.columns), list(names))
        


    # -----------------------------------------------------      
    def test_data_info (self) :
    # -----------------------------------------------------
       '''
       La prueba verifica si un mensaje específico está presente en la salida generada por la función load_data(),
       por ejemplo para verificar si la función imprime información relevante sobre los datos cargados, 
       como una descripción o metadatos del conjunto de datos
       '''

       from io import StringIO                                  # Para capturar la salida de texto que normalmente se enviaría a la consola y redirigirla a un objeto de cadena en memoria
       import sys                                               # Proporciona acceso a algunas funciones y variables utilizadas por el intérprete 
        
       saved_stdout = sys.stdout                                # Guarda una copia de la salida
       try:
            out = StringIO ()                                   # Crea un nuevo objeto StringIO, que será utilizado para capturar la salida estándar
            sys.stdout = out                                    # Redirige la salida estándar al objeto StringIO
            
            data, _ = load_data()                               # Carga los datos. La salida de esta función será capturada por el objeto StringIO
            info_output = out.getvalue().strip()                # Obtener el contenido capturado por el objeto StringIO, convertir en cadena y eliminar los espacios en blanco al principio y al final
            
            # Verifica si la cadena especificada está presente en la salida capturada
            self.assertIn('Información y descripción de las características del dataset abalone.data', info_output)
        
       finally:
            
            sys.stdout = saved_stdout                           # Recuperar la salida estandar
    


    # -----------------------------------------------------      
    def test_data_types(self):
    # -----------------------------------------------------      
        '''
        La prueba asegura que las columnas en el DataFrame tengan los tipos de datos esperados según lo definido en el diccionario 'expected_types'
        Es decir se verifica si los datos se cargaron correctamente y si tienen los tipos de datos correctos
        '''
        data, _ = load_data()
        expected_types = {'Sex': object, 'Length': float, 'Diameter': float, 'Height': float,
                          'Whole_weight': float, 'Shucked_weight': float, 'Viscera_weight': float,
                          'Shell_weight': float, 'Rings': int}
        
        for column, expected_type in expected_types.items():
            self.assertEqual(data[column].dtype, expected_type)



if __name__ == '__main__':
    unittest.main()

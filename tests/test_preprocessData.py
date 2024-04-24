
import unittest
import pandas as pd
from ProyectoWizeLine.code.data.preprocessData import preprocess_data

# ----------------------------------------------------------------------
class TestPreprocessData (unittest.TestCase) :
# ----------------------------------------------------------------------

    # -----------------------------------------------------
    def test_preprocess_data_output (self) :
    # -----------------------------------------------------

        '''
        La prueba unitaria verifica si la función preprocess_data produce un DataFrame correctamente formateado 
        con las columnas adecuadas después de realizar las conversiones e imputaciones de datos
        '''
        # Crear un DataFrame de ejemplo
        data_Mtrx = pd.DataFrame({
            'Sex': ['F', 'M', 'I'],
            'Length': [1.0, 2.0, 3.0],
            'Diameter' : [0.20, 0.40, 0.60],
            'Height' : [0.010, 0.020, 0.030],
            'Whole_weight' : [0.10, 0.20, 0.30],
            'Shucked_weight' : [0.01, 0.02, 0.03],
            'Viscera_weight' : [0.001, 0.002, 0.003],
            'Shell_weight' : [0.099, 0.188, 0.277],
            'Rings' : [7, 8, 9]
        })
        
        # Llamar a la función preprocess_data
        processed_data, _, _, _ = preprocess_data(data_Mtrx)
        
        # Verificar si el DataFrame devuelto tiene las columnas esperadas
        expected_columns = ['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings', 'Sex_F', 'Sex_I', 'Sex_M']
        self.assertEqual(processed_data.columns.tolist(), expected_columns)





    # -----------------------------------------------------
    def test_preprocess_data_imputation (self) :
    # -----------------------------------------------------
       
        '''
        La prueba unitaria es para verificar si no hay valores faltantes en el DataFrame data_Fr 
        '''   
       
        # Crear un DataFrame de ejemplo con valores faltantes
        data_Mtrx = pd.DataFrame({
            'Sex': ['F', 'M', 'I'],
            'Length': [1.0, None, 3.0],
            'Diameter' : [None, 0.40, 0.60],
            'Height' : [0.010, 0.020, None],
            'Whole_weight' : [0.10, None, 0.30],
            'Shucked_weight' : [0.01, None, 0.03],
            'Viscera_weight' : [0.001, None, 0.003],
            'Shell_weight' : [None, 0.188, 0.277],
            'Rings' : [7, 8, None]
        })
        
        # Llamar a la función preprocess_data
        _, _, _, processed_data = preprocess_data(data_Mtrx)
        
        # Extraer el DataFrame procesado (data_Fr)
        data_Fr = processed_data

        # Verificar si no hay valores faltantes en el DataFrame procesado (data_Fr)
        self.assertFalse(data_Fr.isnull().any().any())

    


if __name__ == '__main__':
    unittest.main()

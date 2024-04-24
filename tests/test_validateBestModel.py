
# Importar librerias
import unittest
import pandas as pd
from code.models.validateBestModel import validateBestModel_withValuationDataset

# ---------------------------------------------------------------
class TestValidation(unittest.TestCase):
# ---------------------------------------------------------------

    # -------------------------------------------------
    def setUp(self):
    # -------------------------------------------------
        '''
        Se preparan los datos para las pruebas
        '''
        
        # Configura datos de prueba
        self.X_val = pd.DataFrame({'Shell_weight': [0.01, 0.02, 0.03], 'Diameter': [0.1, 0.2, 0.3]})
        self.y_pred_evaluation = [5, 10, 15]
        self.y_val = pd.Series([6, 9, 14])
        self.info_entry_model = 'LinearRegression'


    # -------------------------------------------------
    def test_output_type(self):
    # -------------------------------------------------
        '''
        Prueba que el tipo de dato del 1er dato que regresa la funcion 'validateBestModel_withValuationDataset' sea una instancia de un modelo
        '''

        # Verifica el tipo de dato de salida
        result, _ = validateBestModel_withValuationDataset(self.X_val, self.y_pred_evaluation, self.y_val, self.info_entry_model)
        self.assertIsInstance(result, str)

    # -------------------------------------------------
    def test_output_format(self):
    # -------------------------------------------------

        '''
        Prueba que el tipo de dato del 2do dato que regresa la funcion 'validateBestModel_withValuationDataset' sea un dataframe con las columnas correctas
        '''
        # Verifica el formato del DataFrame de salida
        _, result = validateBestModel_withValuationDataset(self.X_val, self.y_pred_evaluation, self.y_val, self.info_entry_model)
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertTrue(all(col in result.columns for col in ['Shell_weight', 'Diameter', 'Rings_Predicted']))



if __name__ == '__main__':
    unittest.main()

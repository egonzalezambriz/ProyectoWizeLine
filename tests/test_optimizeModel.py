
import unittest
from code.models.optimizeModel import gen_gridSearchHiperParam

# -------------------------------------------------------------------
class TestYourFunctions (unittest.TestCase) :
# -------------------------------------------------------------------

    '''
    Prueba unitaria de la funcion que genera las rejillas de busqueda de hiper parámetros
    '''

    # -----------------------------------------------
    def test_gen_gridSearchHiperParam (self) :
    # -------------------------------------------------------------------
        # Test for LinearRegression. Verifica si se genera correctamente el parámetro de la cuadrícula para LinearRegression
        param_grid_lr = gen_gridSearchHiperParam('LinearRegression')
        self.assertIsNotNone(param_grid_lr)                 # Asegura que el parámetro de la cuadrícula no sea None
        self.assertIsInstance(param_grid_lr, dict)          # Asegura que el parámetro de la cuadrícula sea un diccionario

        # Test for ElasticNet. Verifica si se genera correctamente el parámetro de la cuadrícula para ElasticNet
        param_grid_en = gen_gridSearchHiperParam('ElasticNet')
        self.assertIsNotNone(param_grid_en)                 # Asegura que el parámetro de la cuadrícula no sea None
        self.assertIsInstance(param_grid_en, list)          # Asegura que el parámetro de la cuadrícula sea una lista

        # Test for RandomForestRegressor. Verifica si se genera correctamente el parámetro de la cuadrícula para RandomForestRegressor
        param_grid_rf = gen_gridSearchHiperParam('RandomForestRegressor')
        self.assertIsNotNone(param_grid_rf)                 # Asegura que el parámetro de la cuadrícula no sea None
        self.assertIsInstance(param_grid_rf, list)          # Asegura que el parámetro de la cuadrícula sea una lista



if __name__ == '__main__':
    unittest.main()
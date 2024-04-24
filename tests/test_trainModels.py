
# Importar librerias
import unittest
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from code.models.trainModels import initialTrainModels


# -------------------------------------------------------------------
class TestInitialTrainModels (unittest.TestCase) :
# -------------------------------------------------------------------

    # --------------------------------------------------
    def test_initialTrainModels (self) :
    # --------------------------------------------------

        '''
        Prueba si la función initialTrainModels es capaz de entrenar modelos de regresión lineal, bosque aleatorio y ElasticNet correctamente
        '''

        # Configurar los datos de las caracteristicas de entrada y la variable objetivo
        X_train = [[0.330, 0.10]]               # 'Shell_weight' y 'Diameter'
        y_train = [8]                           # 'Rings'
        linear_model, rf_model, en_model = initialTrainModels(X_train, y_train)
        
        # Probar si los modelos están entrenados
        self.assertIsNotNone(linear_model)
        self.assertIsNotNone(rf_model)
        self.assertIsNotNone(en_model)

        # Probar si los modelos son del tipo esperado
        self.assertIsInstance(linear_model, LinearRegression)
        self.assertIsInstance(rf_model, RandomForestRegressor)
        self.assertIsInstance(en_model, ElasticNet)


if __name__ == '__main__':
    unittest.main()

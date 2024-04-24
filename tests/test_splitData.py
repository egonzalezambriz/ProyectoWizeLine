
import unittest
import numpy as np
from  ProyectoWizeLine.code.data.splitData import split_dataset


# ----------------------------------------------------------------------
class TestSplitDataset (unittest.TestCase) :
# ----------------------------------------------------------------------

    # -----------------------------------------------------
    def setUp (self) :
    # -----------------------------------------------------
        '''
        Prepara el entorno de prueba antes de ejecutar cada prueba individual
        En este caso específico, el método setUp se utiliza para configurar los datos de ejemplo 
        que se utilizarán en múltiples pruebas dentro de la clase de prueba
        '''
        
        # Configurar datos de ejemplo
        self.X = np.random.rand(100, 2)         # 100 muestras, 2 características
        self.y = np.random.rand(100)            # 100 muestras de variable objetivo
        self.test_size = 0.2



    # -----------------------------------------------------
    def test_split_dataset_sizes (self) :
    # -----------------------------------------------------

        '''
        Prueba unitaria que garantiza que la función split_dataset divida correctamente los datos en conjuntos de 
        entrenamiento, prueba y validación, y que los tamaños de los conjuntos divididos sean coherentes 
        con el tamaño total de los datos de entrada
        '''

        # Llamar a la función split_dataset
        _, _, X_train, y_train, X_test, y_test, X_val, y_val = split_dataset(self.test_size, self.X, self.y)

        # Verificar los tamaños de los conjuntos
        total_samples = len(self.X)

        # Verificar que la suma del tamaño de los 3 subconjuntos de datos es igual al total de datos 
        self.assertEqual(len(X_train) + len(X_test) + len(X_val), total_samples)
        self.assertEqual(len(y_train) + len(y_test) + len(y_val), total_samples)




    # -----------------------------------------------------
    def test_split_dataset_proportions (self) :
    # -----------------------------------------------------

        '''
        Esta prueba garantiza que la función split_dataset divida correctamente los datos en conjuntos de 
        entrenamiento, prueba y validación, y que los tamaños de los conjuntos de prueba y entrenamiento 
        estén en proporción con el tamaño total de los datos de entrada y el tamaño de prueba especificado
        '''
        
        # Llamar a la función split_dataset
        _, _, X_train, y_train, X_test, y_test, X_val, y_val = split_dataset(self.test_size, self.X, self.y)

        # Calcular tamaños esperados
        total_samples = len(self.X)
        expected_test_size = int(total_samples * self.test_size)
        expected_train_size = int(total_samples * (1 - self.test_size) * (1 - self.test_size))

        # Verificar los tamaños de los conjuntos que conicidan con los tamañaos esperados
        self.assertEqual(len(X_test), expected_test_size)
        self.assertEqual(len(X_train), expected_train_size)

    


if __name__ == '__main__':
    unittest.main()

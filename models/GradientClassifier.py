import logging
from sklearn.ensemble import GradientBoostingClassifier
from preprocessdata import preprocesssing as pre
from utils import utils, LoadData


class GClassifier(GradientBoostingClassifier):
    def __init__(self, n_estimators=250, escenario='11', k=5, test_size=0.2):
        super().__init__(n_estimators=n_estimators)
        self.escenario = f"./database-preprosesing/smote/{escenario}/minmax/{escenario}.minmax_smote.pickle"
        self.k = k
        self.test_size = test_size

    def fit(self, X, y):
        return super().fit(X, y)

    def predict(self, X):
        return super().predict(X)

    def score(self, X, y):
        return super().score(X, y)

    """
        Método de preprocesamiento y carga de los datos, la base de datos se encuentra dividida en 13 escenarios 
        lo que hace que sea necesario entrenar y probar el algoritmo con el mismo escenario. En un futuro se establecerá 
        una base de datos centralizada.

        Метод предварительной обработки и загрузки данных, база данных разделена на 13 сценариев.
         что делает необходимым обучение и тестирование алгоритма по одному и тому же сценарию. В будущем будет создано
         централизованная база данных.
        """
    def prepareData(self, cross_val=True):
        logging.info(f"Cargando datos preprocesados desde: {self.escenario}")
        data = './DETECTION SYSTEM/database/*[0123456789].binetflow'
        scalers = 'minmax'  # {'standard', 'minmax', 'robust', 'max-abs'}
        samplers = 'smote'  # 'under_sampling', 'over_sampling', 'smote', 'svm-smote' 'adasyn'
        pre.preprocessing(data, scalers, samplers)  # carga y preprocesamiento de los datos
        train_data, train_labels = utils.load_and_divide(self.escenario)  # carga de datos preprocesados
        if not cross_val:
            X_train, X_test, y_train, y_test = utils.train_test_split(train_data, train_labels, test_size=self.test_size)  # Separar los datos del entrenamiento
            train = (X_train, y_train)
            test = (X_test, y_test)
        else:
            train, test = utils.create_k(train_data, train_labels, self.k)  # conjuntos de entrenamiento prueba de validacion cruzada
        return train, test

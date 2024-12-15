import numpy as np
from utils import utils, LoadData
from sklearn.ensemble import RandomForestClassifier
from sklearn import decomposition
from preprocessdata import preprocesssing as pre


class RFClassifier(RandomForestClassifier):
    def __init__(self, n_estimators=250, max_features='sqrt', escenario="11", k=5, test_size=0.2):
        super().__init__(n_estimators=n_estimators,
                         max_features=max_features, criterion='entropy', random_state=0)
        self.escenario = "./database-preprosesing/smote/" + \
            escenario + "/minmax/" + escenario + ".minmax_smote.pickle"
        self.test_size = test_size
        self.k = k

    def fit(self, X_train, y_train):
        return super().fit(X_train, y_train)

    def predict(self, X_test):
        return super().predict(X_test)

    def score(self, X_test, y_test):
        return super().score(X_test, y_test)

    """
        Método de preprocesamiento y carga de los datos, la base de datos se encuentra dividida en 13 escenarios 
        lo que hace que sea necesario entrenar y probar el algoritmo con el mismo escenario. En un futuro se establecerá 
        una base de datos centralizada.

        Метод предварительной обработки и загрузки данных, база данных разделена на 13 сценариев.
         что делает необходимым обучение и тестирование алгоритма по одному и тому же сценарию. В будущем будет создано
         централизованная база данных.
        """

    def prepareData(self, cross_val=True):
        print(self.escenario)
        # data = './database/*[0123456789].binetflow'
        data = './database/0.binetflow'
        scalers = {'minmax'}  # {'standard', 'minmax', 'robust', 'max-abs'}
        # 'under_sampling', 'over_sampling', 'smote', 'svm-smote' 'adasyn'
        samplers = ['smote']
        # carga y preprocesamiento de los datos
        pre.preprocessing(data, scalers, samplers)
        train_data, train_labels = utils.load_and_divide(
            self.escenario)  # carga de datos preprocesados
        train = []
        test = []
        if not cross_val:
            X_train, X_test, y_train, y_test = utils.train_test_split(
                # Separar los datos del entrenamiento
                train_data, train_labels, test_size=self.test_size)
            train.append([X_train, y_train])
            test.append([X_test, y_test])
        else:
            # conjuntos de entrenamiento prueba de validacion cruzada
            train, test = utils.create_k(train_data, train_labels, self.k)
        return train, test

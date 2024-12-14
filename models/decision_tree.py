from sklearn.tree import DecisionTreeClassifier
from sklearn import decomposition
from utils import utils,LoadData
from preprocessdata import preprocesssing as pre

class DecisionTree(DecisionTreeClassifier):
    def __init__(self, escenario="3", test_size=0.2, k=5):
        super().__init__(criterion="entropy")
        self.escenario = "./database-preprosesing/smote/"+ escenario + "/minmax/" + escenario + ".minmax_smote.pickle"
        self.test_size = test_size
        self.k = k

    def fit(self, X, y, sample_weight = None):
        return super().fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        return super().predict(X)

    def score(self, X, y, sample_weight = None):
        return super().score(X, y, sample_weight=sample_weight)

    """
        Método de preprocesamiento y carga de los datos, la base de datos se encuentra dividida en 13 escenarios 
        lo que hace que sea necesario entrenar y probar el algoritmo con el mismo escenario. En un futuro se establecerá 
        una base de datos centralizada.

        Метод предварительной обработки и загрузки данных, база данных разделена на 13 сценариев.
         что делает необходимым обучение и тестирование алгоритма по одному и тому же сценарию. В будущем будет создано
         централизованная база данных.
        """
    def prepareData(self):
        train_data, train_labels= LoadData.loaddata(self.escenario)
        # Normalizar el conjunto de entrenamiento
        train_data = utils.normalize_data(train_data)

        # Aplicar PCA a los datos para reducir su dimensionalidad
        #pca = decomposition.PCA(n_components=2)
        #pca.fit(train_data)
        #train_data = pca.transform(train_data)

        # Separar los datos del entrenamiento
        X_train, X_test, y_train, y_test = utils.train_test_split(train_data, train_labels, test_size=self.test_size)
        return X_train, X_test, y_train, y_test

    def prepareData2(self, cross_val = True):
        print(self.escenario)
        data = './DETECTION SYSTEM/database/*[0123456789].binetflow'
        scalers = {'minmax'}  # {'standard', 'minmax', 'robust', 'max-abs'}
        samplers = ['smote']  # 'under_sampling', 'over_sampling', 'smote', 'svm-smote' 'adasyn'
        pre.preprocessing(data, scalers, samplers)  # carga y preprocesamiento de los datos
        train_data, train_labels = utils.load_and_divide(self.escenario)  # carga de datos preprocesados
        train = []
        test = []
        if not cross_val:
            X_train, X_test, y_train, y_test = utils.train_test_split(train_data, train_labels,
                                                                      test_size=self.test_size)  # Separar los datos del entrenamiento
            train.append([X_train, y_train])
            test.append([X_test, y_test])
        else:
            train, test = utils.create_k(train_data, train_labels,
                                         self.k)  # conjuntos de entrenamiento prueba de validacion cruzada
        return train, test
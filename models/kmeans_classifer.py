import numpy as np

from sklearn import decomposition
from sklearn.cluster import KMeans
from utils import utils
from preprocessdata import preprocesssing as pre

class Kmeans(KMeans):
    def __init__(self, n_clusters=8, random_state=0, escenario='3', k=5, test_size=0.2):
        super().__init__(n_clusters=n_clusters, random_state=random_state)
        self.escenario = "./database-preprosesing/smote/" + escenario + "/minmax/" + escenario + ".minmax_smote.pickle"
        self.test_size = test_size
        self.k = k

    def fit(self, X, y, sample_weight = None):
        return super().fit(X, y=y, sample_weight=sample_weight)

    def predict(self, X, sample_weight = None):
        return super().predict(X, sample_weight=sample_weight)

    def score(self, X, y, sample_weight = None):
        return super().score(X, y=y, sample_weight=sample_weight)



def kmeans_classifier(train_labels, train_data):
    # Prepara los datos
    train_data, train_labels = prepareData(train_data, train_labels)
    # Instancia la clase Kmeans 
    unique_labels = np.unique(train_labels)
    num_classes = len(unique_labels)
    clf = Kmeans(num_classes)

    centroids = clf.fit(train_data, train_labels)

    predicted_labels = clf.predict(train_data)

    Accuracy = 0
    for index in range(len(train_labels)):
	    # Dividir los datos usando Kmeans
	    current_label = train_labels[index]
	    predicted_label = predicted_labels[index]

	    if current_label == predicted_label:
		    Accuracy += 1

    return Accuracy / len(train_labels)



def prepareData(train_data, train_labels):
    # Normalizar el conjunto de entrenamiento
    train_data = utils.normalize_data(train_data)

    # Aplicar PCA a los datos para reducir su dimensionalidad
    pca = decomposition.PCA(n_components=2)
    pca.fit(train_data)
    train_data = pca.transform(train_data)
    train_data, train_labels = utils.shuffle_data(train_data, train_labels)
    return train_data, train_labels
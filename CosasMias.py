import numpy as np
import pandas as pd
from scipy.special import rel_entr
from scipy.stats import entropy
from sklearn.datasets import load_iris
from utils import utils, LoadData
from scipy.stats import entropy
from scipy.special import entr
import matplotlib.pyplot as plt
from scipy.special import kl_div
from scipy.spatial.distance import jensenshannon
from scipy.spatial import distance
from numpy.linalg import inv

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.decomposition import PCA as pca
from preprocessdata.preprocess import data_cleaning, data_transform, class_balance
import glob

'''
data es un array numpy con los valores de la columna
calculamos la frecuencia relativa de cada valor y con eso la entropia
'''


def calc_shannon(data):
    unique, counts = np.unique(data, return_counts=True)
    freqs = counts / len(data)

    # print(freqs)
    ent = entropy(freqs, base=2)
    return ent


def shannon_media_pond(nEnt, e='3'):
    dict_expVariance = {
        '1': np.array([0.35666183, 0.33186138, 0.23084867, 0.0689442, 0.00690669, 0.00388782, 0.00045744]),
        '2': np.array([0.39070403, 0.29763827, 0.19412535, 0.10445739, 0.0074889, 0.00468122, 0.00054815]),
        '3': np.array([0.37962076, 0.36040663, 0.13714998, 0.10509267, 0.00888123, 0.00591864, 0.00265547]),
        '4': np.array([0.43464399, 0.2596431, 0.17971887, 0.10614641, 0.01268697, 0.00538046, 0.00101864]),
        '5': np.array([0.5517209, 0.23879579, 0.09471701, 0.06118631, 0.04742159, 0.00298028, 0.00211656]),
        '6': np.array([0.39295522, 0.35146312, 0.1546751, 0.08739967, 0.00727655, 0.00451535, 0.00086129]),
        '7': np.array([0.53977881, 0.23428666, 0.09880852, 0.06035776, 0.05697433, 0.00638463, 0.0021552]),
        '8': np.array([0.35318432, 0.31048351, 0.17486612, 0.08469572, 0.06839798, 0.00421982, 0.00255682]),
        '9': np.array([0.43446992, 0.27494765, 0.19693704, 0.08609938, 0.00384923, 0.00260077, 0.00060566]),
        '10': np.array([0.38205831, 0.22908203, 0.18580794, 0.13168935, 0.06450667, 0.00557837, 0.00078845]),
        '11': np.array([0.44148548, 0.22797933, 0.14130709, 0.11591418, 0.06185796, 0.00963631, 0.00076141]),
        '12': np.array([0.52930001, 0.2322891, 0.09531061, 0.0703681, 0.06227798, 0.00747323, 0.00216146]),
        '13': np.array([0.36643762, 0.31170778, 0.17185557, 0.0814886, 0.0614083, 0.00353696, 0.00192767])}

    return np.sum(nEnt * dict_expVariance[e]) / np.sum(dict_expVariance[e])


def calc_shannon_interv(data, intervs):
    bins = np.linspace(min(data), max(data), intervs)
    histogram, bin_edges = np.histogram(data, bins=bins)
    probabilities = histogram / np.sum(histogram)

    return entropy(probabilities, base=2)


def renyi_entropy(data, alpha):
    unique, counts = np.unique(data, return_counts=True)
    probs = counts / len(data)
    if alpha == 1:
        # Use the limit for alpha -> 1, which is the Shannon entropy
        return -np.sum(entr(probs))
    else:
        return 1 / (1 - alpha) * np.log2(np.sum(probs ** alpha))


def prueba_entropia(e='3'):
    data, labels = utils.load_and_divide("./database-preprosesing/smote/" + e + "/minmax/" + e + ".minmax_smote.pickle",
                                         0, 5000)
    print(f'Escenario: {e}')

    for i in range(data.shape[1]):
        print(f'Entropia de Shannon para caracteristica {i}: {calc_shannon(data[:, i])}')
    print('---------------------------------------------------------------------------')
    for i in range(data.shape[1]):
        print(f'Entropia de Shannon por intervalos para caracteristica {i}: {calc_shannon_interv(data[:, i], 2000)}')
    print('---------------------------------------------------------------------------')
    for i in range(data.shape[1]):
        print(f'Entropia de Renyi para caracteristica: {i}: {renyi_entropy(data[:, i], 0.5)}')


def prueba_entropia2(e='3'):
    data, labels = utils.load_and_divide("./database-preprosesing/smote/" + e + "/minmax/" + e + ".minmax_smote.pickle",
                                         0, 5000)
    print(f'Escenario: {e}')

    ent = []
    for i in range(data.shape[1]):
        ent.append(calc_shannon_interv(data[:, i], 2000))
        print(f'Entropia de Shannon por intervalos para caracteristica {i}: {ent[i]}')

    nEnt = np.array(ent)
    print(f'Entropia General por media ponderada {shannon_media_pond(nEnt, e)}')


'''
Esta seria la suma acumulada de las diferencias entre los datos y la media
Si esta suma acumulada sobrepasa el umbral se detecta una anomalia
La suma acumulada no sera menor que cero
Se calcula para una caracteristica
Esta implementacion va a tener problemas con variables categoricas, quizas usar la mediana en ese caso?
'''


def ts_CUSUM(data, threshold, nThreshold, e):
    cusum = 0
    nCusum = 0

    y_axis = np.array([])
    nY_axis = np.array([])

    mean = np.mean(data)
    print(f"Mean of the data = {mean}")

    for i in range(len(data)):
        cusum = max(0, cusum + data[i] - mean)
        nCusum = min(0, nCusum + data[i] - mean)
        y_axis = np.append(y_axis, cusum)
        nY_axis = np.append(nY_axis, nCusum)
        # print(f"CUSUM iter: {i} = {cusum}")

    print(f"Final CUSUM: {cusum}")
    print(f"Final N_CUSUM: {nCusum}")

    anomalies = np.where(y_axis > threshold)[0]
    nAnomalies = np.where(nY_axis < nThreshold)[0]

    plt.plot(y_axis, color='blue')
    plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
    plt.plot(anomalies, y_axis[anomalies], 'ro', label='Anomalías')

    plt.plot(nY_axis, color='blue')
    plt.axhline(y=nThreshold, color='red', linestyle='--')
    plt.plot(nAnomalies, nY_axis[nAnomalies], 'ro')

    plt.title(f"Escenario {e}")
    plt.xlabel('Data')
    plt.ylabel('CUSUM')
    plt.legend()
    plt.grid()
    plt.show()


def testCUSUM(e='3'):
    data, labels = utils.load_and_divide("./database-preprosesing/smote/" + e + "/minmax/" + e + ".minmax_smote.pickle",
                                         0, 50)
    threshold = 10
    n = -5
    ts_CUSUM(data[:, 0], threshold, n, e)


def J_Distance(data, data1, intervs=2):
    # Distribucion p de datos recogidos
    bins = np.linspace(min(data), max(data), intervs)
    histogram, bin_edges = np.histogram(data, bins=bins)
    P = histogram / np.sum(histogram)
    print('Probabilidades: ', P)
    plt.bar(bin_edges[:-1], P, width=np.diff(bin_edges))
    plt.show()

    # Distibucion q de datos "normales"
    primeraMitad, segundaMitad = np.array_split(data1, 2)

    bins = np.linspace(min(primeraMitad), max(primeraMitad), intervs)
    histogram, bin_edges = np.histogram(primeraMitad, bins=bins)
    Q = histogram / np.sum(histogram)
    print('Probabilidades normales: ', Q)
    plt.bar(bin_edges[:-1], Q, width=np.diff(bin_edges))
    plt.show()

    # Calcular Distancia de Hellinger
    jeff = np.sum((np.sqrt(P) - np.sqrt(Q)) ** 2) ** 0.5
    # Calcular Distancia de Jensen–Shannon
    js = jensenshannon(P, Q)
    print(f"Jensen-Shannon: {js}")
    print('---------------------')
    print(f"Hellinger: {jeff}")


def test_Jeffrey(e='3'):
    e2 = e
    data, labels = utils.load_and_divide("./database-preprosesing/smote/" + e + "/minmax/" + e + ".minmax_smote.pickle",
                                         0, 2500)
    data1, labels1 = utils.load_and_divide(
        "./database-preprosesing/smote/" + e2 + "/minmax/" + e2 + ".minmax_smote.pickle",
        0, 5000)

    J_Distance(data[:, 0], data1[:, 0], intervs=750)


'''
Probamos con la columna 0
'''


def KL(e='3', e2='11', intervs=2):
    data, labels = utils.load_and_divide("./database-preprosesing/smote/" + e + "/minmax/" + e + ".minmax_smote.pickle",
                                         0, 2500)

    # define distributions
    # p = [0.10, 0.40, 0.50]
    # q = [0.80, 0.15, 0.05]

    # Distribucion p de datos recogidos
    bins = np.linspace(min(data[:, 0]), max(data[:, 0]), intervs)
    histogram, bin_edges = np.histogram(data[:, 0], bins=bins)
    p = histogram / np.sum(histogram)
    print('Probabilidades: ', p)
    plt.bar(bin_edges[:-1], p, width=np.diff(bin_edges))
    plt.show()

    # Distibucion q de datos "normales"
    # tamaño doble para que al filtrar los bots p y q queden del mismo tamaño
    data1, labels1 = utils.load_and_divide(
        "./database-preprosesing/smote/" + e2 + "/minmax/" + e2 + ".minmax_smote.pickle",
        0, 5000)

    primeraMitad, segundaMitad = np.array_split(data1, 2)

    bins = np.linspace(min(primeraMitad[:, 0]), max(primeraMitad[:, 0]), intervs)
    histogram, bin_edges = np.histogram(primeraMitad[:, 0], bins=bins)
    q = histogram / np.sum(histogram)
    print('Probabilidades normales: ', q)
    plt.bar(bin_edges[:-1], q, width=np.diff(bin_edges))
    plt.show()

    # calculate (P || Q)
    kl_divergence = np.sum(kl_div(p, q))
    kl_divergence2 = np.sum(kl_div(q, p))

    print('KL(P || Q): %.6f nats' % kl_divergence)

    # calculate (Q || P)
    print('KL(Q || P): %.3f nats' % kl_divergence2)


def mahala(e='3'):
    data, labels = utils.load_and_divide("./database-preprosesing/smote/" + e + "/minmax/" + e + ".minmax_smote.pickle",
                                         0, 10)

    #vector de medias
    mean = np.mean(data, axis=0)
    print(mean)

    # Calcular la matriz de covarianza y la inversa
    cov = np.cov(data, rowvar=False)
    inv_cov = inv(cov)

    # Para cada fila.
    dList = []
    for i in range(data.shape[0]):
        # Calcular la distancia de Mahalanobis entre la fila y la media
        d = distance.mahalanobis(data[i], mean, inv_cov)
        dList.append(d)
        print(f"La distancia de Mahalanobis para la observación {i} es {d}")

    ret = np.array(dList)
    finalMean = np.mean(ret)
    print(f'La media de todas las distancias es {finalMean}')
    return finalMean


def tryPCA():
    data = './database/*[0123456789].binetflow'

    scalerDict = {'standard': StandardScaler(with_mean=True, with_std=True),
                  'minmax': MinMaxScaler(),
                  'robust': RobustScaler(),
                  'max-abs': MaxAbsScaler()
                  }

    for escenario in glob.glob(data):
        name = escenario.split('\\')[-1]
        label = name.replace('.binetflow', '')

        X, y = data_cleaning(escenario=escenario, sep=',', label_scenarios=label).loaddata()

        scaler_model = scalerDict['minmax'].fit(X)
        train_data = scaler_model.transform(X)

        # Aplicar PCA
        pca_model = pca(n_components=7)
        results = pca_model.fit_transform(train_data)
        print(escenario)
        print('-/-/-/-/-/-/')
        # print(pca_model.explained_variance_)
        print(pca_model.explained_variance_ratio_)


if __name__ == '__main__':
    # KL('11', '11', 100)
    # testCUSUM('3')
    # test_Jeffrey('11')
    # tryPCA()
    prueba_entropia2('11')
    #mahala('11')

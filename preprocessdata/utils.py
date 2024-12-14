import os
import errno
import pandas as pd
from matplotlib import pyplot as plt


def create_folder(path):
    exist=False
    try:
        os.makedirs(path)
    except OSError as e:
        exist=True
        if e.errno != errno.EEXIST:
            exist=True
    return exist


def scatter_plot(data, labels, data_balanced, labels_balanced, escenario=None, path=None):
    columnas = []
    for i in range(data.shape[1]):
        columnas.append(i)

    data = pd.DataFrame(data, columns=columnas)
    data_balanced = pd.DataFrame(data_balanced, columns=columnas)
    data['label'] = labels
    data_balanced['label'] = labels_balanced
    x1 = data[data['label'] == 0]
    x2 = data[data['label'] == 1]
    x1_balanced = data_balanced[data_balanced['label'] == 0]
    x2_balanced = data_balanced[data_balanced['label'] == 1]

    n_columns = len(columnas)
    folder = path + '/figuras'
    create_folder(folder)
    for i in range(0, n_columns - 1):
        for j in range(i + 1, n_columns):
            save_file = folder + '/columns_' + str(i + 1) + str(j + 1) + '.png'
            plt.figure(figsize=(14, 5), dpi=100)
            plt.suptitle(escenario)
            plt.subplot(1, 2, 1)
            plt.scatter(x1[i], x1[j], marker='+', alpha=0.5, color='green', label='Normal')
            plt.scatter(x2[i], x2[j], marker='o', alpha=0.5, color='red', label='Botnet')
            plt.xlabel('Columna - ' + str(i + 1))
            plt.ylabel('Columna - ' + str(j + 1))
            plt.title('Original: Botnet=%s y Normal=%s' % (labels.tolist().count(1), labels.tolist().count(0)))
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.scatter(x1_balanced[i], x1_balanced[j], marker='+', alpha=0.5, color='green', label='Normal')
            plt.scatter(x2_balanced[i], x2_balanced[j], marker='o', alpha=0.5, color='red', label='Botnet')
            plt.xlabel('Columna - ' + str(i + 1))
            plt.ylabel('Columna - ' + str(j + 1))
            plt.title(
                'SMOTE: Botnet=%s y Normal=%s' % (labels_balanced.tolist().count(1), labels_balanced.tolist().count(0))
            )
            plt.legend()
            plt.savefig(save_file, bbox_inches='tight')
            plt.close()


def pca_model_plot(pca_model, path):
    fig, ax = pca_model.plot()
    fig.savefig(path)


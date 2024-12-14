import os
import glob
import joblib
import pickle
import numpy as np

from preprocessdata.utils import create_folder, scatter_plot, pca_model_plot
from preprocessdata.preprocess import data_cleaning, data_transform, class_balance
from colorama import Fore, init

init()

def preprocessing(data, scalers, samplers):
    """
    :param data: path de los escenarios con codificación basada en glob.
    :param scalers: lista de los escaladores seleccionados.
    :param samplers: lista de los tipos de muestreos seleccionados.
    :return:
    """
    for escenario in glob.glob(data):
        name = os.path.basename(escenario)  # Usa os.path.basename para obtener el nombre del archivo
        label = name.replace('.binetflow', '')

        print(label)

        for scaler in scalers:
            for sampler in samplers:
                base_folder = os.path.join('./database-preprosesing', sampler, label, scaler)
                models_folder = os.path.join(base_folder, 'models')

                name_scaler_model = os.path.join(models_folder, f"{label}.{scaler}_model.pickle")
                name_pca_model = os.path.join(models_folder, f"{label}.{scaler}_PCA_model.pickle")
                name_pca_plot = os.path.join(models_folder, f"{label}.{scaler}_PCs_plot.png")
                name_scaled_data = os.path.join(base_folder, f"{label}.{scaler}.pickle")
                name_sampled_data = os.path.join(base_folder, f"{label}.{scaler}_{sampler}.pickle")

                # Crea las carpetas de forma recursiva
                os.makedirs(models_folder, exist_ok=True)

                # Carga los datos
                X, y = data_cleaning(escenario=escenario, sep=',', label_scenarios=label).loaddata()

                # Escala y reduce la dimensionalidad de los datos
                X_trans, scaler_model, pca_model = data_transform(scaler=scaler, data=X).selection()

                # Almacenar los modelos del escalador y PCA
                joblib.dump(scaler_model, name_scaler_model)
                joblib.dump(pca_model, name_pca_model)

                if sampler == 'no_balanced':
                    # Almacenar los datos no balanceados
                    with open(name_scaled_data, 'wb') as file:
                        pickle.dump([np.array(X_trans), np.array(y)], file)
                else:
                    # Balancear y almacenar los datos
                    X_balanced, y_balanced = class_balance(sampler=sampler, data_x=X_trans, data_y=y).sampling()
                    with open(name_sampled_data, 'wb') as file:
                        pickle.dump([np.array(X_balanced), np.array(y_balanced)], file)

        print(Fore.GREEN + 'Processing of scenario {} done...'.format(label))

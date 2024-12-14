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
    :param scalers: lista de los escaldores seleccionados.
    :param samplers: lista de los tipos de muestreos seleccionados.
    :return:
    """
    for escenario in glob.glob(data):
        name = escenario.split('\\')[-1]
        label = name.replace('.binetflow', '')

        print(label)

        for scaler in scalers:
            for sampler in samplers:
                folders = './database-preprosesing/' + sampler + '/' + label + '/' + scaler + '/models'
                name_scaler_model = './database-preprosesing/' + sampler + '/' + label + '/' + scaler + '/models/' + label + '.' + scaler + \
                                    '_model.pickle '
                name_pca_model = './database-preprosesing/' + sampler + '/' + label + '/' + scaler + '/models/' + label + '.' + scaler + \
                                 '_PCA_model.pickle'
                name_pca_plot = folders + '/' + label + '.' + scaler + '_PCs_plot.png'
                name_scaled_data = './database-preprosesing/' + sampler + '/' + label + '/' + scaler + '/' + label + '.' + scaler + '.pickle'
                name_sampled_data = './database-preprosesing/' + sampler + '/' + label + '/' + scaler + '/' + label + '.' + scaler + '_' + \
                                    sampler + '.pickle'
                # Crea las carpertas de forma recursiva
                if(not create_folder(folders)):
                	# Carga los datos
                	X, y = data_cleaning(escenario=escenario, sep=',', label_scenarios=label).loaddata()

                	# Escala y reduce la dimensionalidad de los datos
	                X_trans, scaler_model, pca_model = data_transform(scaler=scaler, data=X).selection()

	                
	                # Almacenar los modelos del escaldo, pca
	                joblib.dump(scaler_model, r'' + name_scaler_model)
	                joblib.dump(pca_model, r'' + name_pca_model)

	                if sampler == 'no_balanced':
	                    # Almacenar los datos no balanceados
	                    file = open(name_scaled_data, 'wb')
	                    pickle.dump([np.array(X_trans), np.array(y)], file)
	                    file.close()
	                else:
	                    # Balancear y almacenar los datos
	                    X_balanced, y_balanced = class_balance(sampler=sampler, data_x=X_trans, data_y=y).sampling()
	                    file = open(name_sampled_data, 'wb')
	                    pickle.dump([np.array(X_balanced), np.array(y_balanced)], file)
	                    file.close()
        print(Fore.GREEN + 'Processing of scenario {} done...'.format(label))


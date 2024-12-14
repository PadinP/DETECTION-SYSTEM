import numpy as np
import random
import pickle as pck
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
import pandas as pd

# Método para normalizar los datos
def normalize_data(data):
	mean_features, var_features = compute_mean_and_var(data)
	std_features = np.sqrt(var_features)

	for index, sample in enumerate(data):
		data[index] = np.divide((sample - mean_features), std_features) 

	data = data.astype(float)
	return data

# Método para dividir el conjunto de entrenamiento
def train_test_split(X, y, test_size=0.2):
    train_data, train_labels = shuffle_data(X, y)

    split_i = len(y) - int(len(y) // (1 / test_size))
    x_train, x_test = train_data[:split_i], train_data[split_i:]
    y_train, y_test = train_labels[:split_i], train_labels[split_i:]

    return x_train, x_test, y_train, y_test

# Método para mezclar los datos
def shuffle_data(data, labels):
	if(len(data) != len(labels)):
		raise Exception("The given data and labels do NOT have the same length")
	lista=[]

	for i, j in zip(data, labels):
		lista.append(np.r_ [i, [j]])
	lista = np.array(lista)
	np.random.shuffle(lista)
	x,y = [],[]
	for i in lista:
		x.append(i[:-1])
		y.append(i[-1])

	return np.array(x), np.array(y).astype(int)

# Método para calcular la media y la varianza
def compute_mean_and_var(data):
	num_elements = len(data)
	if len(data.shape) > 1:
		total = [0] * data.shape[1]
	else:
		total = [0] * data.shape[0]
	for sample in data:
		total = total + sample
	mean_features = np.divide(total, num_elements)

	if len(data.shape) > 1:
		total = [0] * data.shape[1]
	else:
		total = [0] * data.shape[0]
	for sample in data:
		total = total + np.square(sample - mean_features)

	std_features = np.divide(total, num_elements)

	var_features = std_features ** 2

	return mean_features, var_features

# Método para calcular la distancia euclideana
def euclidean_distance(vec_1, vec_2):
	if(len(vec_1) != len(vec_2)):
		raise Exception("The two vectors do NOT have equal length")

	distance = 0
	for i in range(len(vec_1)):
		distance += pow((vec_1[i] - vec_2[i]), 2)

	return np.sqrt(distance)

def create_k(x, y, k=5):
	train=[]
	test=[]
	data, labels = shuffle_data(x,y)
	kf = KFold(n_splits=k)
	for train_index, test_index in kf.split(data):
		x_train, x_test= data[train_index], data[test_index]
		y_train, y_test= labels[train_index], labels[test_index]

		train.append([x_train,y_train])

		test.append([x_test,y_test])

	return train, test

def cross_validation_train(model, train, test):
	score_max=0
	avg=0
	avg_recall=0
	avg_presicion=0
	avg_f1=0
	best_model=None
	i=1
	for trainn, testss in zip(train,test):
		x_train=trainn[0]
		x_test=testss[0]

		y_train=trainn[1]
		y_test=testss[1]

		model.fit(x_train,y_train)
		predictions = model.predict(x_test)

		score = model.score(x_test,y_test)* 100
		r_score = recall_score(y_test,predictions) * 100
		f_score = f1_score(y_test,predictions) * 100
		p_score = precision_score(y_test,predictions) * 100

		avg = score + avg
		avg_recall = r_score + avg_recall
		avg_f1 = f_score + avg_f1
		avg_presicion = p_score + avg_presicion

		#print("True:",np.unique(y_test, return_counts=True))
		#print("Predict:",np.unique(predictions, return_counts=True))
		print(confusion_matrix(y_test,predictions))

		print("The score of group", i, "is:")
		print("Accuracy:", score)
		print("Recall:", r_score)
		print("F1:", f_score)
		print("Precision:", p_score)
		print("   ")

		i+=1
		if score>score_max:
			print("modelo salvado")
			score_max=score
			best_model=model

	avg=avg/len(train)
	avg_recall = avg_recall/len(train)
	avg_f1 = avg_f1/len(train)
	avg_presicion = avg_presicion/len(train)
	
	print("The final cross_val score is:")
	print("Accuracy:", avg)
	print("Recall:", avg_recall)
	print("F1:", avg_f1)
	print("Precision:", avg_presicion)
	with open ('resultados escenario 0 v2 .txt', 'a') as f:
		print("The final cross_val score is:",file=f)
		print("Accuracy:", avg,file=f)
		print("Recall:", avg_recall,file=f)
		print("F1:", avg_f1,file=f)
		print(f"Precision: ", avg_presicion ,"\n",file=f)
		
	
	#print("Max score is", score_max)
	return best_model, avg, score_max



def load_and_divide(data,since=0,until=5000):
	file = open (data,"ab+")
	file.seek (0)
	inst = pck.load (file)

	x_arr = []
	y_arr = []

	count1=0
	count2=0

	for x_ins, y_ins in zip(inst[0],inst[1]):
		if count1 < until and y_ins == 0:
			if count1>= since:
				x_arr.append(x_ins)
				y_arr.append(y_ins)
			count1+=1
		elif count2 < until and y_ins == 1:
			if count2>= since:
				x_arr.append(x_ins)
				y_arr.append(y_ins)
			count2+=1
		elif count1 == 5000 and count2 == 5000:
			break

	x_arr = np.array(x_arr)
	y_arr = np.array(y_arr)
	file.close()

	return x_arr,y_arr

def load_bots_conjunts(path_file):
	columns = ['','PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7','true_label','pred_label']
	x_columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7']
	df = pd.read_csv(path_file, sep=';', names=columns)
	x=np.array(df[x_columns])[1:].astype(float)
	y=np.array(df['true_label'])[1:].astype(int)
	return x,y

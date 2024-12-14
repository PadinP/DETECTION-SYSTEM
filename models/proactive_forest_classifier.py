import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import decomposition

from utils import utils
from proactive_forest.estimator import DecisionForestClassifier, ProactiveForestClassifier
from mlcomponent.component import Component as comp
from preprocessdata import preprocesssing as pre

class PFClassifier(ProactiveForestClassifier):

	
    def __init__(self,escenario = "11", k=5):
        super().__init__(n_estimators = 250,feature_selection = 'log', split_criterion= 'entropy')
        self.escenario = escenario
        self.component = None
        self.k = k
        self.validate = True

    def fit(self,X_train,y_train):
        return super().fit(X_train,y_train)

    def predict(self,X_test):
        return super().predict(X_test)

    def score(self,X_test,y_test):
        return accuracy_score(y_test, self.predict(X_test))

    def start_component (self, x_clasf, y_clasf, model):
    	if self.component is None:
    		self.component = comp(model)
    	self.component.add_data(x_clasf, y_clasf)
    	self.component.load_file_instances()
    	return self.component

    """
        Método de preprocesamiento y carga de los datos, la base de datos se encuentra dividida en 13 escenarios 
        lo que hace que sea necesario entrenar y probar el algoritmo con el mismo escenario. En un futuro se establecerá 
        una base de datos centralizada.

        Метод предварительной обработки и загрузки данных, база данных разделена на 13 сценариев.
         что делает необходимым обучение и тестирование алгоритма по одному и тому же сценарию. В будущем будет создано
         централизованная база данных.
        """
    def prepareData(self, cross_val = True):
        # data = 'C:/Dani/TESIS-Dany/TESIS/TESIS/DETECTION SYSTEM/database/*[0123456789].binetflow'
        data = 'database/*[0123456789].binetflow'
        scalers = {'minmax'}  # {'standard', 'minmax', 'robust', 'max-abs'}
        samplers = ['smote']  # 'under_sampling', 'over_sampling', 'smote', 'svm-smote' 'adasyn'
        pre.preprocessing(data, scalers, samplers) #carga y preprocesamiento de los datos
        print("sirve")
        train=[]
        test=[]

        try:
            self.escenario = "./database-preprosesing/smote/"+ self.escenario + "/minmax/" + self.escenario + ".minmax_smote.pickle"
            train_data, train_labels= utils.load_and_divide(self.escenario) #carga de datos preprocesados
                  
        except:
            print("File not found")
            return train, test

        try:
            train, test = utils.create_k(train_data, train_labels,self.k)# conjuntos de entrenamiento prueba de validacion cruzada   
        except:
            print("k is not integer")       

        return train, test


            
        

    	
    	
    	
    	
    	
    	
    	

    	
    		
    		
    		
    	
    		
    	
        
        
        
        	
        	
        	
        
        	
        
            
            
            
        
        	

        


        
        
        
        
	
	
	
	











   
    
 
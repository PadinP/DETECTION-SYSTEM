import numpy as np
import xlsxwriter

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
import pickle as pck

from utils import utils, LoadData
from models.random_forest_classifier import RFClassifier
from models.decision_tree import DecisionTree
from utils import utils
#from mlcomponent.component import Component as comp
from sklearn.model_selection import cross_val_predict, cross_val_score

def main(args=None):
	libro = xlsxwriter.Workbook('Ejecuciones RF.xls')
	hoja = libro.add_worksheet()

	col=0
	row=0

	for i in range(1):
		rfc= RFClassifier(escenario = "11", k=5)
		#X_train, X_test, y_train, y_test = rfc.prepareData()

		train,test = rfc.prepareData()
		model, avg , max_score = utils.cross_validation_train(rfc, train, test)
		#print("avg", avg, "max", max_score)
		#print(cross_val_score(rfc, X_train, y_train, cv=5))
		
		#rfc.fit(X_train,y_train)
		#predict = rfc.predict(X_test)
		#score = rfc.score( X_test,y_test)

		#file = open ("./files/model_rf.pckl","ab+")
		#pck.dump (rfc, file)
		#file.close ()

		#uniq, count = np.unique(y_test, return_counts=True)
		#print(uniq, count)
		#uniq, count = np.unique(predict, return_counts=True)
		#print(uniq, count)

		#print("The Score of the Random Forest Classifier is", score * 100)
		

		#x,y = utils.load_and_divide("./database-preprosesing/3/standard/3.standard_smote.pickle")
		#predict = rfc.predict(x)
		#score1 = rfc.score(x,y)
		#print("The Score of the Random Forest Classifier is", score1 * 100)

		#print(accuracy_score(y_test, predict)*100)
		#com = comp(X_test, predict,rfc)
		#com.fit_component()


		#hoja.write(row,col,  "Ejecución-" + str(i+1))
		#hoja.write(row,col+1, score * 100)
		#row+=1
		
	libro.close()



if __name__ == "__main__":
    main()

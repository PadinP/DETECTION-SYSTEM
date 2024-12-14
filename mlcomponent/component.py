import numpy as np
import pickle as pck
import random as ram

from mlcomponent.stadistics import Stats as st
from utils import utils, LoadData
from models.decision_tree import DecisionTree
from mlcomponent.metrics import Metric as met


class Component:


	def __init__(self, classification_model=None):
		self.x_instances = []
		self.y_instances = []
		self.x_positives = []
		self.positive_reclasifcation = []
		self.metrics_characterization = []
		self.pf_model = classification_model
		self.dt = None
		self.characterization_database = None

	def validate(self, tipe, x_new = [], y_new = []):
		if tipe == 1:
			if len (y_new) == 0 or len (x_new) == 0:
				print("New data array is empty")
				return False
		if tipe == 2:
			if len (self.x_instances) == 0:
				print("Array data is empty")
				return False
			if len (self.x_positives) == 0:
				print("No human user data exists")
				return False 
		if tipe == 3:
			if len (self.metrics_characterization) == 0:
				print ("Lista de de metricas de caracterización vacia")
				return False
			elif self.pf_model is None:
				print("Empty characterization metrics array")
				return False
			
		return True

	def show(self, lista):
		uniq, count = np.unique(lista, return_counts=True)
		try:
			print("Number of human users in reclassification:", count[0])
		except :
			print("Number of human users in reclassification:", 0)
		try:
			print("Number of bots users in reclassification:", count[1])
		except:
			print("Number of bots users in reclassification:", 0)

	def split_x_y(self, instances):
		self.x_instances = []
		self.y_instances = []
		for i in instances:
			self.x_instances.append (i [:-1])
			self.y_instances.append (i [-1])
		self.x_instances = np.array (self.x_instances)
		self.y_instances = np.array (self.y_instances)

	def load_file_instances(self):
		file = open ("./files/file_clasf_pf_bueno.pckl","ab+")
		file.seek (0)
		inst = np.array ([])
		try:
			inst = pck.load (file)
			self.split_x_y (inst)
		except:
			print ("Created file of instances classified by proactive forest")
		finally:
			file.close ()
		return inst


	def save_data_new (self, data_new):
		instances_file = self.load_file_instances () 
		try:
			data_new = np.r_[instances_file,data_new]
		except:
			pass		
		file = open ("./files/file_clasf_pf_bueno.pckl","wb")
		pck.dump (data_new, file)
		file.close ()

	def add_data(self, x_new, y_new): #anade las instancias clasificadas en pf al fichero
		data_new=[]
		for i,j in zip (x_new, y_new):
			data_new.append (np.r_ [i, [j]])

		data_new = np.array (data_new)
		self.save_data_new (data_new)

	def simulate_positives (self, x, porcent_cant_min, porcent_cant_max):		
		cant = len(x)
		cant_atributes = x.shape[1]
		cant_modif = int (ram.uniform(porcent_cant_min, porcent_cant_max) * cant)
		print("Number of modified instances", cant_modif)
		max_value = np.max(self.x_instances)
		min_value = np.min(self.x_instances)

		for index in range(cant_modif):
			colum = ram.randint(1,cant_atributes)
			new_value = ram.uniform(min_value, max_value)
			x [index][colum-1] = new_value
		return x


	def set_positives(self, x_clasf, y_clasf): #metodo que devuelve las instancias clasificadas como humanos
		for i,j in zip (x_clasf, y_clasf):
			if j == 0:
				self.x_positives.append (i)
		self.x_positives = self.simulate_positives (np.array (self.x_positives), 0.4, 0.6)

	def reclasification(self): #metodo para la reclasificacion de las instancias obtenidas del modelo pf
		if self.validate (2):
			self.dt = DecisionTree ()
			self.dt.fit (self.x_instances, self.y_instances)
			self.positive_reclasifcation = self.dt.predict (self.x_positives)
			self.show(self.positive_reclasifcation)
		else:
			print ("Cannot fit the tree")
		

	def set_characterization_label(self): #metodo para asignar la etiqueta a la caracterizacion del conjunto de instancias
		if self.validate(3):
			pf_classification = self.pf_model.predict (self.x_positives)
			self.metrics_characterization.append (0) #etiqueta usuarios humanos
			for i in pf_classification:
				if i == 1:
					self.metrics_characterization [-1] = 1 #etiqueta usuarios bots
					break
			print ("Characterization label:", self.metrics_characterization [-1])

	def refresh_characterization_database(self, row): #metodo que actualiza una fila de la base de hechos
		self.characterization_database = open ("./files/characterization_database.txt","r+")
		lines = self.characterization_database.readlines ()
		for index,l in enumerate (lines):
			if l [:-2] == row [:-1] and l[-2] != row[-1]:
				l = l [:-2]
				l += row [-1] + "\n"
				lines [index] = l
				self.characterization_database.seek (0)
				self.characterization_database.writelines (lines)
				print ("Updated row to database")
				self.characterization_database.close ()
				return True
			elif l [:-2] == row [:-1] and l[-2] == row[-1]:
				print("Data already exist")
				return True
		self.characterization_database.close ()
		return False

	def add_row(self, row): #metodo que anade una fila a la base de hechos
		self.characterization_database = open ("./files/characterization_database.txt","a")
		self.characterization_database.write (row + "\n")
		self.characterization_database.close ()

	def save_data_characterization(self): #metodo para salvar la caracterizacion del cojunto de datos en la base de hechos
		if self.validate (3):
			row = ""
			for i in self.metrics_characterization:
				row += str (i) + ";"
			row = row [:-1]
			try:
				if not self.refresh_characterization_database (row):
					self.add_row (row)
					print ("Added row to database")
			except:
				print ("Data base is created")
				self.add_row (row)
				print ("Added row to database")

	def calculate_metrics(self): #se claculan las metricas
		if self.validate (2):
			metrics = met (self.x_positives, self.positive_reclasifcation)
			self.metrics_characterization = metrics.run_metrics () 

	def run_charact(self, x_clasf, y_clasf):
		if(len(x_clasf)==0 or len(y_clasf)==0):
			print("The users array is empty")
		else:

			self.set_positives (x_clasf, y_clasf) #se toman datos de usuarios humanos y se realizan cambios aleatorios

			self.reclasification () #fase 1 - se anaden las instancias clasificadas a un fichero y se reclasifican los positivos

			self.calculate_metrics()#fase 2 - se obtienen la descripcion del conjunto de datos, a partir de metricas

			self.set_characterization_label () # fase 3 -se asigna la etiqueta a la lista de metricas de caracterizacion del conjunto de datos 
			

			self.save_data_characterization () # fase 4 -se guarda la descripcion del conjunto de datos



		
		











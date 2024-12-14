import numpy as np
import sklearn as sklearn


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from problexity import classification
#from problexity import feature_based, linearity, neighborhood, dimensionality, class_imbalance


class Metric:


	def __init__(self, x_instances, y_instances):
		self.x = x_instances
		self.y = y_instances
		self.f1=None
		self.f2=None
		self.f3=None
		self.f4=None
		self.l1=None
		self.l2=None
		self.l3=None
		self.n1=None
		self.n2=None
		self.n3=None
		self.n4=None
		self.t1=None
		self.t2=None
		self.t3=None
		self.t4=None
		self.c1=None
		self.c2=None

		self.x_neigh=None
		self.y_neigh=None


	def run_f1(self):
		try:
			self.f1=classification.feature_based.f1(self.x, self.y)
			if np.math.isnan(self.f1):
				self.f1 = 0.0
		except:
			self.f1=0.0
		

	def run_f2(self):
		try:
			self.f2=classification.feature_based.f2(self.x, self.y)
			if np.math.isnan(self.f2):
				self.f2 = 0.0
		except:
			self.f2=0.0

	def run_f3(self):
		try:
			self.f3=classification.feature_based.f3(self.x, self.y)
			if np.math.isnan(self.f3):
				self.f3 = 0.0
		except:
			self.f3=0.0
		

	def run_f4(self):
		try:
			self.f4=classification.feature_based.f4(self.x, self.y)
			if np.math.isnan(self.f4):
				self.f4 = 0.0
		except:
			self.f4=0.0
	
	def run_l1(self):
		try:
			self.l1 =classification.linearity.l1(self.x, self.y)
			if np.math.isnan(self.l1):
				self.l1 = 0.0
		except:
			self.l1=0.0

	def run_l2(self):
		try:
			self.l2 = classification.linearity.l2(self.x, self.y)
			if np.math.isnan(self.l2):
				self.l2 = 0.0
		except:
			self.l2 = 0.0

	def run_l3(self):
		try:
			self.l3 = classification.linearity.l3(self.x, self.y)
			if np.math.isnan(self.l3):
				self.l3 = 0.0
		except :
			self.l3 = 0.0

	def run_n1(self):
		try:
			self.n1 =classification.neighborhood.n1(self.x_neigh, self.y_neigh)
			if np.math.isnan(self.n1):
				self.n1 = 0.0
		except :
			self.n1 = 0.0

	def run_n2(self):
		try:
			self.n2 = classification.neighborhood.n2(self.x_neigh, self.y_neigh)
			if np.math.isnan(self.n2):
				self.n2 = 0.0
		except:
			self.n2 = 0.0

	def run_n3(self):
		try:
			self.n3 = classification.neighborhood.n3(self.x_neigh, self.y_neigh)
			if np.math.isnan(self.n3):
				self.n3 = 0.0
		except:
			self.n3 = 0.0

	def run_n4(self):
		try:
			self.n4 = classification.neighborhood.n4(self.x_neigh, self.y_neigh)
			if np.math.isnan(self.n4):
				self.n4 = 0.0
		except:
			self.n4 = 0.0

	def run_t1(self):
		try:
			self.t1 = classification.neighborhood.t1(self.x_neigh, self.y_neigh)
			if np.math.isnan(self.t1):
				self.t1 = 0.0
		except:
			self.t1 = 0.0

	def run_t2(self):
		try:
			self.t2 = classification.dimensionality.t2(self.x, self.y)
			if np.math.isnan(self.t2):
				self.t2 = 0.0
		except:
			self.t2 = 0.0

	def run_t3(self):
		try:
			self.t3 = classification.dimensionality.t3(self.x, self.y)
			if np.math.isnan(self.t3):
				self.t3 = 0.0
		except:
			self.t3 = 0.0

	def run_t4(self):
		try:
			self.t4 = classification.dimensionality.t4(self.x, self.y)
			if np.math.isnan(self.t4):
				self.t4 = 0.0
		except:
			self.t4 = 0.0

	def run_c1(self):
		try:
			self.c1 =classification.class_imbalance.c1(self.x, self.y)
			if np.math.isnan(self.c1):
				self.c1 = 0.0
		except:
			self.c1 = 0.0

	def run_c2(self):
		try:
			self.c2 = classification.class_imbalance.c2(self.x, self.y)
			if np.math.isnan(self.c2):
				self.c2 = 0.0
		except:
			self.c2 = 0.0

	def validate_nan(self, metric_list):
		for element in metric_list:
			if np.math.isnan(element):
				element = 0.0 


	def run_metrics(self):

		if len(self.x) != 0 and len(self.y) != 0:

			if len(self.x > 500):
				self.x_neigh=self.x[:500]
				self.y_neigh=self.y[:500]

			self.run_f1 ()
			#print(self.f1)
			#if np.math.isnan(f1):
			#	f1 = 0.0

			self.run_f2 ()
			#print(self.f2)

			self.run_f3 ()
			#print(self.f3)

			self.run_f4 ()
			#print(self.f4)

			self.run_l1 ()
			#print(self.l1)

			self.run_l2 ()
			#print(self.l2)

			self.run_l3 ()
			#print(self.l3)

			self.run_n1 ()
			#print(self.n1)

			self.run_n2 ()

			self.run_n3 ()
			#print(self.n3)

			self.run_n4 ()
			#print(self.n4)

			self.run_t1 ()
			#print(self.t1)
			
			self.run_t2 ()
			#print(self.t2)

			self.run_t3 ()
			#print(self.t3)

			self.run_t4 ()
			#print(self.t4)

			self.run_c1 ()
			#print(self.c1)

			self.run_c2 ()
			#print(self.c2)



			print(" f1 = {:.3f}".format(self.f1), "f2 = {:.3f}".format(self.f2), "f3 = {:.3f}".format(self.f3), "f4 = {:.3f}".format(self.f4), "l1 = {:.3f}".format(self.l1),
				  "\n l2 = {:.3f}".format(self.l2), "l3 = {:.3f}".format(self.l3), "n1 = {:.3f}".format(self.n1), "n2 = {:.3f}".format(self.n2), "n3 = {:.3f}".format(self.n3),
				  "\n n4 = {:.3f}".format(self.n4), "t1 = {:.3f}".format(self.t1), "t2 = {:.3f}".format(self.t2), "t3 = {:.3f}".format(self.t3), "t4 = {:.3f}".format(self.t4),
				  "\n c1 = {:.3f}".format(self.c1), "c2 = {:.3f}".format(self.c2))

			return [self.f1, self.f2, self.f3, self.f4, self.l1, self.l2,\
			        self.l3, self.n1, self.n2, self.n3, self.n4, self.t1,\
			        self.t2, self.t3, self.t4, self.c1, self.c2]
		else:
			print ("No existen datos")
			return []

		
				
				
				
			
		

		
		
				

		

		
		       
		       












		


		
		




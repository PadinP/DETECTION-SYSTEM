import numpy as np
import sklearn as sklearn

class Stats ():
	def __init__(self, x_instances, y_instances, numberOfAttributes, numberOfClass, numberOfInstances):
		self.x_instances=x_instances
		self.y_instances=y_instances
		self.numberOfInstances=numberOfInstances
		self.numberOfClass=int(numberOfClass)
		self.numberOfAttributes=numberOfAttributes
		self.instancesPerClass = np.zeros(self.numberOfClass)
		self.meanPerAttxClass = np.zeros((self.numberOfAttributes, self.numberOfClass))
		self.varPerAttxClass = np.zeros((self.numberOfAttributes, self.numberOfClass))
		self.maxPerAttxClass = np.zeros((self.numberOfAttributes, self.numberOfClass))
		self.minPerAttxClass = np.zeros((self.numberOfAttributes, self.numberOfClass))
		

	def numberInstancesPerClass(self):
		for i in range(self.numberOfInstances):
			self.instancesPerClass[int(self.y_instances[i])] +=1

	def meanComputation(self):
		for i in range(self.numberOfInstances):
			for j in range(self.numberOfAttributes):
				self.meanPerAttxClass[j][int(self.y_instances[i])] = self.x_instances[i][j]

		for i in range(self.numberOfAttributes):
			for j in range(self.numberOfClass):
				self.meanPerAttxClass[i][j] /= self.instancesPerClass[j]

	def maxMinComputation(self):
		self.maxPerAttxClass -= 10000
		self.minPerAttxClass += 10000

		for i in range(self.numberOfInstances):
			for j in range(self.numberOfAttributes):
				self.minPerAttxClass[j][int(self.y_instances[i])] = min(self.minPerAttxClass [j][int(self.y_instances[i])], self.x_instances [i][j])
				self.maxPerAttxClass[j][int(self.y_instances[i])] = max(self.maxPerAttxClass [j][int(self.y_instances[i])], self.x_instances [i][j])

	def varianceComputation(self):
		for i in range(self.numberOfInstances):
			for j in range(self.numberOfAttributes):
				self.varPerAttxClass[j][int(self.y_instances[i])] += np.math.pow(self.x_instances[i][j] - self.meanPerAttxClass[j][int(self.y_instances[i])],2)
			

		for i in range(self.numberOfAttributes):
			for j in range(self.numberOfClass):
				self.varPerAttxClass[i][j] /= self.instancesPerClass[j] 

	def entropyClass(self):
		entropy=0

		for i in range(self.numberOfClass): 
			entropy -= self.instancesPerClass[i] / self.numberOfInstances * ((np.math.log(self.instancesPerClass[i]) / self.numberOfInstances) / np.math.log(2))

		return entropy

	def runStats(self):
		self.numberInstancesPerClass()
		self.meanComputation()
		self.varianceComputation()
		self.maxMinComputation()

	def getMean(self, att, classs):
		return self.meanPerAttxClass[att][classs]

	def getVariance(self, att, classs):
		return self.varPerAttxClass[att][classs]
	
	def getMax(self, att, classs): 
		return self.maxPerAttxClass[att][classs]
	
	def getMin(self, att, classs):
		return self.minPerAttxClass[att][classs]

	def getAll (self):
		print("Media: ", self.meanPerAttxClass)
		print("Varianza: ", self.varPerAttxClass)
		print("Mayor:", self.maxPerAttxClass)
		print("Menor:", self.minPerAttxClass)
		print("Instancias por clase:", self.instancesPerClass)


		






		



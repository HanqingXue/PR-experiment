import numpy as np
from numpy import *
import math
class Guass(object):
	"""docstring for Guass"""
	def __init__(self, mean, cov):
		self.mean = np.matrix(mean)
		self.cov = np.matrix(cov)

	def setGuassargs(self, mean, cov):
		self.mean = mean
		self.mean = cov


	def N(self, sample):
		sample = np.matrix(sample)
		D = self.cov.shape[0]
		indexone = 1 / pow(2 * np.pi, D / 2)
		indextwo = 1 / pow(np.linalg.det(self.cov), 0.5)
		covB = self.cov.I
		result = -0.5*(sample - self.mean)*(covB)*(sample - self.mean).T
		return indexone*indextwo*math.exp(result)

class GMM(object):
	"""docstring for GMM"""
	def __init__(self, Mnum):
		super(GMM, self).__init__()
		self.Mnum = Mnum #The number of Guass
		self.guassList = []
		self.meanList = []
		self.covList = []
		self.loaddata()
		self.piList = []
	
	def loaddata(self):
		self.meanList = np.loadtxt('meanarg.csv', delimiter=',')
		for i in range(0 , self.Mnum):
			covMatrix =  np.loadtxt('cov{}.csv'.format(str(i)), delimiter=',')
			self.covList.append(covMatrix)
		self.piList = [1.0/3, 2.0/3]
		for i in range(0, self.Mnum):
			guassModel = Guass(self.meanList[i], self.covList[i])
			self.guassList.append(guassModel)
	def algothrim(self):
		#E steps
        #M steps
		pass

'''
mean = np.loadtxt('meanarg.csv', delimiter=',')
cov  = np.loadtxt('guassarg.csv', delimiter=',')
sample = np.loadtxt('sampledemo.csv', delimiter=',')
'''

G2 = GMM(2)
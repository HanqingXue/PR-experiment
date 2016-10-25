import numpy as np
from numpy import *
import math
import copy
from numpy.lib.scimath import logn
from math import e
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse, Circle
import matplotlib.pyplot as plt
class Guass(object):
	"""docstring for Guass"""
	def __init__(self, mean, cov):
		self.mean = np.matrix(mean)
		self.cov = np.matrix(cov)

	def setGuassargs(self, mean, cov):
		self.mean = mean
		self.cov = cov

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
	def __init__(self, Mnum, trfname):
		super(GMM, self).__init__()
		self.Mnum = Mnum #The number of Guass
		self.guassList = []
		self.meanList = []
		self.covList = []
		self.piList = []
		self.traindata = []
		self.trfname = trfname
		self.likelihoods = 0.0
		self.loaddata()
		self.calLikelihood()
		self.algothrim()

	def loaddata(self):
		self.meanList = np.loadtxt('meanarg.csv', delimiter=',')
		for i in range(0 , self.Mnum):
			covMatrix =  np.loadtxt('cov{}.csv'.format(str(i)), delimiter=',')
			self.covList.append(covMatrix)

		self.piList = [1.0/3, 2.0/3]
		for i in range(0, self.Mnum):
			guassModel = Guass(self.meanList[i], self.covList[i])
			self.guassList.append(guassModel)

		self.traindata = np.loadtxt(self.trfname, delimiter=',')

	def algothrim(self):
			#E steps
		count = 0
		while True:
			#print '{}th itor'.format(str(count))
			count += 1
			oldlikehood = copy.deepcopy(self.likehoods)
			gauss =  self.guassList
			nxMatrix = []
			for i in range(0, self.Mnum):
				matrixRow = []
				for j in range(0, len(self.traindata)):
					Nx = self.guassList[i].N(self.traindata[j])
					matrixRow.append(Nx*self.piList[i])
				nxMatrix.append(matrixRow)
			nxMatrix = np.matrix(nxMatrix)
			#cal the pr and get the matrix
			colsum = []
			for j in range(0, nxMatrix.T.shape[0]):
				colsum.append(sum(nxMatrix.T[j]))
				pass
	
			prMatrixs = []
			for i in range(0, self.Mnum):
				prrow = []
				for j in range(0, len(self.traindata)):
					prrow.append(nxMatrix.A[i][j]/colsum[j])
					pass
				prMatrixs.append(prrow)
	
			prMatrixs = np.matrix(prMatrixs)
	
			Nk = []
			for i in range(0, self.Mnum):
				s = sum(prMatrixs[i].A)
				Nk.append(s)
			#update center
			for k in range(0, self.Mnum):
				for n in range(0,len(self.traindata)):
					self.meanList[k] += prMatrixs.A[k][n]*self.traindata[n]
				index = (1.0/Nk[k])
				self.meanList[k] *= index
	
			mean = copy.deepcopy(self.meanList)
	
			for k in range(0, self.Mnum):
				temp = []
				for n in range(0, len(self.traindata)):
					sub = np.matrix(self.traindata[n] - mean[k])
					temp.append((sub.T* sub)*prMatrixs.A[k][n])
	
				sumMatrix = temp[0]
				for i in range(1, len(temp)):
					sumMatrix = np.add(sumMatrix, temp[i])
	
	
				index = (1.0 / Nk[k])
				self.covList[k] = sumMatrix*index
	
			pik = map(lambda x:x/float(len(self.traindata)), Nk)
			self.piList = pik
			#print self.meanList
			#print self.likehoods
			for k in range(0, self.Mnum):
				self.guassList[k].setGuassargs(self.meanList[k], self.covList[k])
			self.calLikelihood()
			if self.likehoods > oldlikehood:
				continue
			else:
				break

	def plot(self):
		'''
		for item in self.traindata:
			#print item
			plt.scatter(item[0], item[1])
		plt.show()
		'''
		pass

	def writeResult(self):
		out = open('GMM_{}_Result.csv'.format(self.trfname[:-4]), 'w')
		alphas = map(str, self.piList)
		for i in range(0, self.Mnum):
			out.write('{0},{1},{2}'.format(alphas[i], str(self.meanList[i]), str(self.covList[i])))
			out.write('\n')

	def calLikelihood(self):
		sumlikehood = 0.0
		for n in range(0, len(self.traindata)):
			likehood = 0.0
			for k in range(0, self.Mnum):
				likehood += self.piList[k]*self.guassList[k].N(self.traindata[n])
			sumlikehood += logn(e, likehood)
		self.likehoods = sumlikehood
		pass

	def problailty(self, sample):
		prob = 0.0
		for i in range(self.Mnum):
			prob += self.piList[i]* self.guassList[i].N(sample)
		#prob = round(prob, 6)
		return prob

class TestGMM(object):
	"""docstring for TestGMM"""
	def __init__(self, testname):
		super(TestGMM, self).__init__()
		self.testname = testname
		self.loadtestdata()

	def loadtestdata(self):
		self.testData = np.loadtxt(self.testname, delimiter=',')
		 
	def test(self):
		GMM1 = GMM(2, 'Train1.csv')
		GMM2 = GMM(2, 'Train2.csv')
		error = 0
		for item in self.testData:
			if GMM1.problailty(item) >= GMM2.problailty(item):
				continue
			else:
				error += 1
		ac =  float(error)/float(len(self.testData))

		if self.testname == 'Test2.csv':
			pass
		else:
			ac = 1.0 - ac
		print 'Accuracy:{0}.'.format(str(ac))
		return ac
class plotGMM(object):
	"""docstring for plotGMM"""
	def __init__(self, GMM, fname):
		super(plotGMM, self).__init__()
		self.GMM = GMM
		self.fname = fname
		self.loaddata()

	def loaddata(self):
		data = np.loadtxt('Train1.csv', delimiter=',')
		data1 = np.loadtxt('Train2.csv', delimiter=',')
		fig = plt.figure()
		ax = fig.add_subplot(111)

		ell1 = Ellipse(xy=(0.0, 0.0), width=6, height=14, angle=-70.0, facecolor='yellow', alpha=0.3)
		ell2 = Ellipse(xy=(10, 10), width=9, height=16, angle=-30.0, facecolor='yellow', alpha=0.3)
		ell3 = Ellipse(xy=(2, 10), width=5, height=16, angle=-30.0, facecolor='red', alpha=0.3)
		ell4 = Ellipse(xy=(15, 20), width=2.4, height=16, angle=-65.0, facecolor='red', alpha=0.3)
		ax.add_patch(ell1)
		ax.add_patch(ell2)
		ax.add_patch(ell3)
		ax.add_patch(ell4)

		#help(Ellipse)
		x, y = 0, 0
		ax.plot(x, y, 'ro')

		plt.axis('scaled')
		# ax.set_xlim(-4, 4)
		# ax.set_ylim(-4, 4)
		plt.axis('equal')  # changes limits of x or y axis so that equal increments of x and y have the same length
		plt.axis([-10, 25, -20, 33])
		for item, item1 in zip(data, data1):
			plt.scatter(item[0], item[1], marker='o', color='r')
			plt.scatter(item1[0], item1[1], marker='^')
		plt.show()

if __name__ == '__main__':
	GMM1 = GMM(2, 'Train1.csv')
	plotGMM(GMM1, 'Train1.csv')

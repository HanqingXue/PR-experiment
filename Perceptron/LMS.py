#encoding=utf-8

import numpy as np
from numpy import *
from matplotlib import pyplot  as plt
class LMS(object):
	"""docstring for LMS"""
	def __init__(self):
		super(LMS, self).__init__()
		self.loaddata('LMS.csv')
		self.algorithm()

	def loaddata(self, fname):
		self.data =  np.loadtxt(fname, delimiter=",")

	def algorithm(self):
		data = mat(self.data)
		temp1 = (data.T)*data
		temp2 = temp1.I
		Y= temp2*self.data.T
		b = mat([1,1,1,1,1,1])
		A = (Y*b.T).T
		self.A = A.base
		print type(self.A[0])


	def plot_result(self):
		fig = plt.figure()
		ax1 = fig.add_subplot(111)
		ax1.set_title("Perceptron")
		plt.xlabel('x')
		plt.ylabel('y')
		'''
		color = ['r','b']
		label = ['label1', 'label2']
		marker = ['x', 'o']
		count = 0
		for index in self.data:
			print self.label[count]
			if self.label[count] == '1':
				label = 'o'
				pcolor = 'g'
			else:
				label = '^'
				pcolor = 'b'
			plt.scatter(index[0], index[1], marker=label,color=pcolor,alpha=0.6)

			count += 1
		'''

		x = range(0,3)
		numx = np.array(x)
		print type(self.A[1])
		y = -((0.91891892)/(0.32432432))*numx + (-1.13513514)/0.32432432
		plt.plot(x,y,marker='x',color='r')
		#plt.savefig('Perceptron.jpg')
		plt.show()
L = LMS()
L.plot_result()
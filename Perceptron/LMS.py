#encoding=utf-8

import numpy as np
from numpy import *
from matplotlib import pyplot  as plt
class LMS(object):
	"""docstring for LMS"""
	def __init__(self):
		super(LMS, self).__init__()
		self.label = {}
		self.datadic = {}
		self.loaddata('LMS.csv')
		self.algorithm()

	def loaddata(self, fname):
		self.matrix =  np.loadtxt(fname, delimiter=",")
		self.data =  np.loadtxt('demo.csv', delimiter=",")
		label = open('demoLabel.csv')
		count = 0
		for item in label:
			self.label[count] = item[0:-1]
			self.datadic[count] = self.data[count]
			count += 1
		pass
		

	def algorithm(self):
		data = mat(self.matrix)
		temp1 = (data.T)*data
		temp2 = temp1.I
		Y= temp2*self.matrix.T
		b = mat([1,1,1,1,1,1])
		A = (Y*b.T).T
		self.A = A.getA()


	def plot_result(self):
		fig = plt.figure()
		ax1 = fig.add_subplot(111)
		ax1.set_title("LMS")
		plt.xlabel('x')
		plt.ylabel('y')
		print self.data
		color = ['r','b']
		label = ['label1', 'label2']
		marker = ['x', 'o']
		count = 0
		for index in self.data:
			if self.label[count] == '1':
				label = 'o'
				pcolor = 'g'
			else:
				label = '^'
				pcolor = 'b'
			plt.scatter(index[0], index[1], marker=label,color=pcolor,alpha=0.6)
			count += 1


		x = range(0,3)
		numx = np.array(x)
		y = (-(self.A[0][1])/(self.A[0][2]))*numx + (-self.A[0][0])/self.A[0][2]
		k_value = round((-(self.A[0][1])/(self.A[0][2])), 2)
		k =  str(k_value)
		b =  str((-self.A[0][0])/self.A[0][2])
		ax1.annotate('y={0}*x+{1}'.format(k, b), (x[0], y[0]))
		plt.plot(x,y,marker='x',color='r')
		plt.savefig('LMS.jpg')
		plt.show()
L = LMS()
L.plot_result()
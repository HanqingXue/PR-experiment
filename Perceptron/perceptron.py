#encoding=utf-8

import numpy as np
import random
from matplotlib import pyplot as plt

class Perceptron(object):
	"""docstring for Perceptron"""
	def __init__(self,study_step=0.0001, study_total=10000):
		super(Perceptron, self).__init__()
		self.datadic = {}
		self.label = {}
		self.study_step = study_step                               # 学习步长
		self.study_total = study_total
		self.loaddata()

	def loaddata(self, fname ='demo.csv', labelfile='demoLabel.csv'):
		self.data =  np.loadtxt(fname, delimiter=",")
		label = open(labelfile)
		count = 0
		for item in label:
			self.label[count] = item[0:-1]
			self.datadic[count] = self.data[count]
			count += 1
		pass

	def train(self):
		train_size = len(self.label)
		datadim = len(self.data[0])
		w = np.zeros((datadim, 1))
		b = 0

		study_count = 0  # 学习次数记录，只有当分类错误时才会增加
		nochange_count = 0  # 统计连续分类正确数，当分类错误时归为0
		nochange_upper_limit = 100000

		while True:
			nochange_count += 1
			if nochange_count > nochange_upper_limit:
				break
			index = random.randint(0, train_size-1)
			point = self.data[index]
			label = self.label[index]
			yi = int(label)
			result = yi *(np.dot(point, w) + b )
			if result <= 0:
				item = np.reshape(self.data[index], (datadim, 1))
				w += item*yi*self.study_step
				b += yi * self.study_step
				study_count += 1
				if study_count > self.study_total:
					break
				nochange_count = 0
		self.w = w
		self.b = b
		return w, b

	def train_plot(self):
		fig = plt.figure()
		ax1 = fig.add_subplot(111)
		ax1.set_title("Perceptron")
		#help(ax1.annotate)
		plt.xlabel('x')
		plt.ylabel('y')
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


		x = range(0,3)
		numx = np.array(x)
		y = -((self.w[0])/(self.w[1]))*x - self.b/self.w[1]
		k = str(-((self.w[0])/(self.w[1])))
		b = str(- self.b/self.w[1])
		ax1.annotate('y={0}x +{1} '.format(k[1:-1],b[1:-1]), (x[0],y[0]))
		print 'k:{0}\n'.format(k)
		print 'b:{0}'.format(b)
		#print 'b:{0}\n'
		plt.plot(x,y,marker='x',color='r')
		plt.savefig('Perceptron.jpg')
		plt.show()
		pass

P = Perceptron(100)
print P.train()
P.train_plot()
#coding=utf-8
import random
import copy
from  matplotlib import pyplot as plt
import csv

class Kmean(object):
	"""docstring for Kmean"""
	def __init__(self, k):
		super(Kmean, self).__init__()
		self.k = k
		self.data = []
		self.dim = int
		self.centers = []
		self.clusters = {}
		self.readdata_csv()
		#self.readdata()
		self.setcenter()

	def readdata(self):
		data = open('simulate.txt')
		for point in data:
			point = point[0:-1].split()
			point = map(float, point)
			self.data.append(point)
		self.dim = len(self.data[0])

	def readdata_csv(self):
		f = open('ClusterSamples.csv')
		for item in f:
			item = item[0:-1].split(',')
			item = map(float, item)
			self.data.append(item)



	def dist(self, p1=list, p2=list):
		distsum = 0.0
		for i in range(0, len(p1)):
			distsum = pow(p2[i]- p1[i], 2) + distsum

		return pow(distsum, 0.5)

	def setcenter(self):

		self.centers  = random.sample(self.data, self.k)


	def updatecenter(self, cluster=list):
		if len(cluster) == 0:
			return random.sample(self.data, 1)

		newcenter = []
		for item in range(0, 784):
			newcenter.append(0)

		for item in cluster:
			newcenter = list(map(lambda x: x[0] + x[1], zip(newcenter, item)))

		result = []
		for item in newcenter:
			center = round(item/float(len(cluster)), 2)
			result.append(center)
		return result

	def algorithm(self):
		count = 0
		centers = self.centers
		while True:
			count += 1
			centers_copy = copy.deepcopy(centers)
			clusters = {}

			for i in range(0, self.k):
				clusters[i] = []


			for point in self.data:
				distdict = {}

				for i in range(0, len(clusters)):
					distdict[i] = self.dist(self.centers[i], point)

				for index in distdict:
					if distdict[index] == min(distdict.values()):
						clusters[index].append(point)
						break

			for i in range(0, self.k):
				centers[i] = self.updatecenter(clusters[i])
				pass

			print 'New center:'
			print '{}th itor'.format(str(count))
			if centers_copy == self.centers:
				f = open('out.txt', 'w')
				for item in clusters:
					f.write('{}class\n'.format(item))
					for point in clusters[item]:
						f.write(str(point)+'\n')
					f.write('\n')
				return clusters


	def plot_simulate(self):
		cluster = self.algorithm()
		x_axis = []
		y_axis = []
		fig = plt.figure()
		ax1 = fig.add_subplot(111)
		ax1.set_title("K-means")
		plt.xlabel('x')
		plt.ylabel('y')
		color = ['r','b']
		label = ['label1', 'label2']
		marker = ['x', 'o']
		for index in cluster:
			for point in cluster[index]:
					plt.scatter(point[0], point[1], c=color[index], marker=marker[index], label = label[index],alpha=0.6)
		plt.savefig('k-means_simulate.jpg')

K = Kmean(10)
K.algorithm()

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
		self.index2point = {}
		self.label = {}
		self.getlabel()
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
		count = 1
		for item in f:
			item = item[0:-1].split(',')
			item = map(float, item)
			self.data.append(item)
			self.index2point[count] = item
			count += 1

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

			print '{}th itor'.format(str(count))
			if centers_copy == self.centers:
				f = open('out_index.txt', 'w')
				for item in clusters:
					f.write('{}class\n'.format(item))
					for point in clusters[item]:
						key = self.search(self.index2point, point)
						f.write(str(key)+'\n')
					f.write('\n')
				return clusters

	def getlabel(self):
		f = open('SampleLabels.csv')
		count = 1
		for item in f:
			self.label[count] = item[:-1]
			count += 1

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

	def search(self, d =dict, val = list):
		for item in d:
			if d[item] == val:
				return item

	def plot_minist(self):
		f = open('out_index.txt')
		clusterSize = []
		clusters    = []
		clusterLabel = []
		for item in f:
			if 'class' in item:
				L = []
				continue

			if item == '\n':
				clusters.append(L)
				del(L)
				continue
			L.append(item)

		allLebal = []
		for item in clusters:
			L = []
			for elm in item:
				elm = elm.replace('\n', '')
				L.append(self.label[int(elm)])
			allLebal.append(L)
			del(L)

		print len(allLebal)

		f = open('count3.csv', 'w')
		count = 0
		for item in allLebal:
			f.write('class{},'.format(str(count)))
			for i in range(0, len(allLebal)):
				f.write(str(item.count(str(i)))+',')
			f.write('\n')
			count += 1 

		clusterSize = map(len, clusters)
		axis = []
		label = []
		for i in range(0, len(clusterSize)):
			axis.append(i)
			label.append('Class'+str(i))

		plt.title(u'The histogram of {} clusters using Minist Data (10000 samples)'.format(str(len(clusterSize))))
		plt.bar(axis, clusterSize, color = ['#4682b4'], label = label, align='center', width=0.3, alpha=0.8,linewidth=None)
		plt.savefig('Hist3.jpg')
		plt.show()

K = Kmean(10)
K.algorithm()
K.plot_minist()

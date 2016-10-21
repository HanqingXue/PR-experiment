# coding=utf-8
from sklearn.metrics import f1_score, classification_report
from sklearn.linear_model import Perceptron
import numpy as np
import time
from sklearn.metrics import roc_curve

data = np.loadtxt('TrainSamples.csv', delimiter=",")
label = np.loadtxt('TrainLabels.csv', delimiter=",")
test = np.loadtxt('TestSamples.csv', delimiter=',')
testLabel = np.loadtxt('TestLabels.csv', delimiter=',')

for i in range(0, 200):
	start = time.time()
	if i == 0:
		iter_times = 1
	else:
		iter_times = i * 50
	print 'iter_times{0}'.format(str(iter_times))
	classifier = Perceptron(n_iter=iter_times, eta0=0.001)
	classifier.fit_transform(data, label)
	predictions = classifier.predict(test)
	reportname = './report/{0}.txt'.format('report_{0}'.format(str(iter_times)))
	report = open(reportname, 'w')
	r = classification_report(testLabel, predictions)
	fpr, tpr, thresholds = roc_curve(testLabel, predictions, pos_label=2)
	report.write(r)
	end = time.time()
	report.write('time{0}'.format(str(end - start)))
	report.close()
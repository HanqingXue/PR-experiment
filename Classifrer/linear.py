# coding=utf-8
from sklearn.metrics import f1_score, classification_report
from sklearn.linear_model import Perceptron
import numpy as np
import time, os
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt
import numpy as np


data = np.loadtxt('./data/TrainSamples.csv', delimiter=",")
print data
label = np.loadtxt('./data/TrainLabels.csv', delimiter=",")
print label
test = np.loadtxt('./data/TestSamples1.csv', delimiter=',')
testLabel = np.loadtxt('./data/TestLabels1.csv', delimiter=',')
print 'iter_times{0}'.format(str(1000))
start = time.time()
classifier = Perceptron(n_iter=1000, eta0=0.001)
classifier.fit_transform(data, label)
predictions = classifier.predict(test)
reportname = 'Perceptron.txt'
report = open(reportname, 'w')
r = classification_report(testLabel, predictions)
fpr, tpr, thresholds = roc_curve(testLabel, predictions, pos_label=2)
report.write(r)
end = time.time()
report.write('time{0}'.format(str(end - start)))
report.close()
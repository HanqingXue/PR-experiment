import numpy as np
import time
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report

data = np.loadtxt('./data/TrainSamples.csv', delimiter=",")
label = np.loadtxt('./data/TrainLabels.csv', delimiter=",")
test = np.loadtxt('./data/TestSamples1.csv', delimiter=',')
testLabel = np.loadtxt('./data/TestLabels1.csv', delimiter=',')
start = time.time()
classifier = SVC(kernel="linear", C=0.025)
classifier.fit(data, label)
predictions = classifier.predict(test)
reportname = 'svmlinar.txt'
report = open('./result/'+reportname, 'w')
r = classification_report(testLabel, predictions)
report.write(r)
end = time.time()
report.write('time{0}'.format(str(end - start)))
report.close()

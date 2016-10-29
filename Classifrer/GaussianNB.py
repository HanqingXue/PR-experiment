import numpy as np
from sklearn.naive_bayes import GaussianNB
import time
from sklearn.metrics import classification_report

data = np.loadtxt('./data/TrainSamples.csv', delimiter=",")
label = np.loadtxt('./data/TrainLabels.csv', delimiter=",")
test = np.loadtxt('./data/TestSamples1.csv', delimiter=',')
testLabel = np.loadtxt('./data/TestLabels1.csv', delimiter=',')
start = time.time()
classifier = GaussianNB()
classifier.fit(data, label)
predictions = classifier.predict(test)
reportname = 'GaussianNB'
report = open('./result/'+reportname, 'w')
r = classification_report(testLabel, predictions)
report.write(r)
end = time.time()
report.write('time{0}'.format(str(end - start)))
report.close()
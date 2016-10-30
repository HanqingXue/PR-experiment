from sklearn.neural_network import MLPClassifier
import numpy as np
import time
import numpy as np
import time
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

data = np.loadtxt('./data/TrainSamples.csv', delimiter=",")

label = np.loadtxt('./data/TrainLabels.csv', delimiter=",")
test = np.loadtxt('./data/TestSamples1.csv', delimiter=',')
testLabel = np.loadtxt('./data/TestLabels1.csv', delimiter=',')
start = time.time()
classifier = MLPClassifier(alpha=1)
classifier.fit(data, label)
predictions = classifier.predict(test)
reportname = 'NN1.txt'
#report = open('./result/'+reportname, 'w')
print predictions
np.savetxt('Result.csv', predictions, fmt="%d")
r = classification_report(testLabel, predictions)
print r
#report.write(r)
end = time.time()
#report.write('time{0}'.format(str(end - start)))
#report.close()

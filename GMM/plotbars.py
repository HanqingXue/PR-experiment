#codinf=utf-8
import numpy as np
from matplotlib import pyplot as plt

'''
data = np.loadtxt('total.csv', delimiter=',', skiprows=1, usecols=(0,1,2))
pr = []
recall = []
fscore = []
for item in data:
	print item
	pr.append(item[0])
	recall.append(item[1])
	fscore.append(item[2])
	pass
'''
pr = [0.93, 0.94, 0.95, 0.95, 0.95]
print pr

plt.figure(figsize=(9, 6))
n = 5
X = np.arange(n) + 1
plt.bar(X, pr, width = 0.2,facecolor = 'lightskyblue',edgecolor = 'white', label='Precision')
plt.bar(X+0.2, pr, width = 0.2,facecolor = 'yellowgreen',edgecolor = 'white', label='Recall')
plt.bar(X+0.4, pr, width = 0.2,facecolor = '#ff6666',edgecolor = 'white', label='F1-score')

for x,y in zip(X, list(pr)):
    plt.text(x+0.1, y+0.05, '%.2f' % y, ha='center', va= 'bottom')

for x,y in zip(X,list(pr)):
    plt.text(x+0.3, y+0.05, '%.2f' % y, ha='center', va= 'bottom')

for x,y in zip(X,list(pr)):
    plt.text(x+0.5, y+0.05, '%.2f' % y, ha='center', va= 'bottom')
plt.title('5 group GMM experiment')
claslabel = []
for i in range(1, 10):
	claslabel.append('{0}Guass'.format(str(i)))
x = map(lambda x:x+0.3, X)
plt.xticks(x, claslabel)
plt.ylim(0,+1.25)
plt.legend()
plt.savefig('Guasstwo.png', papertype='a4')
help(plt.savefig)
plt.show()
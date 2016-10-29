#coding=utf-8
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
data=np.loadtxt('TestSamples.csv', delimiter=',')
label = np.loadtxt('TestLabels.csv', delimiter=',')
'''
pca=PCA()
pca.fit(data)
#print pca.components_  #返回模型的各个特征向量
print pca.explained_variance_ratio_  #返回各个成分各自的方差百分比（贡献率）
'''
pca=PCA(3)
pca.fit(data)
low_d=pca.transform(data)  #用这个方法来降低维度
pca.inverse_transform(low_d)  #必要时，可以用这个函数来复原数据

ax=plt.subplot(111, projection='3d')
#for item in low_d:
color = ['r', 'g', 'b', 'y', 'm', 'c', 'k', '#AEEEEE', '#A020F0', '#00868B']
for i in range(0 , data.shape[0]):
	ax.scatter(low_d[i][0], low_d[i][1], low_d[i][2],c=color[int(label[i])], label=color[int(label[i])], alpha=0.7)

ax.set_title("PCA MNIST DATA INTO  3d SPACE AND CLASSIFY")
ax.set_zlabel('Z') #坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
handles, labels = ax.get_legend_handles_labels()
handles = set(handles)
label = set(label)
handles = list(handles)
label = list(label)
#ax.legend(handles, labels)
#ax.legend([1,2,3], ['r','g','b'])
#plt.savefig('PCA.jpg')
plt.show()
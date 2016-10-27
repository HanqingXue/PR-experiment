#coding=utf-8
# 自定义的高维高斯分布概率密度函数
from matplotlib import pyplot as plt
import numpy as np
def gaussian(x,mean,cov):
    dim = np.shape(cov)[0] #维度
    covdet = np.linalg.det(cov+np.eye(dim)*0.01) #协方差矩阵的秩
    covinv = np.linalg.inv(cov+np.eye(dim)*0.01) #协方差矩阵的逆
    xdiff = x - mean
    #概率密度
    prob = 1.0/np.power(2*np.pi,1.0*2/2)/np.sqrt(np.abs(covdet))*np.exp(-1.0/2*np.dot(np.dot(xdiff,covinv),xdiff))

    return prob


#作二维高斯概率密度函数的热力图
'''
mean = [1,1]
cov = [[1,2.3],
      [2.3,1.4]]
x,y = np.random.multivariate_normal(mean,cov,5000).T
cov = np.cov(x,y) #由真实数据计算得到的协方差矩阵，而不是自己任意设定
n=200
x = np.linspace(-6,6,n)
print type(x)
print type(y)
y = np.linspace(-6,6,n)
xx,yy = np.meshgrid(x, y)
print xx
print yy
zz = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        zz[i][j] = gaussian(np.array([xx[i][j],yy[i][j]]),mean,cov)
zz1 = np.loadtxt('meanarg.csv', delimiter=',')
# 选项origin='lower' 防止tuixan图像颠倒
#plt.xticks([5,100,195],[-5,0,5])
#plt.yticks([5,100,195],[-5,0,5])
#plt.scatter([50], [50], marker='^', color='r')
print help(plt.clabel)
gci = plt.imshow(zz,origin='lower')
plt.title(u'高斯函数的热力图',{'fontname':'STFangsong','fontsize':18})
plt.show()

'''


import numpy as np
import matplotlib.pyplot as plt

plt.figure(1)
plt.figure(2)
ax1 = plt.subplot(211)  # 在图表2中创建子图1
ax2 = plt.subplot(212)  # 在图表2中创建子图2

x = np.linspace(0, 3, 100)
for i in xrange(5):
	plt.figure(1)   # 选择图表1
	plt.plot(x, np.exp(i * x / 3))
	plt.sca(ax1)  # 选择图表2的子图1
	plt.plot(x, np.sin(i * x))
	plt.sca(ax2)  # 选择图表2的子图2
	plt.plot(x, np.cos(i * x))

plt.show()

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

delta = 0.2
x = np.arange(-3, 3, delta)
y = np.arange(-3, 3, delta)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

x=X.flatten()
y=Y.flatten()
z=Z.flatten()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.01)
plt.show()

x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)
xx, yy = np.meshgrid(x, y, sparse=True)
z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
h = plt.contourf(x,y,z)


#GDA
#author:Xiaolewen
import matplotlib.pyplot as plt
from numpy import *

#Randomly generate two cluster data of Gaussian distributions
mean0=[0,0]
cov=mat([[3,1],[1,1]])
x0=random.multivariate_normal(mean0,cov,500).T   #The first class point which labael equal 0
y0=zeros(shape(x0)[1])
#print x0,y0
mean1=[10,10]
cov=mat([[2,2],[2,5]])
x1=random.multivariate_normal(mean1,cov,300).T
y1=ones(shape(x1)[1]) #The second class point which label equals 1
#print x1,y1

x=array([concatenate((x0[0],x1[0])),concatenate((x0[1],x1[1]))])
y=array([concatenate((y0,y1))])
m=shape(x)[1]
#print x,y,m
#Caculate the parameters:\phi,\u0,\u1,\Sigma
phi=(1.0/m)*len(y1)
#print phi
u0=mean(x0,axis=1)
#print u0
u1=mean(x1,axis=1)
#print u1

xplot0=x0;xplot1=x1   #save the original data  to plot
x0=x0.T;x1=x1.T;x=x.T
#print x0,x1,x
x0_sub_u0=x0-u0
x1_sub_u1=x1-u1
#print x0_sub_u0
#print x1_sub_u1
x_sub_u=concatenate([x0_sub_u0,x1_sub_u1])
#print x_sub_u

x_sub_u=mat(x_sub_u)
#print x_sub_u

sigma=(1.0/m)*(x_sub_u.T*x_sub_u)
#print sigma

#plot the  discriminate boundary ,use the u0_u1's midnormal
midPoint=[(u0[0]+u1[0])/2.0,(u0[1]+u1[1])/2.0]
#print midPoint
k=(u1[1]-u0[1])/(u1[0]-u0[0])
#print k
x=range(-2,11)
y=[(-1.0/k)*(i-midPoint[0])+midPoint[1] for i in x]



#plot contour for two gaussian distributions
def gaussian_2d(x, y, x0, y0, sigmaMatrix):
    return exp(-0.5*((x-x0)**2+0.5*(y-y0)**2))
delta = 0.025
xgrid0=arange(-2, 10, delta)
ygrid0=arange(-2, 10, delta)
xgrid1=arange(3,11,delta)
ygrid1=arange(3,11,delta)
X0,Y0=meshgrid(xgrid0, ygrid0)   #generate the grid
X1,Y1=meshgrid(xgrid1,ygrid1)
Z0=gaussian_2d(X0,Y0,2,3,cov)
Z1=gaussian_2d(X1,Y1,7,8,cov)

#plot the figure and add comments
plt.figure(1)
plt.clf()
#plt.plot(xplot0[0],xplot0[1],'ko')
#plt.plot(xplot1[0],xplot1[1],'gs')
#plt.plot(u0[0],u0[1],'rx',markersize=20)
#plt.plot(u1[0],u1[1],'y*',markersize=20)
#plt.plot(x,y)
CS0=plt.contour(X0, Y0, Z0)
plt.clabel(CS0, inline=1, fontsize=10)
CS1=plt.contour(X1,Y1,Z1)
plt.clabel(CS1, inline=1, fontsize=10)
data = np.loadtxt('Train1.csv', delimiter=',')
data1 = np.loadtxt('Train2.csv', delimiter=',')
for item, item1 in zip(data, data1):
	plt.scatter(item[0],item[1], marker='o', color='r')
	plt.scatter(item1[0],item1[1], marker='^' )

plt.title("Gaussian discriminat analysis")
plt.xlabel('Feature Dimension (0)')
plt.ylabel('Feature Dimension (1)')
plt.show(1)
#coding=utf-8
import numpy as np
from matplotlib import pyplot as plt

fig,axes = plt.subplots(nrows=1,ncols=3,figsize=(4,12))

# 标准圆形
mean = [0,0]
cov = [[1,0],
      [0,1]]
x,y = np.random.multivariate_normal(mean,cov,5000).T
axes[0].plot(x,y,'x')
axes[0].set_xlim(-6,6)
axes[0].set_ylim(-6,6)

# 椭圆，椭圆的轴向与坐标平行
mean = [0,0]
cov = [[0.5,0],
      [0,3]]
x,y = np.random.multivariate_normal(mean,cov,5000).T
axes[1].plot(x,y,'x')
axes[1].set_xlim(-6,6)
axes[1].set_ylim(-6,6)

# 椭圆，但是椭圆的轴与坐标轴不一定平行
mean = [0,0]
cov = [[3,1],
      [1,3]]
x,y = np.random.multivariate_normal(mean,cov,5000).T
axes[2].plot(x,y,'x'); plt.axis('equal')
axes[2].set_xlim(-6,6)
axes[2].set_ylim(-6,6)

plt.show()
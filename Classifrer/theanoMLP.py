# -*-coding:utf-8-*-
import theano
from theano import *
import numpy as np
import matplotlib.pyplot as plt
import theano.tensor as T


# softmax函数
class softmax:
	# outdata为我们标注的输出，hiddata网络输出层的输入，nin,nout为输入、输出神经元个数
	def __init__(self, hiddata, outdata, nin, nout):
		self.w = theano.shared(value=np.zeros((nin, nout), dtype=theano.config.floatX), name='w');
		self.b = theano.shared(value=np.zeros((nout,), dtype=theano.config.floatX), name='b')

		prey = T.nnet.softmax(T.dot(hiddata, self.w) + self.b)  # 通过softmax函数，得到输出层每个神经元数值(概率)
		self.loss = -T.mean(T.log(prey)[T.arange(outdata.shape[0]), outdata])  # 损失函数
		self.para = [self.w, self.b]
		self.predict = T.argmax(prey, axis=1)
		self.error = T.mean(T.neq(T.argmax(prey, axis=1), outdata))

	# 输入层到隐藏层


class HiddenLayer:
	def __init__(self, inputx, nin, nout):
		a = np.sqrt(6. / (nin + nout))
		ranmatrix = np.random.uniform(-a, a, (nin, nout));
		self.w = theano.shared(value=np.asarray(ranmatrix, dtype=theano.config.floatX), name='w')
		self.b = theano.shared(value=np.zeros((nout,), dtype=theano.config.floatX), name='b')
		self.out = T.tanh(T.dot(inputx, self.w) + self.b)
		self.para = [self.w, self.b]

	# 传统三层感知器


class mlp:
	def __init__(self, nin, nhid, nout):
		x = T.fmatrix('x')
		y = T.ivector('y')
		# åå
		hlayer = HiddenLayer(x, nin, nhid)
		olayer = softmax(hlayer.out, y, nhid, nout)
		# åå
		paras = hlayer.para + olayer.para
		dparas = T.grad(olayer.loss, paras)
		updates = [(para, para - 0.1 * dpara) for para, dpara in zip(paras, dparas)]
		self.trainfunction = theano.function(inputs=[x, y], outputs=olayer.loss, updates=updates)

	def train(self, trainx, trainy):
		return self.trainfunction(trainx, trainy)

	# 卷积神经网络的每一层，包括卷积、池化、激活映射操作


# img_shape为输入特征图，img_shape=（batch_size,特征图个数，图片宽、高）
# filter_shape为卷积操作相关参数，filter_shape=（输入特征图个数、输出特征图个数、卷积核的宽、卷积核的高）
# ，这样总共filter的个数为：输入特征图个数*输出特征图个数*卷积核的宽*卷积核的高
class LeNetConvPoolLayer:
	def __init__(self, inputx, img_shape, filter_shape, poolsize=(2, 2)):
		# 参数初始化
		assert img_shape[1] == filter_shape[1]
		a = np.sqrt(6. / (filter_shape[0] + filter_shape[1]))

		v = np.random.uniform(low=-a, high=a, size=filter_shape)

		wvalue = np.asarray(v, dtype=theano.config.floatX)
		self.w = theano.shared(value=wvalue, name='w')
		bvalue = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
		self.b = theano.shared(value=bvalue, name='b')

		covout = T.nnet.conv2d(inputx, self.w)  # 卷积操作

		covpool = T.signal.downsample.max_pool_2d(covout, poolsize)  # 池化操作

		self.out = T.tanh(covpool + self.b.dimshuffle('x', 0, 'x', 'x'))

		self.para = [self.w, self.b]

	# 读取手写字体数据


trainx = np.loadtxt('./data/TrainSamples.csv', delimiter=",")
trainy = np.loadtxt('./data/TrainLabels.csv', delimiter=",")
trainx = trainx.reshape(-1, 1, 28, 28)
batch_size = 30
m = trainx.shape[0]
ne = m / batch_size

batchx = T.tensor4(name='batchx', dtype=theano.config.floatX)  # 定义网络的输入x
batchy = T.ivector('batchy')  # 定义输出y

# 第一层卷积层
cov1_layer = LeNetConvPoolLayer(inputx=batchx, img_shape=(batch_size, 1, 28, 28), filter_shape=(20, 1, 5, 5))
cov2_layer = LeNetConvPoolLayer(inputx=cov1_layer.out, img_shape=(batch_size, 20, 12, 12),
                                filter_shape=(50, 20, 5, 5))  # 第二层卷积层
cov2out = cov2_layer.out.flatten(2)  # 从卷积层到全连接层，把二维拉成一维向量
hlayer = HiddenLayer(cov2out, 4 * 4 * 50, 500)  # 隐藏层
olayer = softmax(hlayer.out, batchy, 500, 10)  #

paras = cov1_layer.para + cov2_layer.para + hlayer.para + olayer.para  # 网络的所有参数，把它们写在同一个列表里
dparas = T.grad(olayer.loss, paras)  # 损失函数，梯度求导
updates = [(para, para - 0.1 * dpara) for para, dpara in zip(paras, dparas)]  # 梯度下降更新公式

train_function = theano.function(inputs=[batchx, batchy], outputs=olayer.loss, updates=updates)  # 定义输出变量、输出变量
test_function = theano.function(inputs=[batchx, batchy], outputs=[olayer.error, olayer.predict])

testx = np.loadtxt('./data/TestSamples1.csv', delimiter=',')
testy = np.loadtxt('./data/TestLabels1.csv', delimiter=',')
testx = testx.reshape(-1, 1, 28, 28)

train_history = []
test_history = []

for it in range(20):
	sum = 0
	for i in range(ne):
		a = trainx[i * batch_size:(i + 1) * batch_size]
		loss_train = train_function(trainx[i * batch_size:(i + 1) * batch_size],
		                            trainy[i * batch_size:(i + 1) * batch_size])
		sum = sum + loss_train
	sum = sum / ne
	print 'train_loss:', sum
	test_error, predict = test_function(testx, testy)
	print 'test_error:', test_error

	train_history = train_history + [sum]
	test_history = test_history + [test_error]
n = len(train_history)
fig1 = plt.subplot(111)
fig1.set_ylim(0.001, 0.2)
fig1.plot(np.arange(n), train_history, '-')
import numpy as np

def softmax(x):
	if x.ndim == 2:
		x = x - x.max(axis=1, keepdims=True)
		x = np.exp(x)
		x /= x.sum(axis=1, keepdims=True)
	elif x.dim == 1:
		x = x - x.max(x)
		x = np.exp(x) / np.sum(np.exp(x))
	return x

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def relu(x):
	return np.maximum(0, x)

def cross_entropy_error(y, t):
	if y.ndim == 1:
		t = t.reshape(1, t.size)
		y = y.reshape(1, y.size)
	
	#教師データがone-hotベクトル→正解ラベルのインデックスに変換
	if t.size == y.size:
		t = t.argmax(axis=1)

	batch_size = y.shape[0]
	return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

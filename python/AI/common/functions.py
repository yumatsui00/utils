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


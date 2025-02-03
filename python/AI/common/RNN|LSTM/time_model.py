import numpy as np
from .time_layers import *
from .model import BaseModel
from LSTM.LSTM import LSTM
from time

class SimpleRnnlm:
	def __init__(self, vocab_size, wordvec_size, hidden_size):
		V, D, H = vocab_size, wordvec_size, hidden_size
		rn = np.random.randn

		#重みの初期化
		embed_W = (rn(V, D) / 100).astype('f')
		rnn_Wx = (rn(D, H) / np.sqrt(D)).astype('f')
		rnn_Wh = (rn(H, H) / np.sqrt(H)).astype('f')
		rnn_b = np.zeros(H).astype('f')
		affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
		affine_b = np.zeros(V).astype('f')

		#レイヤの作成
		self.layers = [
			TimeEmbedding(embed_W),
			TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
			TimeAffine(affine_W, affine_b)
		]
		self.loss_layer = TimeSoftmaxWithLoss()
		self.rnn_layer = self.layers[1]

		self.params, self.grads = [], []
		for layer in self.layers:
			self.params += layer.params
			self.grads += layer.grads
	
	def forward(self, xs, ts):
		for layer in self.layers:
			xs = layer.forward(xs)
		loss = self.loss_layer.forward(xs, ts)
		return loss

	def backward(self, dout=1):
		dout = self.loss_layer.backward(dout)
		for layer in reversed(self.layers):
			dout = layer.backward(dout)
		return dout

	def reset_state(self):
		self.rnn_layer.reset_state()


class Rnnlm(BaseModel):
    def __init__(self, vocab_size=10000, wordvec_size=100, hidden_size=100):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        # 重みの初期化
        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        # レイヤの生成
        self.layers = [
            TimeEmbedding(embed_W),
            TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layer = self.layers[1]

        # すべての重みと勾配をリストにまとめる
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self, xs, ts):
        score = self.predict(xs)
        loss = self.loss_layer.forward(score, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        self.lstm_layer.reset_state()


#行列の積
class MatMul: #Matrix Multiply
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.X = None

    def forward(self, x):
        W,  = self.params
        out = np.dot(x, W)
        self.x = X
        return out

    def backward(self, dout):
        # 重み最適化のためのdWと、次の伝播のためのdx
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        # 要素の上書きのための...(深いコピー)
        self.grads[0][...] = dW
        return dx


class Sigmoid:
    def __init__(self):
        self.params = []
        self.out = None

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dout):
        dx = dout * (1 - self.out) * self.out
        return dx


class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        self.x = x
        return np.dot(x, W) + b

    def backward(self, dout):
        W, b = self.params
        out = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


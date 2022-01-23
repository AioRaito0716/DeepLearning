import numpy as np

#勾配の計算をする関数
#入力ベクトルx(x1,x2,..,xn)を受け取った時に、偏微分のベクトル.shape = (n,)を返す
def numerical_gradient(f,x):
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):
        delta_x = (f(x[idx]+h) - f(x[idx]-h))/(2*h)
        grad[idx] = delta_x
    return grad

#交差エントロピー　バッチ対応版、教師ラベルがone-hot表現
def cross_entropy1(y,t):
    #入力が1次元配列の場合は2次元配列に拡張する
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)
    return -np.sum(t * np.log(y+1e-7))/y.shape[0]

#勾配降下法
def gradient_descent(f, x_init, lr, step_num):
    x = x_init
    for _ in range(step_num):
        x -= lr * numerical_gradient(f, x)
    return x 

#交差エントロピー　バッチ対応版、教師ラベルが非one-hot
def cross_entropy(y,t):
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)
    return -np.sum(np.log(y[np.arrange(y.shape[0]),t]+1e-7))/y.shape[0]

#ソフトマックス関数　バッチ対応版
def softmax(x):
    x = x - np.max(x)
    x = np.exp(x)
    s = np.sum(x, axis=-1,keepdims=True)
    a = x / s
    return a

#シグモイド関数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#２層のニューラルネットクラスを定義
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size,weight_init_std):
        std = weight_init_std
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
    
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y
    
    def loss(self, x, t):
        y = predict(x)
        return cross_entropy(y, t)
    
    def accuracy(self, x, t):
        y = predict(x)
        #あとで

    def numerical_gradient(self, x, t):
        loss_W = lambda x: loss(x,t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads
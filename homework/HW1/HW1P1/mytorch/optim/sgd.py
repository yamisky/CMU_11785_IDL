import numpy as np

class SGD:

    def __init__(self, model, lr=0.1, momentum=0):
        # 初始化存储模型层、学习率和动量
        self.l = model.layers
        self.L = len(model.layers)
        self.lr = lr
        self.mu = momentum
        # 初始化权重和偏置的速度变量
        self.v_W = [np.zeros(layer.W.shape, dtype="f") for layer in self.l]
        self.v_b = [np.zeros(layer.b.shape, dtype="f") for layer in self.l]

    def step(self):
        # 对每层进行参数更新
        for i in range(self.L):
            if self.mu == 0:
                # 如果动量为0，使用标准的SGD更新
                self.l[i].W -= self.lr * self.l[i].dLdW
                self.l[i].b -= self.lr * self.l[i].dLdb
            else:
                # 如果使用动量，更新速度变量
                self.v_W[i] = self.mu * self.v_W[i] + self.l[i].dLdW
                self.v_b[i] = self.mu * self.v_b[i] + self.l[i].dLdb
                # 使用速度变量进行参数更新
                self.l[i].W -= self.lr * self.v_W[i]
                self.l[i].b -= self.lr * self.v_b[i]

import numpy as np

from mytorch.nn.linear import Linear
from mytorch.nn.activation import ReLU


class MLP0:

    def __init__(self, debug=False):
        """
        Initialize a single linear layer of shape (2,3).
        Use Relu activations for the layer.
        """

        self.layers = [Linear(2, 3), ReLU()]
        self.debug = debug

    def forward(self, A0):
        """
        Pass the input through the linear layer followed by the activation layer to get the model output.
        """

        Z0 = self.layers[0].forward(A0)  # TODO
        A1 = self.layers[1].forward(Z0)  # TODO

        if self.debug:

            self.Z0 = Z0
            self.A1 = A1

        return A1

    def backward(self, dLdA1):
        """
        Refer to the pseudo code outlined in the writeup to implement backpropogation through the model.
        """

        dLdZ0 = self.layers[1].backward(dLdA1)  # TODO
        dLdA0 = self.layers[0].backward(dLdZ0)  # TODO

        if self.debug:

            self.dLdZ0 = dLdZ0
            self.dLdA0 = dLdA0

        return dLdA0


class MLP1:

    def __init__(self, debug=False):
        """
        初始化包含两个线性层的网络。第一层形状为（2,3），第二层的形状为（3,2）
        """

        self.layers = [Linear(2, 3), ReLU(), Linear(3, 2), ReLU()]  # TODO
        self.debug = debug

    def forward(self, A0):
        """
        Pass the input through the linear layers and corresponding activation layer alternately to get the model output.
        """

        Z0 = self.layers[0].forward(A0)  # TODO
        A1 = self.layers[1].forward(Z0)  # TODO

        Z1 = self.layers[2].forward(A1)  # TODO
        A2 = self.layers[3].forward(Z1)  # TODO

        if self.debug:
            self.Z0 = Z0
            self.A1 = A1
            self.Z1 = Z1
            self.A2 = A2

        return A2

    def backward(self, dLdA2):
        """
        Refer to the pseudo code outlined in the writeup to implement backpropogation through the model.
        """

        dLdZ1 = self.layers[3].backward(dLdA2)  # TODO
        dLdA1 = self.layers[2].backward(dLdZ1)  # TODO

        dLdZ0 = self.layers[1].backward(dLdA1)  # TODO
        dLdA0 = self.layers[0].backward(dLdZ0)  # TODO

        if self.debug:

            self.dLdZ1 = dLdZ1
            self.dLdA1 = dLdA1

            self.dLdZ0 = dLdZ0
            self.dLdA0 = dLdA0

        return dLdA0


class MLP4:
    def __init__(self, debug=False):
        """
        初始化具有4个隐藏层和一个输出层的神经网络。各层的形状如下：
        第1隐藏层 (2, 4)，
        第2隐藏层 (4, 8)，
        第3隐藏层 (8, 8)，
        第4隐藏层 (8, 4)，
        输出层 (4, 2)，注意本MLP输出层后面还有个ReLU
        
        使用ReLU激活函数为所有线性层。
        
        参数:
        - debug: 是否在前向和反向传播中保留中间结果以便于调试。
        """
        # 按正确顺序初始化隐藏层和激活层的列表
        self.layers = [
            Linear(2, 4), ReLU(),
            Linear(4, 8), ReLU(),
            Linear(8, 8), ReLU(),
            Linear(8, 4), ReLU(),
            Linear(4, 2), ReLU() 
        ]
        self.debug = debug  # 调试标志

    def forward(self, A):
        """
        交替通过线性层和激活层传递输入，以获得模型的输出。
        
        参数:
        - A: 输入数据
        
        返回:
        - 最终的模型输出
        """
        if self.debug:
            self.A = [A]  # 如果调试模式开启，记录输入

        # 逐层通过网络前向传播
        for layer in self.layers:
            A = layer.forward(A)  # 应用当前层的前向传播

            if self.debug:
                self.A.append(A)  # 如果调试模式开启，记录每一层的输出

        return A  # 返回模型的输出

    def backward(self, dLdA):
        """
        参数:
        - dLdA: 损失函数对模型最终输出的梯度
        
        返回:
        - 损失函数对模型输入的梯度
        """
        if self.debug:
            self.dLdA = [dLdA]  # 如果调试模式开启，记录损失函数的梯度

        # 逐层通过网络反向传播
        for layer in reversed(self.layers):
            dLdA = layer.backward(dLdA)  # 应用当前层的反向传播

            if self.debug:
                self.dLdA.insert(0, dLdA)  # 如果调试模式开启，在列表前端插入每一层的梯度

        return dLdA  # 返回损失函数对模型输入的梯度

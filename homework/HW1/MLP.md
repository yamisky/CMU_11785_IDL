MLP

## 6.2节 - MLP（隐藏层 = 1）`[mytorch.models.MLP1]` [10]
在本节中任务是实现MLP1类的前向和反向属性函数。

MLP1的拓扑结构在图H中可视化。您需要使用该图来推断出模型特定的线性层。为了便于理解，您应该尝试给图进行标注，以显示哪些部分对应于哪些线性层和激活函数。

![image-20240112143333618](./assets/image-20240112143333618.png)

MLP1.forward() 的代码与 MLP0.forward() 非常相似，反向传播也是如此。

```Python
import numpy as np
from mytorch.nn.linear import Linear
from mytorch.nn.activation import ReLU

class MLP1:
    def __init__(self, debug=False):
        """
        初始化包含两个线性层的网络。第一层的形状为 (2,3)，第二层的形状为 (3,2)。
        两个线性层后都使用ReLU激活函数。
        类似于MLP0，将所有层按顺序存储在列表中。
        
        参数:
        - debug: 是否在前向和反向传播中保留中间结果以便于调试。
        """
        self.layers = [Linear(2, 3), ReLU(), Linear(3, 2), ReLU()]  # 定义线性层和激活层的列表
        self.debug = debug  # 调试标志

    def forward(self, A0):
        """
        交替通过线性层和相应的激活层传递输入，以获得模型的输出。
        
        参数:
        - A0: 输入数据
        
        返回:
        - A2: 模型的输出
        """
        Z0 = self.layers[0].forward(A0)  # 第一个线性层的前向传播
        A1 = self.layers[1].forward(Z0)  # 第一个激活层的前向传播
        
        Z1 = self.layers[2].forward(A1)  # 第二个线性层的前向传播
        A2 = self.layers[3].forward(Z1)  # 第二个激活层的前向传播
        
        # 如果处于调试模式，保留前向传播的中间结果
        if self.debug:
            self.Z0 = Z0
            self.A1 = A1
            self.Z1 = Z1
            self.A2 = A2

        return A2  # 返回模型的输出

    def backward(self, dLdA2):
        """
        参考文档中概述的伪代码，通过模型实现反向传播。
        
        参数:
        - dLdA2: 损失函数对模型最终输出的梯度
        
        返回:
        - dLdA0: 损失函数对模型输入的梯度
        """
        # 反向传播激活层和线性层
        dLdZ1 = self.layers[3].backward(dLdA2)  # 第二个激活层的反向传播
        dLdA1 = self.layers[2].backward(dLdZ1)  # 第二个线性层的反向传播
        
        dLdZ0 = self.layers[1].backward(dLdA1)  # 第一个激活层的反向传播
        dLdA0 = self.layers[0].backward(dLdZ0)  # 第一个线性层的反向传播
        
        # 如果处于调试模式，保留反向传播的中间结果
        if self.debug:
            self.dLdZ1 = dLdZ1
            self.dLdA1 = dLdA1
            self.dLdZ0 = dLdZ0
            self.dLdA0 = dLdA0

        return dLdA0  # 返回损失函数对模型输入的梯度
```

## 6.3 MLP (隐藏层 = 4) [mytorch.models.MLP4] [15]



### 


## 
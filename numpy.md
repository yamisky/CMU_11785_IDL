1. 返回 `[0, n)` 数组序列：
   - `np.arange(10)`
   - 输出：`[0 1 2 3 4 5 6 7 8 9]`
   - 注意：它的维度是`(10, )`即1维数组，而不是1 x 10的二维数组。在 NumPy 中，一维数组和二维数组有着本质的区别。
   
2. 返回 `[x, y)` 等差序列：
   - `np.linspace(2.0, 3.0, num=5, endpoint=False)`
   - 输出：`[2.  2.2 2.4 2.6 2.8]`
   
3. 创建未初始化的数组：
   - `np.empty((2, 3))`  # 2行3列
   - 注意：由于没初始化，每个元素值取决于内存当前位置状态。
   
4. 创建全 0 或全 1 数组：
   - `np.zeros((2, 3))`, `np.ones((4, 2))`
   
5. 创建自定义填充值的数组：
   - `np.full((2, 3), 10)`
   
6. 创建跟 `arr` 相同维度，值全 0 或全 1 数组：
   - `np.zeros_like(arr)`, `np.ones_like(arr)`
   
7. 创建跟 `arr` 相同维度，自定义填充值的数组：
   1. `np.full_like(arr, 0.1, dtype=np.double)`
   
8. list转array：
   1. `np.array(python_list)`
   
9. array转list:
	1. `arr.tolist()`
	
10. 创建随机整数数组：
    1. `np.random.randint(0, 10, size=(2,3))`
    2. 备注：[0, 10)区间，2行3列数组
    3. 如要固定随机结果，添加`np.random.seed(0)`
    
11. 创建给定形状的数组，并用[0,1)上均匀分布的随机值填充:
	1. `np.random.rand(2,3)`
	
12. 创建元素值为标准正态分布的数组：
    1. `np.random.randn(2,3)`
    2. 标准正态分布是以 0 为均值（mean）和 1 为标准差（standard deviation）的正态分布。
    3. 大约 68% 的值会落在均值的一个标准差范围内（-1 到 1）。
    4. 大约 95% 的值会落在均值的两个标准差范围内（-2 到 2）。
    5. 大约 99.7% 的值会落在均值的三个标准差范围内（-3 到 3）。
    
13. 创建均值为`mean`，标准差为`sigma`的正态分布数组：
	1. `mean + sigma * np.random.randn(2, 3)`
	
14. 切片：`n[0::2, 1:4, 1::2]`
    1. `0::2`：这是对第一个维度（通常是行）的切片。它从索引 0 开始，以步长为 2 选择元素。这意味着它会选择索引为 0, 2, 4, ... 的元素。
    2. `1:4`：这是对第二个维度的切片。它从索引 1 开始，到索引 4 结束，但不包括索引 4。因此，它选择索引为 1, 2, 3 的元素。
    3. `1::2`：这是对第三个维度的切片。它从索引 1 开始，以步长为 2 选择元素。这意味着它会选择索引为 1, 3, 5, ... 的元素。
    
15. 深拷贝
    1. `np.copy(xxx)`
    2. 关于深拷贝：`n_copy = np.copy(n)` 用于创建数组的一个内存独立副本，而 `n_copy = n` 只是为相同的数组创建了一个新的引用。
    
16. 获取维度、改变形状
    1. `arr.shape`
    2. `arr.reshape(m, n)`
    
17. 转置数组：
	1. `np.transpose(arr)`
	
18. 自定义转置维度：
    1. `np.transpose(arr, axes=(0, 2, 1))` #第2，3维度调换顺序
    2. 假设原维度是`(3, 4, 5)`，调整后变为`(3, 5, 4)`
    
19. 将多维数组拉平：
    1. `arr.flatten()` #默认按`行`顺序展平
        1. 备注：`arr.flatten('F')` #按`列`顺序展平
    2. `arr.reshape(-1)`
    3. 两者区别：
        1. `.reshape(-1)` 在不需要复制数据时更为高效，但可能影响原始数组；而 `.flatten()` 总是创建新的数组副本，保证了原始数据的不变性。
    
20. 移除数组中大小为1的维度：
    1. `np.squeeze(arr)`
    2. 假设你有一个形状为 `(1, 3, 1, 5)` 的数组。使用 `np.squeeze` 后，这些大小为 1 的维度将被移除，因此结果数组的形状将变为 `(3, 5)`。
    3. 如果指定了轴（axis）参数，`np.squeeze(arr, axis=xxx)` 将只尝试移除指定的轴。如果指定轴的大小不为 1，将抛出一个错误。
    
21. 在指定位置增加一个轴（axis）:
    1. `np.expand_dims(y, axis=1)` #(4, 5) -> (4, 1, 5)
    2. `np.expand_dims(y, axis=(0, 2))` #(4, 5) -> (1, 4, 1, 5)
    
22. 拼接数组、堆叠数组：
	1. 拼接：`np.concatenate((a1, a2, ...), axis=0)`
		1. `a1, a2, ...` 是要连接的数组序列。**它们必须具有相同的形状，除了在指定的轴上**。
		2. `axis` 参数指定了连接的轴。默认是 0，意味着沿着第一个轴（通常是行）进行连接。如果 `axis=1`，则沿着第二个轴（通常是列）进行连接。
		3. ```python
           a = np.array([1, 2, 3])
           b = np.array([4, 5, 6])
           # 使用 np.concatenate 连接这两个数组
           result = np.concatenate((a, b))
           #结果
           [1 2 3 4 5 6]
          ```
	2. 堆叠：`np.stack((a, b), axis = )`
		1. 它可以将多个具有相同形状的数组堆叠成一个新的数组，新数组的维度将比原数组多一个。
		2. ```python
           a = np.array([1, 2, 3])
           b = np.array([4, 5, 6])
           result = np.stack((a, b))
           #结果
           [[1 2 3]
            [4 5 6]]
           
           result = np.stack((a, b), axis=1)
           #结果
           [[1 4]
            [2 5]
            [3 6]]
           ```
	3. 总结
        1. 使用 `np.concatenate` 时，你沿着现有的轴连接数组，不会增加数组的维度。
        2. 使用 `np.stack` 时，你在一个新轴上堆叠数组，会增加一个额外的维度。
    
23. numpy的广播机制
    1. 广播的基本思想是在某些情况下，使较小的数组“广播”到较大数组的大小，以便它们具有兼容的形状。这使得算术运算在形状不完全匹配的数组之间变得可能。
    2. ```python
       import numpy as np
              
       # 例子 1: 将标量与数组相加
       a = np.array([1, 2, 3])
       b = 2
       print(a + b)  # 输出: [3 4 5]
              
       # 例子 2: 将两个数组相加，其中一个数组在一个维度上大小为 1
       a = np.array([[1, 2, 3], 
                     [4, 5, 6]])
       b = np.array([1, 0, 1])
       print(a + b)  
       # 输出: 
       [[2 2 4], 
        [5 5 7]]
              
       # 例子 3: 将两个数组相乘，其中一个数组在两个维度上大小为 1
       a = np.array([[1, 2, 3], [4, 5, 6]])
       b = np.array([[1], [2]])
       print(a * b)  
       # 输出: [[1 2 3], [8 10 12]]
              
       在每个例子中，较小的数组会在必要的维度上进行扩展（复制其值），以匹配较大数组的形状，然后执行元素级的运算。这样做的优点是可以避免不必要的数据复制，从而提高运算效率。
       ```
    
24. 相关数学操作：
    1. np.abs(arr)
    2. np.sqrt(arr)
    3. np.max(arr)
    4. np.min(arr)
    5. np.sum(arr)
    6. np.argmax(arr, axis= ) #找出数组中元素最大值的索引
    7. np.argmin(arr, axis= )
    8. np.mean(arr)
    9. np.std(arr) #计算数组的标准差（标准差是数据分布离散程度的一种度量，表示数据集中的数值相对于平均值的偏离程度。）
    10. np.linalg.norm(arr, ord= ) #计算范数。范数是一个表示“大小”或“长度”的数学概念。
		1. `ord` 参数指定范数的类型。例如，`ord=2` 表示欧几里得范数（向量的情况下是常规的长度计算），而 `ord=1` 表示向量的各元素绝对值之和（即 L1 范数）
	
25. numpy的比较操作：
    1. 元素级别上比较数组中的数据。这些比较操作会返回布尔型数组，表示每个元素是否满足比较条件。
    2. ```python
       import numpy as np
             
       a = np.array([1, 2, 3, 4, 5])
       b = np.array([3, 3, 3, 3, 3])
             
       # 等于
       print(a == b)  # 输出: [False False  True False False]
       ```
    
26. 矩阵乘法
	1. `np.matmul(a, b)`
		1. **对于两个一维数组**：它计算的是这两个数组的内积。
		2. **对于两个二维数组**：它计算的是标准的矩阵乘法。
		3. **对于高维数组**：它将这些数组视为堆叠的矩阵，并执行适当的乘法运算。例如，如果 `a` 和 `b` 是维度为 `(n, m, k)` 和 `(n, k, l)` 的数组，则 `np.matmul(a, b)` 的结果将是一个 `(n, m, l)` 维度的数组。
		4. **一维和二维的组合**：如果一个数组是一维的，而另一个数组是二维的，`np.matmul` 将执行适当的矩阵-向量或向量-矩阵乘法。
	2. `arr1 @ arr2`
		1. 在 NumPy 中，`array1 @ array2` 是一种更简洁的语法，用于计算两个数组的矩阵乘法。这个 `@` 运算符在 Python 3.5 及以后的版本中引入，作为专门用于矩阵乘法的运算符。
	
27. 点积
	1. `np.dot(a, b)`
	2. 这个函数在不同情况下有不同的行为，具体取决于输入数组的维度。
		1. **两个一维数组**：计算它们的内积。
		2. **二维数组**：执行矩阵乘法。
		3. **更高维度的数组**：如果任一输入数组的维度大于 2，`np.dot` 将其视为堆叠在一起的矩阵，执行相应的乘法运算。
		4. **一维和二维数组的组合**：如果一个数组是一维的，而另一个数组是二维的，`np.dot` 将执行矩阵-向量乘法或向量-矩阵乘法。
		5. 值得注意的是，对于二维数组，`np.dot` 和 `np.matmul`（或 `@` 运算符）效果相同，都执行标准的矩阵乘法。然而，对于高维数组，`np.dot` 和 `np.matmul` 的行为可能会不同。
	
28. 张量点积
    1. `np.tensordot` ：是 NumPy 中的一个函数，用于计算两个数组的张量点积（tensor dot product）。与普通的点积或矩阵乘法不同，张量点积可以在任意数量的轴上进行合并，这使得它在处理高维数据时非常强大和灵活。
    
    2. 作用：理解张点函数将帮助你为你的作业，尤其是卷积神经网络作业，编写简洁的代码。

    3. ```python
       a = np.arange(60.).reshape(3,4,5)
       b = np.arange(24.).reshape(4,3,2)
       print('A \'s dimension ', a.shape, '\n')
       print('B \'s dimension ', b.shape, '\n')
       
       # compute tensor dot product along specified axes.
       c = np.tensordot(a,b, axes=([1,0],[0,1]))
       print("A⨂B =\n", c, ' with dimension', c.shape, '\n')
       
       # this equals to
       d = np.zeros((5,2))
       for i in range(5):
         for j in range(2):
           for k in range(3):
             for n in range(4):
               d[i,j] += a[k,n,i] * b[n,k,j]
       print("tensor dot is equal to sum over certain dimensions.\n")
       print(c==d)
       
       #结果
       A 's dimension  (3, 4, 5) 
       B 's dimension  (4, 3, 2) 
       
       A⨂B =
        [[4400. 4730.]
        [4532. 4874.]
        [4664. 5018.]
        [4796. 5162.]
        [4928. 5306.]]  with dimension (5, 2) 
       
       tensor dot is equal to sum over certain dimensions.
       
       [[ True  True]
        [ True  True]
        [ True  True]
        [ True  True]
        [ True  True]]
       ```






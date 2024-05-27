# BP神经网络(Backpropagation)

BP神经网络由以下两个组成：

1. 2层线性层：分别用于特征提取（输入），分类
    1. 分类：输出，一般叫做分类器，会接在模型的最后一层给特征进行分类（除了全卷积模型）
2. 激活函数（在两层线性层中间）

## MNIST 数据集

- training data: 60000
- Testing data: 10000
- 黑白图，28*28，数字0-9
- 数字越大位置的颜色越白，0 代表黑色

![Untitled](BP%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(Backpropagation)%20f3d9d1197866432dbc7772f089b16fd2/Untitled.png)

- 数据集解析：给每一个28*28 变成一个1*784的形状（拉平），这个被认定为图片的特征， 最后就是60000*728
    - 这种做法实现了降维

## 分类任务实现逻辑

![Untitled](BP%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(Backpropagation)%20f3d9d1197866432dbc7772f089b16fd2/Untitled%201.png)

- 图片分类和文本分类的实现逻辑是一样的
- 我们要做的事情是给28*28的图转换成一个1*1的数字
- 在第二步（线性回归）， 我们要×一个W矩阵（784*1），根据结果判断分类
    - 缺点是矩阵乘法的结果可能是实数范围内的任意一个数字
    
    <aside>
    😾 这点应该是矩阵乘法必须要注意的一个问题，如何给这个任意数字转换成我们所需要的？在这里是0-9
    
    </aside>
    
- 所以更新第二步， 变成1*784的矩阵*W（784*10）最后的结果是一个1*10 的矩阵，取一个最大的所在下标

![Untitled](BP%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(Backpropagation)%20f3d9d1197866432dbc7772f089b16fd2/Untitled%202.png)

- BP会再×一个矩阵
- 那么loss怎么算？ 1*10的矩阵和1*1的矩阵，如果只算最大值的下标的话，那是没办法求导的，所以要用one-hot

## One-Hot

- 在我们的例子中，给1*1的label，改成1*10的矩阵，在对应下标标记1，其余位置0

![Untitled](BP%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(Backpropagation)%20f3d9d1197866432dbc7772f089b16fd2/Untitled%203.png)

- pre数组里，2的位置无限向1学习，其他位置无限像0学习，但是特别的大的数字向0，1converge是很难的，所以我们要改造一下预测值
- 给pre（我们的预测）加入softmax激活函数

![Untitled](BP%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(Backpropagation)%20f3d9d1197866432dbc7772f089b16fd2/Untitled%204.png)

- 非负数，和为1，里面的值有概率的意思
- 用softmax改造之后的值去学习

## Softmax

- 用于多分类

![Untitled](BP%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(Backpropagation)%20f3d9d1197866432dbc7772f089b16fd2/Untitled%205.png)

- 对于非负数，归一化很简单，只需要计算出每一项的占比：A/(A+B+C)
- 负数：所以用exp(-X) 如果x是负数， 那么占比就变成exp(A)/(exp(A)+exp(B)+exp(-x))

![Untitled](BP%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(Backpropagation)%20f3d9d1197866432dbc7772f089b16fd2/Untitled%206.png)

- exp 会放大个体差距，缩小小的，所以整体的差距会变大
- 选择用e是因为方便求导

![Untitled](BP%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(Backpropagation)%20f3d9d1197866432dbc7772f089b16fd2/Untitled%207.png)

x是输入层，w2是分类层

w1是一个线性层，也就是矩阵相乘，得到一个hidden（隐层的特征）→ 是一个特征提取，

- 给1*784个特征 提取成h-dimension个特征，所以不可以是1，有点太小
- w1的大小784*？， ？是h的大小，h不宜过大或者过小
- 理论上特征越多越好，但是模型承载太多了
- 一般128，256，1024 等等，可以自己选
- 得到的第一个h需要加一个激活函数

第二层w2的大小就应该为x * 10 ， 因为我们想要的是10个特征，x的大小根据选择的h来决定，假设我们选择256，那么h就是1*258，w2的大小就是256*10，因为我们最后想得到10个特征，p的大小是1*10

- 第一层的输出层 h 变层了第2层的输入，有的可能有好几千层，但是前面都是特征提取，只有最后的是归类
- 得到的p进行softmax，然后计算loss

考虑batch size

![Untitled](BP%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(Backpropagation)%20f3d9d1197866432dbc7772f089b16fd2/Untitled%208.png)

假设拿了3个图片，那么x就变成3*784了，h也变，这个变化是一个动态的变化过程

<aside>
👻 sigmoid, softmax都可以给数字变成0-1之间，softmax多了一个去负

</aside>

## BP Backward

![Untitled](BP%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(Backpropagation)%20f3d9d1197866432dbc7772f089b16fd2/Untitled%209.png)

- 深度学习实际上是一个不断更新参数的过程，我们这里要更新w1和w2→ backward先从后面算，先算的是w2 导数从pre算起
- h = x@w1, sig_h = sigmoid(h), p = sig_h @ w2, pre = softmax(p), loss = label * log(pre) + (1-label)*log(1-pre)
    
    [详细这里](https://www.notion.so/58044e8fee8442ef8cfa34797657a488?pvs=21)
    
- 先求loss对pre的导数，然后p
- $G_2 = \partial loss/\partial p = \partial loss/\partial pre * \partial pre/\partial p = pre - label$ 3*10
- $\partial loss/\partial w_2 = sig_h.T @ G_2$ = ∆w2 — 矩阵导数 →(1) 256*10
- ∆sig_h =  G2 @ W2.T
- update: delta_w2 = w2- lr * ∆w2, w2知道了之后开始w1，backward从sig_h开始
- $\partial loss/\partial h = \partial loss/\partial sig_h * \partial sig_h/\partial h$ = ∆sig_h *(sig_h**(1-sig_h)) = ∆h = G.    3*256
    - ∆sig_h =$\partial loss/\partial sig_h$   = G2 @ W2.T    3*256
    - ∆sigmoid =  sigmoid*(1-sigmoid)
- ∆w1 = x1.T @ ∆h.     784*256

## Coding part

```python
import numpy as np
1. bytearray() # 用于 表示一个字节序列
2. data[8:] # 从列表或类似的数据结构中取出第8到最后的所有元素
3. bytearray(data[8:]) #将这些元素标称一个字节数组
4. dtype=np.int32 #表示每一个元素都应该被解释成一个32位整数
5. np.asanyarray #用于将输入转换成一个‘ndarray’
```

<aside>
📌 debug模式要记得用，用于看数据，matrix的大小等等

</aside>

![Untitled](BP%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(Backpropagation)%20f3d9d1197866432dbc7772f089b16fd2/Untitled%2010.png)

在图像处理领域中，图像数据除以 255 是一种常见的做法，这主要是为了进行所谓的归一化处理。这里的归一化指的是将数据缩放到一个固定的范围（通常是 0 到 1）

```python
train_datas = load_images("data\\train-images.idx3-ubyte")/255
```

步骤：

1. 读取对应的数据和label
2. 加载数据
3. one hot编码
4. epoch训练
5. forward
6. backward
7. update 
8. calculate precision 

if batch is too big, the data will overflow (nan) - 初始梯度会变大，那么后面的数据都会变得非常大，如果学习率不够小就会炸了

- G2 = G2/batch_size, loss / batch_size 可以有效解决（pytorch就是这样计算的)

<aside>
📌 同一个epoch的情况下:

</aside>

- batch越大梯度下降越慢，运行速度越快
    - 梯度变化慢：因为每次迭代包含了更多的数据点，所以每一步的更新更加平滑，这可能导致梯度的整体变化较慢。这种情况下，虽然每一次迭代处理的数据更多，但是整体需要更多的迭代次数来达到同样的误差率，尤其是在靠近最优解的区域。 → 学习速度慢
    - 运行速度快：利用了现代计算框架（如GPU）的并行处理能力，能够更快地完成每一次迭代的计算，尽管可能需要更多的迭代次数。
- batch越小梯度下降越快，运行速度越慢
    - 梯度变化快：每次迭代只使用少量数据计算梯度，使得参数更新更频繁且步长变化大。这样的梯度估计虽然噪声较多，但有助于模型跳出局部最优解，可能在某些情况下能更快找到全局最优解。
    - 运行速度慢：由于每次迭代计算量小，不能充分利用高度并行的硬件资源，使得总体运行时间增加，特别是在数据量很大的情况下
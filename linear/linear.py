import numpy as np
import struct
import matplotlib.pyplot as plt

from tifffile.tifffile_geodb import Linear


# load data
def load_labels(file):
    with open(file, "rb") as f:
        data = f.read()
    return np.asanyarray(bytearray(data[8:]), dtype=np.int32)


def load_images(file):
    with open(file, "rb") as f:
        data = f.read()
    magic_number, num_items, rows, cols = struct.unpack(">IIII", data[:16])
    return np.asanyarray(bytearray(data[16:]), dtype=np.uint8).reshape(num_items, -1)

def get_datas():
    train_datas = load_images("data\\train-images.idx3-ubyte") / 255
    # make one hot
    train_labels = make_one_hot(load_labels("data\\train-labels.idx1-ubyte"))
    test_datas = load_images("data\\t10k-images.idx3-ubyte") / 255
    test_labels = load_labels("data\\t10k-labels.idx1-ubyte")

    return train_datas, train_labels, test_datas, test_labels

def make_one_hot(labels, class_num=10):
    result = np.zeros((len(labels), class_num))
    for index, label in enumerate(labels):
        result[index][label] = 1
    return result


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    ex = np.exp(x)
    # softmax. sum each row
    sum_ex = np.sum(ex, axis=1, keepdims=True)
    return ex / sum_ex

# linear layer
class Linear:
    def __init__(self, in_num,out_num):
        self.weight = np.random.normal(0,1,size=(in_num, out_num))

    def forward(self,x):
        self.x = x
        return x @ self.weight

    def backward(self,G):
        delta_weight = self.x.T @ G
        delta_x = G @ self.weight.T
        self.weight -= learning_rate * delta_weight # 优化器内容，用的梯度下降优化器 SGD
        return delta_x
    def __call__(self,x):
        return self.forward(x)
class Sigmoid:
    def forward(self, x):
        self.r = sigmoid(x)
        return self.r

    def backward(self,G):
        return G*self.r * (1 - self.r)
    def __call__(self,x):
        return self.forward(x)

class Softmax:
    def forward(self, x):
        self.r = softmax(x)
        return self.r
    def backward(self,G): # G = label
        return (self.r-G)/self.r.shape[0]
    def __call__(self,x):
        return self.forward(x)

# 封装 layers
class Mymodel:
    def __init__(self,layers):
        self.layers = layers

    def forward(self,x,label=None):
        for layer in self.layers:
            x = layer(x)
        self.x = x
        if label is not None:
            self.label = label
            loss = -np.sum(label * np.log(x)) /x.shape[0]
            return loss
    def backward(self):
        G = self.label
        for layer in self.layers[::-1]:
            G = layer.backward(G)

        return G
    #变参数
    def __call__(self, *args,):
        return self.forward(*args)



if __name__ == "__main__":

    train_datas, train_labels, test_datas, test_labels = get_datas()

    epoch = 10
    batch_size = 100  # our example is 3. calculation is very slow
    learning_rate = 0.1
    hidden_num = 256

    #w1 = np.random.normal(0, 1, size=(784, hidden_num))  # randomized w1 with correct shape
    #w2 = np.random.normal(0, 1, size=(hidden_num, 10))
    # linear1 = Linear(784,hidden_num)
    #one activation is needed between 2 linear layers
    # sigmoid1 = Sigmoid()
    # linear2 = Linear(hidden_num,10)
    # softmax1 = Softmax()
    # layers = [
    #     Linear(784,hidden_num),
    #     Sigmoid(),
    #     Linear(hidden_num,10),
    #     Softmax()
    # ] # 在这部分的代码可以直接加layer
    model = Mymodel(
        [
            Linear(784, hidden_num),
            Sigmoid(),
            Linear(hidden_num, 10),
            Softmax()
        ]
    )
    batch_times = int(np.ceil(len(train_datas) / batch_size))
    for e in range(epoch):
        for batch_index in range(batch_times):
            # --------------get data----------------
            x = train_datas[batch_index * batch_size:(batch_index + 1) * batch_size]
            batch_label = train_labels[batch_index * batch_size:(batch_index + 1) * batch_size]

            # -------------- forward--------------
            # h = linear1.forward(batch_x)# A @B  = C, G = loss --> partial C, delta_A = B @ G.T, delta_B = A.T @ G
            # sig_h = sigmoid1.forward(h)
            # p = linear2.forward(sig_h)
            # pre = softmax1.forward(p)
            '''
            change to x
            '''
            # x = linear1.forward(batch_x)
            # x = sigmoid1.forward(x)
            # x = linear2.forward(x)
            # x = softmax1.forward(x)
            '''
            use layers
            '''
            # for layer in layers:
            #     x = layer.forward(x)
            '''
            use Mymodel，同理给下面换成backward
            '''
            loss = model(x,batch_label)
            if batch_index % 100 ==0:
                print(f"loss = {loss:.3f}")



            # --------------backward--------------
        # partial loss/partial pre
            # softmax backward and loss
            # G2 = (pre - batch_label) / batch_size
            # delta_sig_h = linear2.backward(G2)
            # delta_h = sigmoid1.backward(delta_sig_h)
            # linear1.backward(delta_h)

            # G = (x - batch_label)/batch_size

            # G = softmax1.backward(G)
            # G = linear2.backward(G)
            # G = linear1.backward(G)
            model.backward()
            # -------------- update --------------
            # linear1.weight = linear1.weight - learning_rate * delta_w1
            # linear2.weight = linear2.weight - learning_rate * delta_w2

        # -------------- induction and precision--------------
        x = test_datas
        model(x)
        # 最大值下标所在位置 -> 真正的预测值
        pre = np.argmax(model.x, axis=1)
        acc = np.sum(pre == test_labels) / len(test_labels)
        print(f"{'*'*20}\nacc = {acc:.3f}")






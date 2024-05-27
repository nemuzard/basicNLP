import numpy as np
import struct
import matplotlib.pyplot as plt

#load data
def load_labels(file):
    with open(file,"rb") as f:
        data = f.read()
    return np.asanyarray(bytearray(data[8:]),dtype=np.int32)

def load_images(file):
    with open(file,"rb") as f:
        data = f.read()
    magic_number,num_items,rows,cols = struct.unpack(">IIII",data[:16])
    return np.asanyarray(bytearray(data[16:]),dtype=np.uint8).reshape(num_items,-1)

def make_one_hot(labels,class_num=10):
    result = np.zeros((len(labels),class_num))
    for index,label in enumerate(labels):
        result[index][label] = 1
    return result
def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    ex = np.exp(x)
    # softmax. sum each row
    sum_ex = np.sum(ex, axis=1,keepdims=True)
    return ex/sum_ex


if __name__ == "__main__":
    # 0-255: make between 0 and 1
    train_datas = load_images("data\\train-images.idx3-ubyte")/255
    train_labels = make_one_hot(load_labels("data\\train-labels.idx1-ubyte"))

    test_datas = load_images("data\\t10k-images.idx3-ubyte")/255
    test_labels = load_labels("data\\t10k-labels.idx1-ubyte")
    epoch = 10
    batch_size = 100 # our example is 3. calculation is very slow
    learning_rate = 0.1

    hidden_num = 256
    w1 = np.random.normal(0,1,size=(784,hidden_num)) # randomized w1 with correct shape
    w2 = np.random.normal(0,1,size=(hidden_num,10))

    batch_times = int(np.ceil(len(train_datas)/batch_size))
    for e in range(epoch):
        for batch_index in range(batch_times):
            #--------------get data----------------
            batch_x = train_datas[batch_index*batch_size:(batch_index+1)*batch_size]
            batch_label = train_labels[batch_index*batch_size:(batch_index+1)*batch_size]


            #-------------- forward--------------
            h = batch_x @ w1
            sig_h = sigmoid(h)
            p = sig_h @ w2
            pre = softmax(p)
            # 多分类
            loss = -np.sum(batch_label * np.log(pre))/batch_size

            #--------------backward--------------
            G2 = (pre-batch_label)/batch_size
            delta_w2 = sig_h.T @ G2
            delta_sig_h = G2 @ w2.T
            delta_h = delta_sig_h * sig_h *(1-sig_h)
            delta_w1 = batch_x.T @ delta_h
            
            #-------------- update gradient --------------
            w1 = w1 - learning_rate * delta_w1
            w2 = w2 - learning_rate * delta_w2

        #-------------- precision--------------
        h = test_datas@ w1
        sig_h = sigmoid(h)
        p = sig_h @ w2
        pre = softmax(p)
        #最大值下标所在位置 -> 真正的预测值
        pre = np.argmax(pre, axis=1)
        acc = np.sum(pre==test_labels)/len(test_labels)
        print(acc)






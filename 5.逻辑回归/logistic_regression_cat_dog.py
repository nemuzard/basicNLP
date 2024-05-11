import numpy as np

def sigmoid(x):
    # For positive values of x
    pos_mask = (x >= 0)
    # For negative values of x, shift the computation to avoid overflow
    neg_mask = (x < 0)
    z = np.zeros_like(x, dtype=float)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x, dtype=float)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)
# ---------------毛发长，腿长
dogs = np.array([[8.9,12],[9,11],[10,13],[9.9,11.2],[12.2,10.1],[9.8,13],[8.8,11.2]],dtype = np.float32)   # 0
cats = np.array([[3,4],[5,6],[3.5,5.5],[4.5,5.1],[3.4,4.1],[4.1,5.2],[4.4,4.4]],dtype = np.float32)        # 1

# --  猫和狗的数据大概这样子
# --                        组合起来变成 14*2， 上面7*2是狗，下面7*2是猫
# --
# --
# --
# --                   以label是14*1的，上面都是狗 所以7个0，下面都是猫，所以是7个1
# 这个代码首先创建[0,0,0,0,0,0,0, 1,1,1,1,1,1,1],数组的数据类型被指定为 float32
#-1 在 reshape 方法中用来自动计算该维度的大小，使总元素数量保持不变（14）。在这个例子中，因为原始数组已经有 14 个元素，所以 reshape(-1, 1) 会创建一个新的二维数组，其中有 14 行，每行一个元素。
# 1 表示新的二维数组的每行有一个元素。 reshape这里有几个参数就代表着几个维度
labels = np.array([0]*7+[1]*7,dtype = np.float32).reshape(-1,1)

#拼接dog和cat
X = np.vstack((dogs,cats))
#数值太少了可能不准
k = np.random.normal(0,1,size=(2,1))
epoch = 1000
lr = 0.05
b = 0
for e in range(epoch):
    p = X @ k +b #计算结果

    #加上激活函数 sigmoid, 预测
    pre = sigmoid(p)
    # pre 和 labels 都是矩阵， 所以最后的loss要算sum，因为要给所有的都加上,再算平均， 加上负号是为了loss变成正数
    loss = -np.mean(np.sum(labels * np.log(pre) + (1-labels)*np.log(1-pre)))
    G = pre - labels # G 是梯度
    delta_k = X.T @ G
    # delta_b = G, 但是G是数字。 B只是一个偏置项，loss对p的导数和loss 对b的导数是一样的
    delta_b = np.sum(G)
    k = k - lr * delta_k
    b = b - lr * delta_b

    print(loss)

f1 = float(input("毛发长："))
f2 = float(input("脚长："))
# 1 row , 2 col
test_x = np.array([f1,f2]).reshape(1,2)
p = test_x @ k + b
p = sigmoid(p)
if p>0.5:
    print("cat")
else:
    print("dog")

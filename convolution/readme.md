
# 卷积（Convolution）

- nlp也可以用，比如textCNN，但是图片提取特征用的比较多，是deep learning的基础
- 图片 3 = 通道， 中间的10是高，最后为宽
    
<img width="248" alt="Screenshot_2024-05-19_at_02 46 18" src="https://github.com/nemuzard/basicNLP/assets/44145324/28941afc-bb46-4503-af42-751827069a64">

    
- 卷积和：有拉普拉斯卷积和，高斯卷积和。。。 每个数字都不一样

<img width="137" alt="Screenshot_2024-05-19_at_02 47 56" src="https://github.com/nemuzard/basicNLP/assets/44145324/16df6d9a-f974-4b7f-af1f-c6a27e5486ef">


- 数字是随便写的，通道数，高，宽
- 卷积和的通道数量 = 图片输入数量
- 卷积和不宜过大或者过小，超级参数（一般不会超过图片大小）

## 计算方式

- 对位置相乘再相加（所有通道都算，一个和）
    <img width="245" alt="Screenshot_2024-05-19_at_02 51 46" src="https://github.com/nemuzard/basicNLP/assets/44145324/bc636ccd-2636-42ce-903f-7527028624f1">


    - 小的卷积和会移动，直到覆盖了所有，每次移动都要进行上面的计算。每一次移动的结果为新的图片的数字
- 在我们的例子里最后会得到一个1*8*8的图片 → 特征图，所以卷积的过程叫做特征提取
- 下面的情况我们会得到4*1*8*8的特征图，就是4个1*8*8
    
<img width="522" alt="Screenshot_2024-05-19_at_02 58 13" src="https://github.com/nemuzard/basicNLP/assets/44145324/701af707-0dbe-46f1-a428-e3b1724cc802">

    
    <aside>
    💡 numpy的广播机制可以很好的利用
    
    </aside>
    
- 多加了一个卷积和，那么就是4（batch）*2*8*8的特征图（4个2*8*8）
    <img width="108" alt="Screenshot_2024-05-19_at_03 00 35" src="https://github.com/nemuzard/basicNLP/assets/44145324/24d8eeb4-f8f9-4a7f-ac94-b540ab552ca6">


    
    <aside>
    💡 第一个2叫做输出channel：想给一张图片输出2张特征图
    
    </aside>
    
- 4（batch ）*3（in channel）*10*10（图片大小）- 4张图片同时运算
- 2（out channel）*3（in channel，原始图片的通道数量）**3*3（kernel size）*
    - kernel size会决定特征图的大小，
- 特征图4（batch）*2（out channel-每张图片有几个特征图）*8*8
- 每组卷积和，输入通道要保持一致


## 加速img2col

给零散的相乘相加方式变成矩阵运算，加速的方式就会有很多
![Untitled](https://github.com/nemuzard/basicNLP/assets/44145324/d3f9e88b-ec15-4274-8cb7-89c74b3bd2c3)



- C 等于特征图的矩阵
- 给特征图A从3*3*3 reshape成1*27
- B：第一个3*3*3的也reshape成1*27的，然后右、下移动 变成新的一列，b最终变成27*64的矩阵
    - 原本的像素点3*10*10 = 300，新的是27*64>300 因为有重复，大约是n*n倍（假设卷积和是n*n），过程叫做img to col，不可以用reshape代替

![Untitled 1](https://github.com/nemuzard/basicNLP/assets/44145324/5dd78a00-a10e-41a0-85ac-33a6418071e4)


- 最后就结果是1*64，最后要reshape一下变成特征图（8*8）

### 多维img2col
![Untitled 2](https://github.com/nemuzard/basicNLP/assets/44145324/3095f135-a6b8-4659-922b-051643f9252b)



最后的特征图矩阵4*1*64，变成4*8*8（reshape）

如果卷积和变成4*3*3*3，那么就变成4*27，给拼在一起了

<aside>
🔥 输出的图像尺寸 = （输入尺寸+2*padding - kernel_size)/stride+1

</aside>

# Optimizer

when given a gradient, how do you update weights?

1. SGD:  Stochastic Gradient Descent.
    1. use current weight `self.weight` which is the weight matrix currently being used in the neural network model, 
    2.  gradient `delta_weight`  which represents the gradient of the loss function with respect to weights and indicates the direction in which the weights should be adjusted to minimize the loss
    3. `self.weight = self.weight -lr*delta_weight` 
2. PyTorch:  `import torch.optim as opt` 

![Untitled](https://github.com/nemuzard/basicNLP/assets/44145324/9bf51aad-2623-4e1b-a756-5a78d30d2bec)


if let mu = 0, then  v = 

![Untitled 1](https://github.com/nemuzard/basicNLP/assets/44145324/156a720e-bc96-4679-ba0e-904253b9d5fd)


and that is `lr*delta_weight` 

- mu is a number between 0 and 1(percentage) , if mu = 1, it may cause overflow, accuracy will be super low
- momentum sgd:  $\mu vt$ , vt: updates

## Comparison

SGD:
![Untitled 2](https://github.com/nemuzard/basicNLP/assets/44145324/aaef32ef-7a50-46b1-93e4-1dfdb5ba48e9)



MSGD
![Untitled 3](https://github.com/nemuzard/basicNLP/assets/44145324/8face501-fc2e-42ba-ac9f-2f845a0a567e)


## Sigmoid Overflow/underflow

- Happen when dealing with very large or very small inputs in the sigmoid function
- `1/(1+np.exp(-x))`
- if x is big and positive, then `np.exp(-x)`  becomes very close to 0. The computational limitations mean that for sufficiently large x, `np.exp(-x)` may be computed as exactly 0
- if x is a ‘large’ negative input, the exponential function grows quickly, and computational systems have a maximum number that can be represented. Beyond this max number, overflow happens.
![Untitled 4](https://github.com/nemuzard/basicNLP/assets/44145324/ed1b5564-d26f-4477-beb7-64c798c89241)


$$
e^x
$$

- use  `x = np.clip(x,lower_range, upper_range)` , for example: `x = np.clip(x,-100,1000000)`

## Softmax Overflow/underflow

- happens when the exponents of large numbers are computed, leading to inf or NaN results
- still can be solved by using `x = np.clip(x,lower_range,upper_range)` ,for example, `lower_range = -1000; upper_range = 100`

But this is not the best solution; it may lead to other numerical inaccuracies and does not address the root cause of the problem.

- the prob remains the same while x is higher than 10 or lower than -1000 since the softmax is supposed to convert to prob

Softmax: Note: we need a prob returned when the input is a negative number 

![Untitled 5](https://github.com/nemuzard/basicNLP/assets/44145324/ecea035c-8a8a-403d-bf52-b925050704d9)


<aside>
❓ Why can we use the clip to address the problem of overflow in Softmax but not in Sigmoid?

</aside>

- Clipping values in softmax can distort the resulting probability. Because Softmax is sensitive to the difference between the maximum value and other values. (large input, large output, sum = 1)
- While sigmoid is more stable since its output at extreme values of the input. (less sensitive)

Solution:

- each_input - max_input , and their relative differences are the same
- In the real model, `x` is a matrix, in our example, the shape of x is (100,10), 100 images and 10 numbers, means, we have 100 maximum numbers, and each row will subtract its maximum number

```python
def softmax(x):
    # if without keepdims=True, the shape of max_x is (100,); now is 100,1
    max_x = np.max(x, axis=1,keepdims=True)
    # x.shape = 100,10; broadcasting 
    ex = np.exp(x-max_x)
    # softmax. sum each row
    sum_ex = np.sum(ex, axis=1, keepdims=True)
    return ex / sum_ex
```

Broadcasting is a method that NumPy uses to perform operations on arrays of different shapes in a way that *automatically expands the smaller array along its singleton dimensions* without using extra memory.

if `x = [[1, 2, 3],[4, 5, 6]]`, and `max_x = [[1],[4]]` , after broadcasting, `max_x = [[1,1,1],[4,4,4]]` 

## Loss Underflow

![Untitled 6](https://github.com/nemuzard/basicNLP/assets/44145324/36c07b23-6d90-4cc7-9514-56344ffc37aa)


As input closer to 0, may cause data underflow `-inf`

So we have to ‘control’ x. (Note: x resulted in Softmax cannot be negative but can be closer to 0 )

- So we have to modify Softmax

```python
def softmax(x):
    # if without keepdims=True, the shape of max_x is (100,); now is 100,1
    max_x = np.max(x, axis=1,keepdims=True)
    # x.shape = 100,10; broadcasting
    ex = np.exp(x-max_x)
    # softmax. sum each row
    sum_ex = np.sum(ex, axis=1, keepdims=True)
    result = ex/sum_ex
    #  Add a lower limit 
    result = np.clip(result, 1e-100,1)
    return result
```


modified bp.py


- In backpropagation, more layers mean we need to have more weighted matrices.
    - In a neural network, each layer requires the output from the previous layer as its input. If the previous layer has 𝑛 n neurons and the next layer has 𝑚 m neurons, the number of weighted parameters (or connections) between these two layers will indeed be 𝑛 × 𝑚.
    - improved feature extraction ability
- Coding part: Encapsulate the previously written backpropagation code to make it more aligned with PyTorch's logic, facilitating easier learning.

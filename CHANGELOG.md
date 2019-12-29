# CHANGE LOG

TODO
https://www.kaggle.com/mtax687/l2-regularization-of-neural-network-using-numpy
Add L1, L2 Regularization

### Commit #7

#### Added Functionality

- Use different weight and bias initialization strategy: Xavier`s
  [Link 1](https://hackernoon.com/how-to-initialize-weights-in-a-neural-net-so-it-performs-well-3e9302d4490f)
  [Link 2](http://cs231n.github.io/neural-networks-2/)

```python
# W = 0.01* np.random.randn(D,H) , old methods
w =np.random.randn((x_dim,y_dim))*np.sqrt(1/(ni+no)) # better method
 # `randn` samples from a zero mean, unit standard deviation gaussian.
```

Calibrating the variances with 1/sqrt(n), because variance grows with the number of inputs.

- Use 1/sqrt(n) for `tanh`
- Use 2/sqrt(n) for `relu`

Biases can be initialized safely at zeroes.

### Commit #6

#### Added Functionality

- Use different activation function for each layer
- Use different loss function for network

Available Activation Functions

- `sig`(Default): Sigmoid
- `relu`: `Re`ctified `l`inear `u`nit
- `tanh`: Hyperbolic Tangent Function

Available Loss Functions

- `mse`(Default): Mean squared error
- `bce`: Binary Cross Entropy

---

### Commit #5:

#### Refactor

- SinlePass into forward and backward pass

#### Bug Fix:

- Divided delta_W and delta_B with batch_size for proper calculation.

#### Minor Changes:

- Moved some Network class variables inside functions like loss_history etc.

---

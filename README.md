# todo

- use some data from online


## Why divided by batch_size
```
w[-k] += self.learning_rate * delta_w/batch_size
b[-k] += self.learning_rate * delta_b/batch_size
```
Since, delta bias or weights is calc over a batch of size say n,
sum of deltas can be above 1 upto n and np.exp can shoot
usually Numpy overflow error...
since weights are b/w 0, 1 we keep delta_w, delta_b b/w 0 and 1
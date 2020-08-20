# Gradient Descent

## Tip 1: Tuning your learning rates

### Vanilla Gradient descent

$$
w^{t+1} \leftarrow w^{t}-\eta^{t}g^{t} \\
\eta^{t}=\frac{\eta}{\sqrt{t+1}}, g^{t}=\frac{\partial{L(\Theta^{t})}}{\partial{w}}
$$



w is one parameters

### Adagrad

$$
w^{t+1} \leftarrow w^{t}-\frac{\eta^{t}}{\sigma^{t}}g^{t}\\
\eta^{t}=\frac{\eta}{\sqrt{t+1}}, g^{t}=\frac{\partial{L(\Theta^{t})}}{\partial{w}}\\
\sigma^{t}=\sqrt{\frac{\sum_{i=0}^{t}{(g^{i})^{2}}}{t+1}}
$$

整理得：
$$
w^{t+1} \leftarrow w^{t}-\frac{\eta^{t}}{\sqrt{\sum_{i=0}^{t}{(g^{i})^{2}}}}g^{t}
$$
原理：

The best step is :
$$
\frac{|First\;derivative|}{|Second\;derivative|}
$$

$$
\sum_{i=0}^{t}{(g^{i})^{2}} use \; first\;derivative\;to\;estimate\;second\;derivative
$$

## Tip 2: Stochastic Gradient Decent

Making the training faster.

### Gradient Descent

$$
\Theta^{i}=\Theta^{i-1}-\eta\nabla L(\Theta^{i-1})
$$



### Stochastic Gradient Descent

Pick an example 
$$
x^{i}
$$

$$
L^{n}=(\hat{y}^{n}-(b+\sum{w_{i}x_{i}^{n}}))^{2}\\\Theta^{i}=\Theta^{i-1}-\eta\nabla L(\Theta^{i-1})
$$

Loss for only one example.

## Tip 3: Feature Scaling

Make different features have the same scaling

For each dimension i:
$$
mean: m_{i}\\
standard\ deviation: \sigma_{i}\\
x^{r}_{i}\leftarrow\frac{x^{r}_{i}-m_{i}}{\sigma_{i}}
$$

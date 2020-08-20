# New Optimizers for Deep Learning

## SGD

$$
Start\ at\ position\ \Theta^{0}\\
Compute\ gradient\ at\ \Theta^{0}\\
Move\ to\ \Theta^{1}=\Theta^{0}-\eta\nabla{L(\Theta^{0})}\\
Comput\ gradient \ at \ \Theta^{1}\\
Move \ to \ \Theta^{2} =\Theta^{1}-\eta\nabla{L(\Theta^{1})}\\
...\\
Stop\ until\ \nabla(\Theta^{t})\approx0
$$

### SGD的示意图如下：

<img src="D:\2020暑期学习\李宏毅课程\[HW1]\sgd.PNG" alt="sgd" style="zoom:80%;" />

## SGDM（SGD with Momentum）

Movement: movement of last step minus gradient at present.
$$
Start\ at \ point \ \Theta^{0}\\
Movement \ v^{0}=0\\
Compute\ gradient \ at \ \Theta^{0}\\
Movement \ v^{1} = \lambda v^{0}-\eta \nabla L(\theta^{0})\\
Move \ to \ \Theta^{1}=\Theta^{0}+v^{1}\\
Compute \ gradient \ at \ \Theta^{1} \\
Movement\ v^{2}=\lambda v^{1}-\eta \nabla L(\theta^{1})\\
Move \ to \ \theta^{2}=\theta^{1}+v^{2}\\
....
$$
Movement not just based on gradient, but previous movement.
$$
v^{i} \ is \ actually \ the \ weighted \ sum \ of \ all\ the \ previous \ gradient:\\
\nabla L(\theta^{0}), \ \nabla L(\theta^{1}) \, \ ... \ \nabla(\theta^{i-1}) \\
v^{0}=0\\
v^{1}=-\eta \nabla L(\theta^{0})\\
v^{2}=-\lambda \eta \nabla L(\theta^{0})-\eta \nabla L(\theta^{1})\\
...
$$

### momentum的好处

在局部最优点即使导数为零，由于momentum的存在，使得在该点之后仍然能移动

### SGDM的示意图如下

![sgdm](D:\2020暑期学习\李宏毅课程\[HW1]\sgdm.PNG)

## Adagrad

$$
\theta_{t}=\theta_{t-1}-\frac{\eta}{\sqrt{\sum_{i=0}^{t-1}{(g_{i})^{2}}}}g_{t-1}
$$

Adagrad can solve : What if the gradients at the first few time steps are extremely large.

缺点是无法解决局部最优的问题，并且随着adagrad的进行，更新会越来越来越慢。

## RMSProp

$$
\theta_{t}=\theta_{t-1}-\frac{\eta}{\sqrt{v_{t}}}g_{t-1}\\
v_{1}=g_{0}^{2}\\
v_{t} = av_{t-1}+(1-a)(g_{t-1})^{2}
$$

Exponential moving average (EMA) of squared gradients is not monotonically increasing.

有效解决随着gradient descent的进行，收敛越来越慢的缺点，但是没有解决局部最优化的问题。

## Adam

将SGDM与RMSProp结合起来
$$
\theta_{t}=\theta_{t-1}-\frac{\eta}{\sqrt{\hat{v}_{t}}+\epsilon}\hat{m}_{t}\\
\hat{m}_{t}=\frac{m_{t}}{1-\beta_{1}^{t}}\\
m_{t}=\beta_{1}m_{t-1}+(1-\beta_{1})g_{t-1}\\
\hat{v}_{t}=\frac{v_{t}}{1-\beta_{2}^{t}}\\
v_{1}=g_{0}^{2}\\
v_{t}=\beta_{2}v_{t-1}+(1-\beta_{2})(g_{t-1})^{2}\\
\beta_{1}=0.99\\
\beta_{2}=0.999\\
\epsilon=10^{-8}
$$

### Adam vs SGDM

**Adam:** fast training, large generalization gap, unstable

**SGDM:** stable, little generalization gap, better convergence


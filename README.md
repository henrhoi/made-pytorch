# MADE (Masked Autoencoder Distribution Estimation) in PyTorch
PyTorch implementations of MADE (Masked Autoencoder Distribution Estimation) based on [MADE: Masked Autoencoder for Distribution Estimation](https://arxiv.org/abs/1502.03509) by Germain et. al. and autoregressive models.

## Models

**MADE - Fitting 2D Data:**

Implemented a [MADE](https://arxiv.org/abs/1502.03509) model through maximum likelihood to represent  <img src="https://render.githubusercontent.com/render/math?math=p(x_0,x_1)">  on the given datasets, with any autoregressive ordering.

**MADE:**

Implemented a MADE model on the binary image datasets Shape and MNIST. Given some binary image of height <img src="https://render.githubusercontent.com/render/math?math=H"> and width <img src="https://render.githubusercontent.com/render/math?math=W">, we can represent image <img src="https://render.githubusercontent.com/render/math?math=x\in \{0, 1\}^{H\times W}"> as a flattened binary vector <img src="https://render.githubusercontent.com/render/math?math=x\in \{0, 1\}^{HW}"> to input into MADE to model <img src="https://render.githubusercontent.com/render/math?math=p_\theta(x) = \prod_{i=1}^{HW} p_\theta(x_i|x_{ < i})">.

Illustration of MADE-model:

![made](https://i.imgur.com/agJN65k.png)

**Fitting a histogram:**

Let <img src="https://render.githubusercontent.com/render/math?math=\theta = (\theta_0, \dots, \theta_{d-1}) \in \mathbb{R}^d"> and define the model <img src="https://render.githubusercontent.com/render/math?math=p_\theta(x) = \dfrac{e^{\theta_x}}{\sum_{x'}e^{\theta_{x'}}}">

Fit <img src="https://render.githubusercontent.com/render/math?math=p_\theta"> with maximum likelihood via stochastic gradient descent on the training set, using <img src="https://render.githubusercontent.com/render/math?math=\theta"> initialized to zero.

**Discretized Mixture of Logistics:**

Let us model <img src="https://render.githubusercontent.com/render/math?math=p_\theta(x)"> as a **discretized** mixture of 4 logistics such that <img src="https://render.githubusercontent.com/render/math?math=p_\theta(x) = \sum_{i=1}^4 \pi_i[\sigma((x+0.5 - \mu_i)/s_i) - \sigma((x-0.5-\mu_i)/s_i)]">


For the edge case of when <img src="https://render.githubusercontent.com/render/math?math=x = 0">, we replace <img src="https://render.githubusercontent.com/render/math?math=x-0.5"> by <img src="https://render.githubusercontent.com/render/math?math=-\infty">, and for <img src="https://render.githubusercontent.com/render/math?math=x = d-1">, we replace <img src="https://render.githubusercontent.com/render/math?math=x+0.5"> by <img src="https://render.githubusercontent.com/render/math?math=\infty">.

One may find the [PixelCNN++](https://arxiv.org/abs/1701.05517) helpful for more information on discretized mixture of logistics.

## Results and samples

| Model | Dataset |  Result/Samples |
|:---:|:---:|:---:|
| MADE 2D (Smiley)                | ![3](https://i.imgur.com/OLtbWDr.png)  | ![33](https://i.imgur.com/En6OGv4.png)  |
| MADE 2D (Geoffrey Hinton)       | ![4](https://i.imgur.com/c8PYEtz.png)  | ![44](https://i.imgur.com/kvPCx4y.png)  |
| MADE (Shapes)                   | ![5](https://i.imgur.com/QGWaShQ.png)  | ![55](https://i.imgur.com/o0z8lEC.png)  |
| MADE (MNIST)                    | ![6](https://i.imgur.com/T0acOeg.png)  | ![66](https://i.imgur.com/pMjtBbj.png)  |
| Simple histogram                | ![1](https://i.imgur.com/jPCPDmD.png)  | ![11](https://i.imgur.com/MLujTCa.png)  |
| Discretized Mixture of Logistics| ![2](https://i.imgur.com/jPCPDmD.png)  | ![22](https://i.imgur.com/OthJ9Gf.png)  |

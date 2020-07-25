<img src="https://avatars1.githubusercontent.com/u/4284691?v=3&s=200" title="BreezeForest Structure" >

# BreezeForest

> An efficient flow based generative model


## Setup:


```shell
$ pip install requirements
```


## Run Demonstration:

```shell
$ python multi_dataset_demo
```
or 

```shell
$ python one_dataset_demo
```
## Background

FLOW based generative models are usually used to model the latent space density, 
Block-wise Neural Autoregressive Flow (BNAF) first published by Nicola De CAO: 
 
> https://nicola-decao.github.io/assets/pdf/UAI_poster.pdf  
> https://arxiv.org/abs/1904.04676  
> https://github.com/nicola-decao/BNAF   
 
 is one of the most powerful among them. A BNAF with only one hidden layer has been proven to be a universal density estimator. Consequently, compare to other models, BNAF requires much fewer layers to achieve the same precision. Furthermore, as an autoregressive flow model, BNAF can be used to boost Explanable AI as described by:
Graphical Normalizing Flow: https://arxiv.org/pdf/2006.02548.pdf. 

## Contributions

I had an idea similar with BNAF,  I named it "BreezeForest".  BNAF had been published 3 month before I finished experiments and planned to write down the paper for "BreezeForest".  Despite the disappointement, I am also encouraged by people who share the same idea with me and they did really good job. So I decided to go deeper in this direction.

This repository show part of of my results, the contributions compare to BNAF is the following:

1. BreezeForest(~BNAF) has a special architecture, which enabled us to use numerical differential operator to compute the loss function instead of computing layer by layer the whole Jacobian matrix as did BNAF. This reduces the objective complexity by one order of magnitude(from O(N^3) to O(N^2)).

2. I developed a batched bisection algorithm to find the inverse of the BreezeForest. This can be used to generate new samples from random uniform distribution.

3. Given a finite number of samples and complex enough model, One can always get infinitely high log likelihood by replicating from samples. Consequently, a generative model should have properly defined constraint to avoid this issue so as to generate unseen sample. I merged BreezeForest with a Gaussian like density estimator into one neural network to solve regularize this issue. 

## Method Illustration

### Theorical fundation:

BreezeForest is a bijective function (BF) that map n dimensional continious distribution X \~P(X) to n dimensional independant Uniform distribution U \~uniform(U):
U = BF(X)
We assume the cumulative density function of P(x) is: 

F(x1,x2...xn) = F(xn|xn-1...x1)...F(x3|x1, x2)F(x2|x1)F(x1)
let Fi(x) = F(x|xi-1...x1), note that Fi depends on x1...xi-1 and every where non decreasing. 
x1...xi-1 give influence on Fi using breeze connection.
Then we can parametrize F using BF by posing:
BF(x1,x2...xn) = F1(x1), F2(x2)...Fn-1(xn-1), Fn(xn)

logP(X) = logdet(JacobianBF(X)) where JacobianBF(X) is always lower triangular
=  log(dF1(x1)/dx1) + log( dF1(x2)/dx2) ... + log(dFn(xn)/dxn)



### Jacobian Free determinant computation using numerical approximation to the derivative:

Instead of computing jacobian matrix layer by layer, we can compute the determinant of jacobian by doing 
only 2 forward pass.

First forward pass can compute: 

BF(x1, x2...xn)  = F1(x1), F2(x2)...Fn-1(xn-1), Fn(xn) with all breeze connections used to parametrize Fi

Once Fi is computed, we can do the second forward pass through them to get: 

F1(x1+delta), F2(x2+delta)...Fn-1(xn-1+delta), Fn(xn+delta) 

Finally:
logP(x1,x2...xn) = logP(xn|xn-1...x1) +...+ logP(x3|x1, x2) + logP(x2|x1) + logP(x1)
=limit delta->0 : sum{log((Fi(xi+delta)-Fi(xi))/delta)} for (i=1..n)

### Control the ability of generating unseen samples:



# BreezeForest

Setup:

pip install requirements

Run Demonstration:

python multi_dataset_demo
or
python one_dataset_demo

Background

FLOW based generative models are usually used to model latent space density, 

Block-wise Neural Autoregressive Flow(BNAF) first published by Nicola De CAO: 
  https://nicola-decao.github.io/assets/pdf/UAI_poster.pdf
  https://arxiv.org/abs/1904.04676
  https://github.com/nicola-decao/BNAF
is one of the FLOW based generative models.  A BNAF with only one hidden layer can be proven to be an universal density estimator. consequently,  
compare to other models, BNAF requires much fewer layers to achieve the same precision. Furthermore, as an autoregressive flow model, BNAF  can be used to boost Explanable AI as described by:
Graphical Normalizing Flow: https://arxiv.org/pdf/2006.02548.pdf. 

Contributions

I came up with the similar idea and code(I named it "BreezeForest") as BNAF 3 month after it's publication. Encouraged by  people who share the same idea with me, I conducted deeper research on this direction. 

This repository is a snippet of my code developed for this research topic. The contributions of this repository compare to previous works is the following:

1. BNAF has a special architechture, which enabled us to use numerical differantial operator to compute the determinant of the triangular jacobian matrix of BNAF without computing layer by layer the whole jacobian matrix. This reduces the objective complexity by one order of magnitude.

2. I developed a batched bisection algorithm to find the inverse of BNAF. This can be used to generate new samples from gaussian noises, the generation demonstration of this repo is based on it. 

3. Given finite number of samples and complex engough model, One can always get inifintely high log likelihood by replicating from samples. Consequently, a generative model should have
properly defined constraint to avoid this issue so as to be able to generate unseen sample. I merged BNAF with a gaussian like density estimator  into one neural network to solve regularize this issue.   

Method Illustration

1. Theorical fundation:

BreezeForest is a bijective function (BF) that map n dimensional continious distribution X~P(X) to n dimensional independant Uniform distribution U~uniform(U):
U = BF(X)
We assume the cumulative density function of P(x) is: 

F(x1,x2...xn) = F(xn|xn-1...x1)...F(x3|x1, x2)F(x2|x1)F(x1)
let Fi(x) = F(x|xi-1...x1), note that Fi depends on x1...xi-1 and every where non decreasing. 
x1...xi-1 give influence on Fi using breeze connection.
Then we can parametrize F using BF by posing:
BF(x1,x2...xn) = F1(x1), F2(x2)...Fn-1(xn-1), Fn(xn)

logP(X) = logdet(JacobianBF(X)) where JacobianBF(X) is always lower triangular
=  log(dF1(x1)/dx1) + log( dF1(x2)/dx2) ... + log(dFn(xn)/dxn)



2. Jacobian Free determinant computation using numerical approximation to the derivative:

Instead of computing jacobian matrix layer by layer, we can compute the determinant of jacobian by doing 
only 2 forward pass.

First forward pass can compute: 

BF(x1, x2...xn)  = F1(x1), F2(x2)...Fn-1(xn-1), Fn(xn) with all breeze connections used to parametrize Fi

Once Fi is computed, we can do the second forward pass through them to get: 

F1(x1+delta), F2(x2+delta)...Fn-1(xn-1+delta), Fn(xn+delta) 

Finally:
logP(x1,x2...xn) = logP(xn|xn-1...x1) +...+ logP(x3|x1, x2) + logP(x2|x1) + logP(x1)
=limit delta->0 : sum{log((Fi(xi+delta)-Fi(xi))/delta)} for (i=1..n)

3. Control the ability of generating unseen samples:



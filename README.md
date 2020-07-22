# BreezeForest

Background

FLOW based generative models are usually used to model latent space density, 

Block-wise Neural Autoregressive Flow(BNAF) first published by Nicola De CAO: 
  https://nicola-decao.github.io/assets/pdf/UAI_poster.pdf
  https://arxiv.org/abs/1904.04676
  https://github.com/nicola-decao/BNAF
is one of the FLOW based generative models.  A BNAF with only one hidden layer can be proven to be an universal density estimator. consequently,  
compare to other models, BNAF requires much fewer layers to achieve the same precision. Furthermore, as an autoregressive flow model, BNAF  can be used to boost Explanable AI as described by 
Graphical Normalizing Flow: https://arxiv.org/pdf/2006.02548.pdf. 

Contributions

I came up with the similar idea and code(I named it "BreezeForest") as BNAF 3 month after it's publication. Encouraged by  people who share the same idea with me, I conducted deeper research on this directrion. 

This repository is a snippet of my code developed for this research topic. The contributions of this repository compare to previous works is the following:

1. BNAF has a special architechture, which enabled me to use numerical differantial operator to compute the determinant of the triangular jacobian matrix of BNAF  without computing layer by layer the whole
jacobian matrix. This reduces the objective complexity by one order of magnitude.

2. I developed a bisection algorithm to find the inverse of BNAF. This can be used to generate new samples from gaussian noises, the generation demonstration of this repo is based on it. 

3. Given finite number of samples and complex engough model, One can always get inifintely high log likelihood by replicating from samples. Consequently, a generative model should have
properly defined constraint to avoid this issue so as to be able to generate unseen sample. I merged BNAF with a gaussian like density estimator  into one neural network to solve regularize this issue.   

Method Illustration
0.
1.
2.
3.

Run Demonstration

python multi_dataset_demo

# Bayesian Optimization of Neural Architecture
This project is implemented based on https://arxiv.org/abs/1802.07191.
# Introduction
Exploring a better Neural Network Architecture for Image Classification has been an actively researched topic in Deep Learning. In our project, we shed our light on using Bayesian Optimization to automatically optimize Neural Network Architecture for classifying CIFAR-10 Image Dataset. 

For more information, please refer to the paper and our report pdf in this repo.
# Usage
Create the initial pool of architecture:
```
python reinit_pool.py
```
Based on the accuarcy result in current pool (saved in models/), get new architectures with highest acquisition value to evaluate:
```
python pool.py
```

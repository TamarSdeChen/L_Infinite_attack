# L∞ Attacks via Program Synthesis
This repository contains the code for the project "L∞ Attacks via Program Synthesis," which is based on the paper [One Pixel Adversarial Attacks via Sketched Programs](https://dl.acm.org/doi/pdf/10.1145/3591301). The project was conducted by [Tamar Sde Chen](https://github.com/TamarSdeChen) and [Hadar Rosenberg](https://github.com/HadarRosenberg), under the supervision of [Tom Yuviler](https://tomyuviler.github.io/) and [Dana Drachsler-Cohen](https://ddana.net.technion.ac.il/) from the Technion, Israel Institute of Technology.
Our project introduces an approach to generate L∞ adversarial attacks with a reduced number of queries to the neural network, leveraging the power of program synthesis.

## Introduction
Neural networks have shown tremendous success in a variety of applications. At the same time, they have also been shown to be vulnerable to various attacks. One attack that has drawn a lot of attention is the adversarial example attack.
An adversarial example is generated by adding a small perturbation (customized small noise) to a correctly classified input to cause a network classifier to misclassify. For example, consider a neural network implementing an image classifier given an image, it classifies the image into a class. An attacker may add a small noise to the image, such that the new image is misclassified by the network, although a human being will identify that the classification need not change. The amount of noise is typically constrained by some norm and a numerical budget.

In our project, we will implement the L∞ attack. In this attack, the attacker is given a small real number ϵ and he can perturb any pixel by ±ϵ. 

Our project is based on a recent paper "One Pixel Adversarial Attacks via Sketched Programs". This study proposed the first algorithm that computes a computer program for computing L0 attacks. In an L0 attacks, the attacker is given a small bound t and he can perturb at most t pixels, each perturbed pixel can have an arbitrary value (in its valid range).
(one and a few pixel attacks). The idea presented in the original paper is to generate candidate attacks and submit them one-by-one to the network based on a priority queue. If the attack is successful, the algorithm is completed. Otherwise, the prioritization of the remaining candidates is updated based on the learned conditions. 

The goal of our project is to extend the approach that has been shown in the original paper to L∞ attacks and show how a computer program can compute an L∞ attack with a minimal number of queries. We have started with a program sketch like the L0 program sketch and updated it to the L∞ setting.
<img src="beforeandafter.png" alt="Example Image" width="500"/>
## Results
Our L∞ attack reached to 6.5% success rate on 2X2 grid, which means that the algorithm found a successful attack to 6.5% from the test set images with 134 as the average number of queries until finding the successful attack.
Additionally, our analysis includes calculating the standard deviation, resulting in a value of 324.

To analyze the variability in our dataset and pinpoint any potential anomalies, we create a histogram illustrating the distribution of attack outcomes:
![Example Image](histogram.png = 250x250)

## Models
CIFAR-10 models adapted from [![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.4431043.svg)](http://dx.doi.org/10.5281/zenodo.4431043). 

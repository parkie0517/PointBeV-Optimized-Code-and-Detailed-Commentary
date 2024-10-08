# PointBeV-Optimized-Code-and-Detailed-Commentary

## What is PointBeV?
- PointBev is a SOTA BeV perception model.
- Below is a summary of this model.
    1. The authors proposed a Sparse Feature Pulling module  
        - This reduced the computations during the training and inference time.
    2. Pulls Features from RGB feature maps using 3D coordinates.
    3. Uses past frames by using Submanifold Attention module.

## Purpose of thie Repo
- The purpose of this repo is three-fold.
  1. This repository contains detailed comments to facilitate understanding.
  2. Optimized some of the original code for better efficiency without sacrificing performance.
  3. Most importantly, for my own study!

## Prerequisites  
- This repo ueses **Pytorch Lightning**  
  - Therefore, the code does not explicitly use a <ins>for loop</ins> to iterate through epochs.
  - Rather, it uses a <ins>Trainer</ins> class to trian the neural network. 
  - If you have any difficulties understanding how the Pytorch Lightning's Trainer class operates, I highly recommend that you read the [Official Documentation](https://lightning.ai/docs/pytorch/stable/common/trainer.html)

## Citation
- Link to the original code
  - https://github.com/valeoai/PointBeV
- Link to the paper
  - https://arxiv.org/abs/2312.00703
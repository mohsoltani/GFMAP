# A General Framework for Multimodal Argument Persuasiveness Classification of Tweets

## Introduction
This repository houses the official implementation of A General Framework for Multimodal Argument Persuasiveness Classification of Tweets, alongside the paper accepted for presentation at the Argument Mining workshop at the EMNLP 2023. 

## Framework Design

The framework consists of three main models:

### Model 1 and Model 2
- Model 1 and Model 2 extract text and image features from each tweet as vectors of sizes `a` and `b`, respectively.

### Multimodal Fusion
- Multimodal fusion combines these vectors into a single vector of size `c`, where `c = a + b`.

### Formation of Feature Matrix
- The n tweet feature vectors jointly form a matrix `C` ∈ ℝ<c×n>.
- Alongside the n task-specific labels, they serve as input for Model 3.
  
![Framework Design](framework.svg)

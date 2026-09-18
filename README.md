# MLwPTaSL
Machine Learning with PyTorch and Scikit-Learn

![alt text](bookcover/image.png)

---

# Table of Content

- Preface  (xxiii)



| [ < ](https://github.com/engineer-e) | T |  N |
|---|--- |--- |
| 1 | [Giving Computers the Ability to Learn from Data](#chapter-1-giving-computers-the-ability-to-learn-from-data)|  1 | 
| 2 | [Training Simple Machine Learning Algorithm for Classification](#chapter-2-training-simple-machine-learning-algorithm-for-classification) |  19 |
| 3 | [A Tour of Machine Learning Classifiers Using Scikit-Learn](#chapter-3-a-tour-of-machine-learning-classifiers-using-scikit-learn) |  53 |
| 4 | Building Good Training Datasets - Data Preprocessing   | 105 |
| 5 | Compressing Data via Dimensionality Reduction | 139 |
| 6 | Learning Best Practices for Model Evaluation and Hyperparameter Tuning  |171 |
| 7 | Combining Different Models for Ensemble Learning  | 205 |
| 8 | Applying Machine Learning to Sentiment Analysis  | 247 |
| 9 | Predicting Continuous Target Variables with Regression Analysis  | 269 |
| 10 | Working with Unlabeled Data - Clustering Analysis  | 305 |
| 11 | Implementing a Multilayer Artifical Neural Network from Scratch | 335 |
| 12 | Parallelizing Neural Network Training with PyTorch  | 369 |
| 13 | Going Deeper - The Mechanics of PyTorch  | 409 |
| 14 | Classifying Images with Deep Convolutional Neural Networks  | 451 |
| 15 | Modeling Sequential Data Using Recurrent Neural Networks  | 499 |
| 16 | Transformers - Improving Natural Language Processing with Attention Mechanisms  | 539 |
| 17 | Generative Adversarial Networks for Synthesizing New Data  | 589 |
| 18 | Graph Neural Networks for Capturing Dependencies in Graph Structured Data  | 637 |
| 19 | Reinforcement Learning for Decision Making in Complex Environment  | 673 |

- Other Books You May Enjoy (719)
- Index (723)

---

# Chapter 1: Giving Computers the Ability to Learn from Data



| [ < ](#table-of-content) | T | N |
|---|--- |--- |
| 1 | Building intelligent machines to transform data into knowledge  | 1 | 
| 2 | [The three different types of machine learning](#the-three-different-types-of-machine-learning) | 2 |
| 3 | [Introduction to the basic terminology and notations](#introduction-to-the-basic-terminology-and-notations)  | 9 |
| 4 | [A roadmap for building machine learning systems](#a-roadmap-for-building-machine-learning-systems) | 12 |
| 5 | [Using Python for maching learning](#using-python-for-maching-learning)  | 14 |
| 6 | Summary  | 17|

## The three different types of machine learning

| [ < ](#chapter-1-giving-computers-the-ability-to-learn-from-data) | T | N |
|---|--- |--- |
| 1 | [Making predictions about the future with supervised learning](#making-predictions-about-the-future-with-supervised-learning)  | 3|
| 2 | Solving interactive problems with reinforcement learning | 6 |
| 3 | [Discovering hidden structures with unsupervised learning](#discovering-hidden-structures-with-unsupervised-learning) | 7|

### Making predictions about the future with supervised learning

| [ < ](#the-three-different-types-of-machine-learning) | T | N |
|---|--- |--- |
| 1 | Classification for predicting class lebels | 4 |
| 2 | Regression for predicting continuous outcomes | 5 |

### Discovering hidden structures with unsupervised learning

| [ < ](#the-three-different-types-of-machine-learning) | T | N |
|---|--- |--- |
| 1 | Finding subgroups with clustering  | 8 |
|2 | Dimensionality reduction for data compression | 8|


## Introduction to the basic terminology and notations


| [ < ](#chapter-1-giving-computers-the-ability-to-learn-from-data) | T | N |
|---|--- |--- |
| 1 | Notation and conventions used in this book | 9 |
| 2 | Machine learning terminology | 11 |

## A roadmap for building machine learning systems


| [ < ](#chapter-1-giving-computers-the-ability-to-learn-from-data) | T | N |
|---|--- |--- |
| 1 | Preprocessing - getting data into shape | 13 |
| 2 | Training and selecting a predictive model | 13 |
| 3 | Evaluating models and predicting unseen data instances | 14 |


## Using Python for maching learning 

| [ < ](#chapter-1-giving-computers-the-ability-to-learn-from-data) | T | N |
|---|--- |--- |
| 1 | Installing Python and packages from the Python Package Index | 14 |
| 2 | Using the Anaconda Python distribution and package manager | 15 |
| 3 | Packages for scientific computing, data science, and machine learning | 16 |

---

# Chapter 2: Training Simple Machine Learning Algorithm for Classification

| [ < ](#table-of-content) | T | N |
|---|--- |--- |
| 1 | [Artifical neurons - a brief glimpse into the early history of machine learning](#artifical-neurons---a-brief-glimpse-into-the-early-history-of-machine-learning) | 19 |
| 2 | [Implementing a perceptron learning algorithm in Python](#implementing-a-perceptron-learning-algorithm-in-python) | 25 |
| 3 | [Adaptive linear neurons and the convergence of learning](#adaptive-linear-neurons-and-the-convergence-of-learning) | 35 |
| 4 | Summary | 51 |

## Artifical neurons - a brief glimpse into the early history of machine learning

| [ < ](#chapter-2-training-simple-machine-learning-algorithm-for-classification) | T | N |
|---|--- |--- |
| 1 | The formal definition of an artifical neuron | 20 |
| 2 | The perceptron learning rule | 22 |


## Implementing a perceptron learning algorithm in Python
| [ < ](#chapter-2-training-simple-machine-learning-algorithm-for-classification) | T | N |
|---|--- |--- |
| 1 | An object-oriented perceptron API  | 25 |
| 2 | Training a perceptron model on the Iris dataset | 29 |



## Adaptive linear neurons and the convergence of learning

| [ < ](#chapter-2-training-simple-machine-learning-algorithm-for-classification) | T | N |
|---|--- |--- |
| 1 | Minimizing loss functions with gradient descent | 37 | 
| 2 | Implementing Adaline in Python | 39 | 
| 3 | Improving gradient descent through feature scaling | 43 | 
| 4 | Large-scale machine learning and stochastic gradient descent | 45 |

---

# Chapter 3: A Tour of Machine Learning Classifiers Using Scikit-Learn

| [ < ](#table-of-content) | T | N |
|---|--- |--- |
| 1 | Choosing a classification algorithm  | 53 |
| 2 | First steps with Scikit-Learn - training a perceptron | 54 |
| 3 | [Modeling class probabilities via logistic regression](#modeling-class-probabilities-via-logistic-regression) | 59 |
| 4 | [Maximum margin classification with Support vector machine](#maximum-margin-classification-with-support-vector-machine) | 76| 
| 5 | [Solving nonlinear problem using a Kernel SVM](#solving-nonlinear-problem-using-a-kernel-svm) | 80 |
| 6 | [Decision Tree Learning](#decision-tree-learning) | 86 | 
| 7 | K-nearest neighbors - a lazy learning algorithm | 98 | 
| 8 | Summary | 102 | 

## Modeling class probabilities via logistic regression

| [ < ](#chapter-3-a-tour-of-machine-learning-classifiers-using-scikit-learn) | T | N |
| --- | --- | --- |
| 1 | Logistic regression and conditional probabilities | 60 |
| 2 | Learning the model weights via the logistic loss function | 63 |
| 3 | Converting an Adaline implementation into an algorithm for logistic regression | 66 |
| 4 | Training a logistic regression model with scikit-learn | 70 |
| 5 | Tackling overfitting via regularization | 73 |


## Maximum margin classification with Support vector machine

| [ < ](#chapter-3-a-tour-of-machine-learning-classifiers-using-scikit-learn) | T | N |
| --- | --- |--- |
| 1 | Maximum margin intuition | 77 |
| 2 | Dealing with a nonlinearly separable case using slack variables | 77 |
| 3 | Alternative implementation in Scikit-learn | 79 |


## Solving nonlinear problem using a Kernel SVM

| [ < ](#chapter-3-a-tour-of-machine-learning-classifiers-using-scikit-learn) | T | N |
| --- | --- | --- |
| 1 | Kernal methods for linearly inseparable data | 80 |
| 2 | Using the kernel trick to find separating hyperplanes in a high dimensional spaces | 82 |


## Decision Tree Learning

| [ < ](#chapter-3-a-tour-of-machine-learning-classifiers-using-scikit-learn) | T | N |
| --- | --- |--- |
| 1 | Maximizing IG - getting the most bang for your buck | 88 |
| 2 | Building a decision tree | 92 |
| 3 | Combining multiple decisions tree via random forests | 95 |

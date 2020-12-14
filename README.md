
<br />
<p align="center">
  <a href="https://github.com/mohitgulla/Edge">
    <img src="data/results/archive/banner.png" alt="Logo" width="500" height="100">
  </a>

  <h3 align="center">Energy Efficient Deep Learning</h3>

  <p align="center">
    Repository of Capstone Work at Data Science Institute, Columbia University in collaboration with GE Research 
    <br />
  </p>
</p>


<summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#contributors">Contributors</a>
    </li>
    <li><a href="#usage">Usage</a>
        <ul><a href="#demo-/-tutorial">Demo / Tutorial</a></ul>
        <ul><a href="#directory-structure">Directory Structure</a></ul>
    </li>
    <li>
      <a href="#methods">Methodology</a>
        <ul><a href="#post-training-quantization">Post-Training Quantization</a></ul>
        <ul><a href="#pruning">Pruning</a></ul>
        <ul><a href="#quantization-aware-training">Quantization-Aware Training</a></ul>
    </li>
    <li>
      <a href="#future-work">Future Work</a>
    </li>
  </ol>


## About The Project

The project aims to develop techniques for training and inference of machine learning models with reduced carbon footprint. 
Recent estimates suggest training deep learning models such as BERT, ELMo and GPT-2 requires training on multiple GPU's for a few days. 
The carbon emissions from training a deep learning model is equivalent to 5 times the lifetime emissions of an average car. 
Hence, GE requires low-latency and lighter deep learning models without compromising accuracy, which can be deployed on GE's EDGE devices. 
Our objective is to explore techniques that enable us to store a model in lower precision and assess its effect during inference. 

## Contributors

<b>Mentors:</b> 

Tapan Shah - Lead Machine Learning Scientist, GE Research<br>
Eleni Drinea - Lecturer, Data Science Institute, Columbia University<br>

<b>Capstone Team:</b>

<a href="https://github.com/mohitgulla">Mohit Gulla</a>,
<a href="https://github.com/kumari-nishu">Kumari Nishu</a>,
<a href="https://github.com/NeelamPatodia">Neelam Patodia</a>,
<a href="https://github.com/Prasham8897">Prasham Sheth</a>,
<a href="https://github.com/Pritam-Biswas">Pritam Biswas</a>



## Usage

#### Demo / Tutorial

For a detailed walkthrough of the main techniques, i.e. multi-point mixed precision post-training quantization, pruning, and quantization aware training, please refer to notebook `Demo_Code.ipynb`.

#### Directory Structure

- `data` - contains .py files with contain class definition of PyTorch dataset and the corresponding .dat file. The datasets explored are ANN based Classification: <a href="https://www.kaggle.com/filippoo/deep-learning-az-ann">Churn</a> and <a href="http://archive.ics.uci.edu/ml">Telescope</a>, ANN based Regression: <a href="https://sci2s.ugr.es/keel/dataset.php?cod=84#sub1">MV Data</a> and <a href="https://sci2s.ugr.es/keel/dataset.php?cod=83#sub2 ">California Housing</a> and CNN based Classification: <a href = "https://www.cs.toronto.edu/~kriz/cifar.html">CIFAR-100</a> and <a href="https://deepobs.readthedocs.io/en/stable/api/datasets/fmnist.html">FMNIST</a>. A subdirectory `results` conatins .csv files which track training and validation accuracy and loss at different precision levels from the experiments that were conducted.   

- `model` - contains .py files with model class definition for Dense Neural Networks (DNNs) and Convolutional Neural Networks (CNNs). The various architectures of each model type are defined as separate class objects within its corresponding .py file. 

- `model_artifacts` - contains .pt files of full precision trained models.

- `utils` - contains .py files with post-training quantization, pruning and quantization-aware training methods which were explored. In post-training quantization we have implemented single-point methods such as mid-rise quantization, regular rounding, stochastic rounding and multi-point methods such as mixed precision multipoint quantization. Each method is designed to be a standalone functionality that can be used anywhere else if needed. It also contains utility code for fetching datasets, plotting graphs, etc.

All *.ipynb and *.py files in main directory has the comprehensive code for model training, quantization and evaluation. They leverage the code base from the sub-directories.

## Methodology

#### Post Training Quantization

<i>Single-point Quantization approximates a weight value using a single low precision number.</i>

1. Mid-Rise 
- Delta - controls granularity of data quantization, high delta implies high quantization and significant loss of information
- Uniform division of range of Weight values into 2^p bins for p precision
- w_quantized = Delta * (floor(w/Delta) + 0.5)

2. Regular Rounding
- Quantization Set - collect a set of landmark values using uniform bin, histogram, prior normal on weight values
- Map each weight value to the nearest landmark value from quantization set

3. Stochastic Rounding
- Quantization Set - collect a set of landmark values using uniform bis, histogram, prior normal on weight values
- Assign each weight value to either the closest smaller value or the closest larger value from quantization set probabilistically

<i>Multi-point Quantization approximates a weight value using linear combination of multiple values of low precision.

4.  Multi-point - mixed precision method 
- Assign more bits to important layers, and fewer bits to unimportant layers to balance the accuracy and cost more efficiently
- Achieves the same flexibility as mixed precision hardware but using only a single-precision level
- The quantization set is constructed using a uniform grid on [-1, 1] with increment epsilon and each weight value w is approximated as a linear combination of low precision weight vectors.

#### Pruning

<i>It is a method of compression that involves removing less contributing weights from a trained model.</i>

- Setting the neural network parameters’ values to zero to remove what we estimate are less contributing (unnecessary connections) between the layers of a neural network.
- Using the magnitude of weights to determine the importance of the weights towards the model’s performance.

#### Quantization-Aware Training

<i>It is a process of training the model assuming that it will be quantized later during inference.</i> 

The steps involved in QAT are:
1. Initialize a full precision model
2. Quantize model weights per layer
3. Forward propagate and compute gradients
4. Update gradients using straight through estimator
5. Backprop on full precision model and return quantized model

## Future Work

<b>Model Size</b>

To get a complete picture of each method’s effectiveness, we need to observe model size at different levels of precision. This relates to our objective of reducing the carbon footprint of deep learning models.

<b>Quantize Activations</b>

Along with quantization of weights, explore quantization of activations as well.

<b>Improve Training Algorithm</b>

Most of the carbon emissions are caused due to the intensive computations required during the training. (e.g.) BERT and GPT-3 require a lot a computation to learn the parameters. We can explore techniques to get smart weight updates and reduce computations required during the training. 

<b>Hardware Simulations</b>

Experiment on specialized low-precision hardware to accurately evaluate different quantization techniques.


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo.svg?style=for-the-badge
[contributors-url]: https://github.com/mohitgulla/Edge/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/github_username



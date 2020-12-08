
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
        <ul><a href="#demo">Demo / Tutorial</a></ul>
        <ul><a href="#directories">Directory Structure</a></ul>
    </li>
    <li>
      <a href="#methods">Methodology</a>
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

- `data` - contains .py files with contain class definition of PyTorch dataset and the corresponding .dat file. The datasets explored in our experiments are ANN based Classification: <a href="https://www.kaggle.com/filippoo/deep-learning-az-ann">Churn Data</a> and <a href="http://archive.ics.uci.edu/ml">Telescope Data</a>, ANN based Regression: <a href="https://sci2s.ugr.es/keel/dataset.php?cod=84#sub1">MV Data</a> and <a href="https://sci2s.ugr.es/keel/dataset.php?cod=83#sub2 ">California Housing Data</a> and CNN based Classification: <a href = "https://www.cs.toronto.edu/~kriz/cifar.html">CIFAR-100</a> and <a href="https://deepobs.readthedocs.io/en/stable/api/datasets/fmnist.html">FMNIST Data</a>. A subdirectory `results` conatins .csv files which track training and validation accuracy and loss at different precision levels from the experiments that were conducted.   

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



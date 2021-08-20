

```
           _       _ _____   ___  ______ 
          (_)     (_)  __ \ / _ \ | ___ \
 _ __ ___  _ _ __  _| |  \// /_\ \| |_/ /
| '_ ` _ \| | '_ \| | | __ |  _  ||  __/ 
| | | | | | | | | | | |_\ \| | | || |    
|_| |_| |_|_|_| |_|_|\____/\_| |_/\_|    
                                         
```

### Introduction

miniGAP is a proxy application for molecular and materials property prediction using the Gaussian Process Approximation.
This is code is meant to run in multiple architectures, such as many-core and accelerators.

### Installation

This code could be installed within an `conda` enviroment as: 

`conda create -f environment.yml`

Then 

`conda activate minigap`

#### Creating an custom kernel in Jupiter

```
conda activate minigap
python -m ipykernel install --user --name "minigap"
```
#### Dependencies:

- python >= 3.6 
- dscribe 
- SYCL compiler
- Tensorflow
- Tensorflow-probability
- GPflow
- scikit-learn

### What is inside?

- data: Initial XYZ, sample trajectories, and downloaded material.
- code: Repo specific modules for training and creating the models.
- notebooks: 
- results: Figures and models
- media: Assorted Images

### Contributors

Contributions are always welcome. Contributors should fork this repository and submit a merge request for review of the code.



### References

Dscribe

GAP



Copyright 2021 Argonne UChicago LLC



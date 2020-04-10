# Multi-Duration Saliency Models

Source code for our paper "How Many Glances? Modeling Multi-duration Saliency". 

This repo contains models and source code for predicting multi-duration saliency. To get started, download the [CodeCharts1k multi-duration saliency dataset](http://multiduration-saliency.csail.mit.edu/codecharts_data.zip). 

<!-- and walk through the notebook `train_multiduration.ipynb`. --> 

<!-- Models are written in Keras 2. --> 


<!-- ## Models -->

<!-- This repo contains source code for the following models: --> 
<!-- - Multi-Duration Saliency Excited Model (MD-SEM): a lightweight network designed for predicting multi-duration saliency -->
<!-- - SAM Multi-Duration (SAM-MD): a version of SAM modified to produce multiple outputs corresponding to multiple viewing durations -->
<!-- - SAM-Resnet: a reimplementation of the original SAM in Keras 2. Handles only single-duration saliency prediction. -->

<!-- ## Datasets -->

<!-- Our data-loading code supports the following datasets (must be downloaded separately): -->

<!-- #### Multi-duration datasets -->
<!-- - CodeCharts1k -->
<!-- - SALICON-MD (SALICON data broken into approximate times based on timestamps) -->

<!-- #### Single-duration datasets -->
<!-- - SALICON -->
<!-- - CAT2000 -->
<!-- - MIT1003 -->
<!-- - MIT300 -->

<!-- ## Contents -->

<!-- #### Running code -->

<!-- We provide two ipython notebooks to demonstrate how to use our code. `train_multiduration.ipynb` walks through training and evaluating multi-duration models (like MD-SEM or SAM-MD), while `train_singleduration.ipynb` covers models that only handle one duration (SAM). Fill in the cells marked "FILL IN HERE" with the appropriate values. --> 

## Contents

#### Important source files

- `src/multiduration_models.py`: model definitions for multi-duration models. 
- `src/singleduration_models.py`: model definitions for single-duration models. 
- `src/losses_keras2.py`: loss functions used in training
- `src/data_loading.py`: helper functions to load saliency data sets
- `src/eval.py`: helper functions for evaluating models on common saliency metrics and saving predictions
- `src/util.py`: utilities for loading models and losses

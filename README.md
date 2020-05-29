# Multi-Duration Saliency Code

Source code for our paper "How Many Glances? Modeling Multi-duration Saliency".

This repo contains models and source code for predicting multi-duration saliency and its applications. 

## Downloading the CodeCharts1K Dataset

To get started, download the [CodeCharts1k multi-duration saliency dataset](http://multiduration-saliency.csail.mit.edu/data/codecharts_data.zip). Please see the `README` contained in the CodeCharts1k zip file for a detailed description of its contents.

## Performing multi-duration inference

### Inference notebook
To perform inference on a few images, we provide the `mdsem_simple_inference.ipynb` notebook, where our pretrained checkpoints on codecharts and salicon can be loaded and saliency maps can be generated from a few test images. 

To run this notebook: 
1. Clone the repository
2. Place your test images in the `images/` folder. 
3. Download the checkpoints from our website ([codecharts_checkpoint](http://multiduration-saliency.csail.mit.edu/data/mdsem_codecharts0_cameraready_weights.hdf5), [salicon-md checkpoint](http://multiduration-saliency.csail.mit.edu/data/mdsem_salicon_cameraready_weights.hdf5) and place them in a folder named `ckpt/`. 
4. Run the notebook.

### Important source files

- `src/multiduration_models.py`: model definitions for multi-duration models.
- `src/singleduration_models.py`: model definitions for single-duration models.
- `src/losses_keras2.py`: loss functions used in training
- `src/data_loading.py`: helper functions to load saliency data sets
- `src/eval.py`: helper functions for evaluating models on common saliency metrics and saving predictions
- `src/util.py`: utilities for loading models and losses

### Plotting predictions

Use the notebook `notebooks/plot_mdsem_predictions.ipynb` to plot predictions from MD-SEM.

## Multi-duration applications 

Please see the [`applications`](https://github.com/diviz-mit/multiduration-saliency/tree/master/applications) folder for the source code for our multi-duration saliency applications. 

If you want to test out the applications without running the MD-SEM model, you can download the MD-SEM predictions [here](http://multiduration-saliency.csail.mit.edu/data/mdsem_preds.zip). 

# README

This package contains tools for training a simple 2D convolutional neural network deep learning model to identify and subclassify acute intracranial hemorrhage at noncontrast CT. Training data come from the [RSNA Intracranial Hemorrhage Detection](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection) dataset on Kaggle.

## Installation

1. Make sure a version of [`mxnet`](https://mxnet.incubator.apache.org/versions/master/install/) is installed with GPU support. For example:

    ```bash
    pip install mxnet-cu92mkl
    ``` 

1. Download this repository, either with `git clone <URL>` (where `<URL>` is the git repository URL) or by clicking the download link in the git web interface.

1. Install the companion python module. From the command line:
    ```
    cd /path/to/unet_brats/
    pip install .
    ```
    Alternatively, install directly from the git repository like:

    ```bash
    pip install git+https://github.com/johncolby/rsna_heme
    ```

## Usage

Example Jupyter notebooks for training and testing/inference are included in the [notebooks](notebooks) directory.

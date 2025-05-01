
# Neural Network and Deep Learning PJ1  
Image Classification with Neural Networks

## Abstract

In this project, we implemented image classification on the MNIST handwritten digits dataset using either a Multilayer Perceptron (MLP) or a Convolutional Neural Network (CNN). To improve model performance, we explored various strategies, including:

- Number of hidden layers and weight initialization methods  
- Multiple optimizers (SGD / MomentumSGD / Adam)  
- Learning rate schedulers (Step / MultiStep / Exponential)  
- Regularization techniques (e.g., L2 regularization, Dropout)  
- Loss function selection (e.g., cross-entropy loss)  
- Data augmentation (rotation, translation, center cropping, noise)  
- Model weight visualization utilities

## Model Weights and Dataset

Model weights and dataset files can be downloaded from the following link:

[Click here to access Google Drive](https://drive.google.com/drive/folders/1abmgx2KvP7EycrMlQSdj_In8WHGEOOz-?usp=drive_link)

## Preparation

1. Set up the dataset directory structure as follows:

```
dataset/
└── MNIST/
    ├── train-images-idx3-ubyte.gz
    ├── train-labels-idx1-ubyte.gz
    ├── t10k-images-idx3-ubyte.gz
    └── t10k-labels-idx1-ubyte.gz
```

2. No additional installation is required – the project is implemented entirely using `numpy`.

## Code Overview

The project consists of seven Jupyter notebooks, each corresponding to a specific task. You can open and run any of them directly. Before running, navigate to the project directory:

```bash
cd /path/to/project
```

## Quick Training (MLP)

To train an MLP model:

```bash
python simple_train_MLP.py
```

You can modify the model architecture, optimizer, and other hyperparameters in the code.

## Quick Training (CNN)

To train a CNN model:

```bash
python simple_train_CNN.py
```

As with MLP, you can modify the CNN structure and parameters in the source code.

## Model Testing(MLP)

To evaluate a trained model:

```bash
python test_model_MLP.py
```

！！！ Don't forget to modify the address of the model weights to be tested

## Model Testing(CNN)

To evaluate a trained model:

```bash
python test_model_CNN.py
```

## Visualizing Model Weights

See `Q7.ipynb` for model weight visualization.


## Best Results Achieved
| Model   | Val Accuracy | Test Accuracy | Params |
|---------|---------|----------|--------|
| MLP     | 96.07%   | 96.22%    | 1.1M   |
| CNN     | 97.73%   | 98.04%    |  0.388 M  |

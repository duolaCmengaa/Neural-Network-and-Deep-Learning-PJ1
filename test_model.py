import mynn 
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

# model = mynn.models.Model_MLP()
conv_configs = [   
    {
        'layer_type': 'conv',
        'in_channels': 1,
        'out_channels': 8,     # 减小通道数
        'kernel_size': 3,
        'stride': 1,
        'padding': 1,
        'weight_decay': True,
        'weight_decay_lambda': 1e-4
    },
    {
        'layer_type': 'pool',
        'pooling_type': 'max',   # 从 average 改为 max
        'kernel_size': 2,
        'stride': 2
    },
    {
        'layer_type': 'conv',
        'in_channels': 8,
        'out_channels': 16,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1,
        'weight_decay': True,
        'weight_decay_lambda': 1e-4
    },
    {
        'layer_type': 'pool',
        'pooling_type': 'max',   # 从 average 改为 max
        'kernel_size': 2,
        'stride': 2
    }
]

fc_configs = [
    {"in_dim": 784, "out_dim": 10, "weight_decay": True, "weight_decay_lambda": 1e-4},
]


model = mynn.models.Model_CNN(
    conv_layer_config = conv_configs,
    fully_connected_config = fc_configs,
    activation_function = 'ReLU',
    apply_global_avg_pool = False
)

# 加载模型
model.load_model(r'.\saved_models\Q5\cnn_model2\best_model.pickle')
# model.load_model(r'.\saved_models\Q1\hidden_16_He\best_model.pickle')
model.evaling()
test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs / test_imgs.max()
test_imgs = test_imgs.reshape(-1, 1, 28, 28)
print(test_imgs.shape)
logits = model(test_imgs)
print(mynn.metric.accuracy(logits, test_labs))



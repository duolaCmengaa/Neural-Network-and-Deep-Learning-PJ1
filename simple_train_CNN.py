# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn 
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

# fixed seed for experiment
np.random.seed(309)

train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)


# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))
# save the index.
with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# normalize from [0, 255] to [0, 1]
train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()

# 在现有预处理基础上增加维度扩展
train_imgs = train_imgs.reshape(-1, 1, 28, 28)  # 转换为[N, C, H, W]格式
valid_imgs = valid_imgs.reshape(-1, 1, 28, 28)

def train_model(conv_layer_config,fully_connected_config,activation_function, apply_global_avg_pool,lr,epochs,save_dir, early_stop=10):
    # 初始化CNN模型
    model = mynn.models.Model_CNN(
        conv_layer_config = conv_layer_config,
        fully_connected_config = fully_connected_config,
        activation_function = activation_function,
        apply_global_avg_pool = apply_global_avg_pool
    )
    
    opt = mynn.optimizer.MomentGD(init_lr=lr, model=model, mu=0.9)
    scheduler = mynn.lr_scheduler.MultiStepLR(opt, milestones=[8000,50000, 75000], gamma=0.5)
    loss_fn = mynn.op.MultiCrossEntropyLoss(model=model)
    metric = mynn.metric.accuracy
    
    trainer = mynn.runner.RunnerM(
        model=model,
        optimizer=opt,
        metric=metric,
        loss_fn=loss_fn,
        batch_size=64,
        scheduler=scheduler
    )

    trainer.train(
        train_set=(train_imgs, train_labs),
        dev_set=(valid_imgs, valid_labs),
        num_epochs=epochs,
        log_epochs=1,  # 每1个epoch打印一次日志
        save_dir=save_dir,
        early_stop=early_stop  # 传递早停参数
    )


    # 绘制训练曲线
    _, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(trainer.train_loss, label='Train Loss')
    axes[0].plot(trainer.dev_loss, label='Val Loss')
    axes[0].set_title('Loss Curve')
    axes[0].legend()
    
    axes[1].plot(trainer.train_scores, label='Train Acc')
    axes[1].plot(trainer.dev_scores, label='Val Acc')
    axes[1].set_title('Accuracy Curve')
    axes[1].legend()
    
    plt.show()
    
    return trainer

conv_configs = [   
    {
        'layer_type': 'conv',
        'in_channels': 1,
        'out_channels': 8,  # 减少通道数
        'kernel_size': 3,   # 更小的卷积核
        'stride': 1,
        'padding': 1,       # 依然保持尺寸
        'weight_decay': True,
        'weight_decay_lambda': 1e-4
    },
    {
        'layer_type': 'pool',
        'pooling_type': 'average',
        'kernel_size': 2,
        'stride': 2
    },
    {
        'layer_type': 'conv',
        'in_channels': 8,
        'out_channels': 16,  # 第二层也减小
        'kernel_size': 3,
        'stride': 1,
        'padding': 1,
        'weight_decay': True,
        'weight_decay_lambda': 1e-4
    },
    {
        'layer_type': 'pool',
        'pooling_type': 'average',
        'kernel_size': 2,
        'stride': 2
    }
]
fc_configs = [
    {"in_dim": 784, "out_dim": 10, "weight_decay": True, "weight_decay_lambda": 1e-4},
]



trainer_0 = train_model(conv_configs, fc_configs, 'ReLU', False, 0.1, 100, f'./saved_models/simple_train_CNN', early_stop=7) 
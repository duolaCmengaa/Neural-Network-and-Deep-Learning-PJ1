# you may do your own hyperparameter search job here.

import mynn 
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd
import time
from datetime import datetime
import json
import argparse
import itertools


def load_data(model_type="MLP"):
    print("Loading dataset...")
    np.random.seed(309)

    train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
    train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'
    test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
    test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'
    
    with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
    with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)
    
    with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
    with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)
    
    idx = np.random.permutation(np.arange(train_labs.shape[0]))
    
    train_imgs = train_imgs[idx]
    train_labs = train_labs[idx]
    valid_imgs = train_imgs[:10000]
    valid_labs = train_labs[:10000]
    train_imgs = train_imgs[10000:]
    train_labs = train_labs[10000:]
    
    train_imgs = train_imgs / train_imgs.max()
    valid_imgs = valid_imgs / valid_imgs.max()
    test_imgs = test_imgs / test_imgs.max()
    
    print("train_imgs shape:", train_imgs.shape)
    print("train_labs shape:", train_labs.shape)

    if model_type == "CNN":
        train_imgs = train_imgs.reshape(-1, 1, 28, 28)
        valid_imgs = valid_imgs.reshape(-1, 1, 28, 28)
        test_imgs = test_imgs.reshape(-1, 1, 28, 28)
    
    return train_imgs, train_labs, valid_imgs, valid_labs, test_imgs, test_labs

class GradientChecker:
    """
    完整的梯度检查工具类
    可以用于任何实现了forward和backward方法的神经网络模型
    """
    
    def __init__(self, model, loss_layer, epsilon=1e-7):
        """
        初始化梯度检查器
        
        参数:
            model: 要检查的神经网络模型
            loss_layer: 损失函数层(需实现forward和backward)
            epsilon: 数值梯度计算的小扰动值(默认1e-7)
        """
        self.model = model
        self.loss_layer = loss_layer
        self.epsilon = epsilon
    
    def check_layer(self, layer, input_data, label, param_name='W', verbose=True):
        """
        检查指定层的指定参数的梯度
        
        参数:
            layer: 要检查的层
            input_data: 输入数据(单个样本或小批量)
            label: 对应的标签
            param_name: 要检查的参数名('W'或'b')
            verbose: 是否打印详细信息
            
        返回:
            (最大绝对误差, 最大相对误差)
        """
        # 首先执行一次正常的前向和反向传播
        output = self.model.forward(input_data)
        loss = self.loss_layer(output, label)
        self.model.backward(self.loss_layer.grads)
        
        # 获取解析梯度
        analytic_grad = layer.grads[param_name]
        
        # 初始化数值梯度
        param = layer.params[param_name]
        numerical_grad = np.zeros_like(param)
        
        # 遍历参数的每个元素计算数值梯度
        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            original_value = param[idx]
            
            # 计算f(θ + ε)
            param[idx] = original_value + self.epsilon
            loss_plus = self.loss_layer(self.model.forward(input_data), label)
            
            # 计算f(θ - ε)
            param[idx] = original_value - self.epsilon
            loss_minus = self.loss_layer(self.model.forward(input_data), label)
            
            # 恢复原始值
            param[idx] = original_value
            
            # 计算中心差分梯度
            numerical_grad[idx] = (loss_plus - loss_minus) / (2 * self.epsilon)
            
            it.iternext()
        
        # 计算误差
        abs_diff = np.abs(numerical_grad - analytic_grad)
        rel_diff = abs_diff / (np.abs(numerical_grad) + np.abs(analytic_grad) + 1e-10)
        
        max_abs_diff = np.max(abs_diff)
        max_rel_diff = np.max(rel_diff)
        
        if verbose:
            print(f"\n检查层: {layer.__class__.__name__} 参数: {param_name}")
            print(f"数值梯度范围: [{numerical_grad.min():.6f}, {numerical_grad.max():.6f}]")
            print(f"解析梯度范围: [{analytic_grad.min():.6f}, {analytic_grad.max():.6f}]")
            print(f"最大绝对误差: {max_abs_diff:.6e}")
            print(f"最大相对误差: {max_rel_diff:.6e}")
            
            if max_rel_diff < 1e-5:
                print(" 梯度检查通过")
            elif max_rel_diff < 1e-3:
                print(" 警告: 梯度差异稍大，可能需要检查")
            else:
                print(" 错误: 梯度差异过大，实现可能有误")
                
            print("="*60)
        
        return max_abs_diff, max_rel_diff
    
    def check_all_layers(self, input_data, label, verbose=True):
        """
        检查模型中所有可训练层的梯度
        """
        results = {}
        
        # 首先执行一次正常的前向和反向传播
        output = self.model.forward(input_data)
        loss = self.loss_layer(output, label)  # 这会自动计算并存储grads
        
        # 现在loss_layer.grads应该已经被计算
        if self.loss_layer.grads is None:
            raise ValueError("Loss layer gradients not computed!")
        
        # 执行反向传播
        self.model.backward(self.loss_layer.grads)
        
        # 检查每一层
        for i, layer in enumerate(self.model.layers):
            if not hasattr(layer, 'params') or not layer.optimizable:
                continue
                
            layer_results = {}
            
            # 检查权重参数W
            if 'W' in layer.params:
                max_abs, max_rel = self.check_layer(
                    layer, input_data, label, 'W', verbose)
                layer_results['W'] = {'max_abs': max_abs, 'max_rel': max_rel}
            
            # 检查偏置参数b(如果有)
            if 'b' in layer.params:
                max_abs, max_rel = self.check_layer(
                    layer, input_data, label, 'b', verbose)
                layer_results['b'] = {'max_abs': max_abs, 'max_rel': max_rel}
            
            results[f"Layer_{i}_{layer.__class__.__name__}"] = layer_results
        
        return results

def create_cnn_model(conv_layer_config, fully_connected_config, activation_function='ReLU', apply_global_avg_pool=True):
    return mynn.models.Model_CNN(conv_layer_config, fully_connected_config, activation_function, apply_global_avg_pool)

# ==================== 使用示例 ====================

if __name__ == "__main__":
    config =  {
                "name": "cnn_baseline",
                "conv_layer_config": [
                    {"layer_type": "conv", "in_channels": 1, "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 0}, 
                    {"layer_type": "conv", "in_channels": 32, "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 0},
                ],
                "fully_connected_config": [
                    {"in_dim": 64*24*24, "out_dim": 128, "weight_decay": True, "weight_decay_lambda": 1e-4},
                    {"in_dim": 128, "out_dim": 10, "weight_decay": True, "weight_decay_lambda": 1e-4}
                ],
                "activation_function": "ReLU",
                "apply_global_avg_pool": False,
                "optimizer": "SGD",
                "learning_rate": 0.1,
                "scheduler": "MultiStepLR",
                "scheduler_params": {"milestones":[10, 20, 40, 80], "gamma":0.5},
                "batch_size": 1,
                "num_epochs": 5,
                "log_iters": 100,
                "augmentation_config": {"if_use_augment":True, "rotation_prob":0.4}
            }
        
    # 1. 创建模型和损失函数
    model = create_cnn_model(
            conv_layer_config=config.get("conv_layer_config"),
            fully_connected_config=config.get("fully_connected_config"),
            activation_function=config.get("activation_function", "ReLU"),
            apply_global_avg_pool=config.get("apply_global_avg_pool", True)
        )    
    loss_layer = mynn.op.MultiCrossEntropyLoss(model=model, max_classes=10)
    
    # 2. 创建梯度检查器
    checker = GradientChecker(model, loss_layer, epsilon=1e-7)
    test_input, test_label, valid_imgs, valid_labs, test_imgs, test_labs = load_data("CNN")
    test_input = test_input[:10]
    test_label = test_label[:10]
    
    # 4. 执行梯度检查
    print("开始梯度检查...")
    results = checker.check_all_layers(test_input, test_label)
    
    # 5. 分析结果
    print("\n梯度检查汇总:")
    for layer_name, params in results.items():
        print(f"\n{layer_name}:")
        for param_name, errors in params.items():
            print(f"  {param_name}: 最大相对误差 = {errors['max_rel']:.2e}")
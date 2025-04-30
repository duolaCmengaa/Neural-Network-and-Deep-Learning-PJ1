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


def create_mlp_model(train_imgs, hidden_layers, activation='ReLU', weight_decay=None):
    size_list = [train_imgs.shape[-1]] + hidden_layers + [10]
    if weight_decay is None: 
        weight_decay = [1e-4] * (len(size_list) - 1)
    return mynn.models.Model_MLP(size_list, activation, weight_decay)


def create_cnn_model(conv_layer_config, fully_connected_config, activation_function='ReLU', apply_global_avg_pool=True):
    return mynn.models.Model_CNN(conv_layer_config, fully_connected_config, activation_function, apply_global_avg_pool)


def create_optimizer(optimizer_name, init_lr, model, mu=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8):
    if optimizer_name == "SGD":
        return mynn.optimizer.SGD(init_lr=init_lr, model=model)
    elif optimizer_name == "MomentGD":
        return mynn.optimizer.MomentGD(init_lr=init_lr, model=model, mu=mu)
    elif optimizer_name == "Adam":
        return mynn.optimizer.Adam(init_lr=init_lr, model=model, beta1=beta1, beta2=beta2, epsilon=epsilon)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def create_scheduler(scheduler_name, optimizer, **kwargs):
    if scheduler_name == "StepLR":
        step_size = kwargs.get("step_size", 30)
        gamma = kwargs.get("gamma", 0.1)
        return mynn.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == "MultiStepLR":
        milestones = kwargs.get("milestones", [8000, 80000])
        gamma = kwargs.get("gamma", 0.5)
        return mynn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=gamma)
    elif scheduler_name == "ExponentialLR":
        gamma = kwargs.get("gamma", 0.95)
        return mynn.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def run_experiment(config, model_type="MLP", base_dir="ablation_results"):
    print(f"\n\n{'='*50}")
    print(f"Running experiment: {config.get('name', 'experiment')}")
    print(f"{'='*50}\n")
    
    train_imgs, train_labs, valid_imgs, valid_labs, test_imgs, test_labs = load_data(model_type)
    
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"{model_type}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    experiment_name = config.get("name", f"experiment_{timestamp}")
    experiment_dir = os.path.join(run_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    with open(os.path.join(experiment_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    if model_type == "MLP":
        model = create_mlp_model(
            train_imgs,
            hidden_layers=config.get("hidden_layers", [128]),
            activation=config.get("activation", "ReLU"),
            weight_decay=config.get("weight_decay", None)
        )
    else:
        model = create_cnn_model(
            conv_layer_config=config.get("conv_layer_config"),
            fully_connected_config=config.get("fully_connected_config"),
            activation_function=config.get("activation_function", "ReLU"),
            apply_global_avg_pool=config.get("apply_global_avg_pool", True)
        )
    
    optimizer = create_optimizer(
        optimizer_name=config.get("optimizer", "SGD"),
        init_lr=config.get("learning_rate", 0.01),
        model=model,
        mu=config.get("momentum", 0.9),
        beta1=config.get("beta1", 0.9),
        beta2=config.get("beta2", 0.999),
        epsilon=config.get("epsilon", 1e-8)
    )
    
    scheduler = create_scheduler(
        scheduler_name=config.get("scheduler", "MultiStepLR"),
        optimizer=optimizer,
        **config.get("scheduler_params", {})
    )
    
    loss_fn = mynn.op.MultiCrossEntropyLoss(model=model, max_classes=10)
    
    augmentation_config = config.get("augmentation_config", {"enabled": True})
    runner = mynn.runner.RunnerM(
        model=model,
        optimizer=optimizer,
        metric=mynn.metric.accuracy,
        loss_fn=loss_fn,
        batch_size=config.get("batch_size", 64),
        scheduler=scheduler,
        augmentation_config=augmentation_config
    )
    
    start_time = time.time()
    
    runner.train(
        train_set=(train_imgs, train_labs),
        dev_set=(valid_imgs, valid_labs),
        num_epochs=config.get("num_epochs", 100),
        log_epochs=config.get("log_epochs", 1),
        save_dir=experiment_dir,
        early_stop=config.get("early_stop", 10),
        )
    
    training_time = time.time() - start_time
    
    with open(os.path.join(experiment_dir, "runner_data.pkl"), "wb") as f:
        pickle.dump({
            "train_scores": runner.train_scores,
            "dev_scores": runner.dev_scores,
            "train_loss": runner.train_loss,
            "dev_loss": runner.dev_loss,
        }, f)
    
    test_accuracy, test_loss = runner.evaluate([test_imgs, test_labs])

    
    result = {
        "name": experiment_name,
        "config": config,
        "training_time": training_time,
        "best_validation_accuracy": runner.best_score,
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "final_train_accuracy": runner.train_scores[-1],
        "final_train_loss": runner.train_loss[-1]
    }
    
    results.append(result)
    
    print(f"\nExperiment '{experiment_name}' completed:")
    print(f"  Training time: {training_time:.2f} seconds")
    print(f"  Best validation accuracy: {runner.best_score:.4f}")
    print(f"  Test accuracy: {test_accuracy:.4f}")
    print(f"  Test loss: {test_loss:.4f}")
    
    return result, results


def run_mlp_ablation(model_type="MLP", configs=None, output_dir="ablation_results"):
    """
    Run an ablation study for MLP models.

    Args:
        model_type (str): Model type, default "MLP".
        configs (list, optional): List of configurations to run.
        output_dir (str): Directory to save the results.
    """
    if configs is None:
        configs = [
            {
                "name": "mlp_baseline",
                "hidden_layers": [128],
                "activation": "ReLU",
                "optimizer": "SGD",
                "learning_rate": 0.1,
                "scheduler": "ExponentialLR",
                "scheduler_params": {"gamma": 1.0},
                "batch_size": 32,
                "num_epochs": 100,
                "log_iters": 100,
                "augmentation_config": {
                        "if_use_augment": True,
                        "translation_prob": 0,  
                        "rotation_prob": 0.3,
                        "resize_prob": 0,
                        "noise_prob": 0.3,                   
                        "max_shift": 5, 
                        "max_rotation": 15, 
                        "noise_std": 0.05,
                        "image_shape": (28, 28)
                        }
            },
            {
                "name": "mlp_wd_1e-4",
                "hidden_layers": [128],
                "activation": "ReLU",
                "weight_decay": [1e-4, 1e-4],  
                "optimizer": "SGD",
                "learning_rate": 0.1,
                "scheduler": "ExponentialLR",
                "scheduler_params": {"gamma": 1.0},
                "batch_size": 32,
                "num_epochs": 2,
                "log_iters": 100
            }
        ]
    
    all_results = []
    for config in configs:
        result, _ = run_experiment(config, model_type=model_type, base_dir=output_dir)
        all_results.append(result)

    # Save summary of all results
    results_df = pd.DataFrame(all_results)
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, f"mlp_ablation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"), index=False)

    print("\nAll MLP ablation experiments completed and results saved!")


def run_cnn_ablation(model_type="CNN", configs=None, output_dir="ablation_results"):
    """
    Run an ablation study for CNN models.

    Args:
        configs (list): List of configurations to run.
        output_dir (str): Directory to save the results.
    """
    if configs is None:
        # Default configurations for CNN ablation study
        # configs = [
        #     {
        #         "name": "cnn_baseline",
        #         "conv_layer_config": [
        #             {"layer_type": "conv", "in_channels": 1, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1},
        #             {"layer_type": "pool", "pooling_type": "max", "kernel_size": 2}
        #         ],
        #         "fully_connected_config": [
        #             {"in_dim": 16*14*14, "out_dim": 64, "weight_decay": True, "weight_decay_lambda": 1e-4},
        #             {"in_dim": 64, "out_dim": 10, "weight_decay": True, "weight_decay_lambda": 1e-4}
        #         ],
        #         "activation_function": "ReLU",
        #         "apply_global_avg_pool": False,
        #         "optimizer": "SGD",
        #         "learning_rate": 0.1,
        #         "scheduler": "ExponentialLR",
        #         "scheduler_params": {"gamma": 1.0},
        #         "batch_size": 32,
        #         "num_epochs": 5,
        #         "log_iters": 100
        #     }
        # ]
        # configs = [
        #     {
        #         "name": "cnn_baseline",
        #         "conv_layer_config": [
        #                 {"layer_type": "conv", "in_channels": 1, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1},
        #                 {"layer_type": "pool", "pooling_type": "max", "kernel_size": 2},
        #                 {"layer_type": "conv", "in_channels": 16, "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1},
        #                 {"layer_type": "pool", "pooling_type": "max", "kernel_size": 2},
        #                 {"layer_type": "conv", "in_channels": 32, "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1}
        #         ],
        #         "fully_connected_config": [
        #             {"in_dim": 64*7*7, "out_dim": 10, "weight_decay": True, "weight_decay_lambda": 1e-4},
        #         ],
        #         "activation_function": "ReLU",
        #         "apply_global_avg_pool": False,
        #         "optimizer": "SGD",
        #         "learning_rate": 0.1,
        #         "scheduler": "MultiStepLR",
        #         "scheduler_params": {"milestones":[10, 20, 40, 80], "gamma":0.5},
        #         "batch_size": 32,
        #         "num_epochs": 5,
        #         "log_iters": 100
        #     }
        # ]
        configs = [{ 
            "name": "cnn_baseline",
            "conv_layer_config": [
                {
                    'layer_type': 'conv',
                    'in_channels': 1,          # MNIST灰度图通道数
                    'out_channels': 16,       
                    'kernel_size': 5,
                    'stride': 1,
                    'padding': 0,
                    'weight_decay': True,      # 可选：添加正则化
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
                    'in_channels': 16,
                    'out_channels': 32,       
                    'kernel_size': 5,
                    'stride': 1,
                    'padding': 0,
                    'weight_decay': True,     
                    'weight_decay_lambda': 1e-4
                },
                {
                    'layer_type': 'pool',
                    'pooling_type': 'average',
                    'kernel_size': 2,
                    'stride': 2
                }
            ],
            "fully_connected_config": [
                {"in_dim": 512, "out_dim": 10, "weight_decay": True, "weight_decay_lambda": 1e-4}
            ],
            "activation_function": "ReLU",
            "apply_global_avg_pool": False,
            "optimizer": "SGD",
            "learning_rate": 0.1,
            "scheduler": "MultiStepLR",
            "scheduler_params": {"milestones": [7500, 75000], "gamma": 0.5},
            "batch_size": 64,
            "num_epochs": 100,
            "log_epochs": 1,
            "augmentation_config": {"if_use_augment": False}
        }]

    all_results = []
    
    # Run each configuration
    for config in configs:
        result, _ = run_experiment(config, model_type="CNN", base_dir=output_dir)
        all_results.append(result)
    
    # Save summary of all results
    results_df = pd.DataFrame(all_results)
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, f"cnn_ablation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"), index=False)

    print("\nAll experiments completed and results saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation studies for neural network models.")
    parser.add_argument("--model", type=str, choices=["MLP", "CNN"], default="CNN", help="Type of model to use")
    parser.add_argument("--output_dir", type=str, default="ablation_results", help="Directory to save results")
    args = parser.parse_args()

    run_cnn_ablation(model_type=args.model, output_dir=args.output_dir)

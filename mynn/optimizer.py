from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] = layer.params[key] - self.init_lr * layer.grads[key]


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu):
        super().__init__(init_lr, model)
        self.mu = mu
        self.velocities = {}  # Momentum dictionary

        for i, layer in enumerate(self.model.layers):
            if layer.optimizable:
                self.velocities[i] = {}
                for key in layer.params.keys():
                    self.velocities[i][key] = np.zeros_like(layer.params[key])
                
    def step(self):
        for i, layer in enumerate(self.model.layers):
            if layer.optimizable:
                for key in layer.params.keys():
                    v = self.velocities[i][key]
                    grad = layer.grads[key]

                    # Update velocity
                    v = self.mu * v - self.init_lr * grad
                    self.velocities[i][key] = v

                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)

                    # Update parameters
                    layer.params[key] += v

class Adam(Optimizer): 
    def __init__(self, init_lr, model, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(init_lr, model)
        self.beta1 = beta1  # Momentum decay factor
        self.beta2 = beta2  # Squared gradient decay factor
        self.epsilon = epsilon  # Small constant to avoid division by zero
        self.t = 0  # Time step
        self.m = {}  # First moment estimate (momentum)
        self.v = {}  # Second moment estimate (squared gradients)

        for i, layer in enumerate(self.model.layers):
            if layer.optimizable:
                self.m[i] = {}
                self.v[i] = {}
                for key in layer.params.keys():
                    self.m[i][key] = np.zeros_like(layer.params[key])
                    self.v[i][key] = np.zeros_like(layer.params[key])

    def step(self):
        self.t += 1
        for i, layer in enumerate(self.model.layers):
            if layer.optimizable:
                for key in layer.params.keys():
                    grad = layer.grads[key]

                    # Update first moment estimate (momentum)
                    self.m[i][key] = self.beta1 * self.m[i][key] + (1 - self.beta1) * grad
                    # Update second moment estimate (squared gradients)
                    self.v[i][key] = self.beta2 * self.v[i][key] + (1 - self.beta2) * (grad ** 2)

                    # Bias correction
                    m_hat = self.m[i][key] / (1 - self.beta1 ** self.t)
                    v_hat = self.v[i][key] / (1 - self.beta2 ** self.t)

                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)

                    # Update parameters
                    layer.params[key] -= self.init_lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

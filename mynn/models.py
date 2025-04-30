from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None, dropout_p = None, initialize_method='Normal'):
        self.size_list = size_list
        self.act_func = act_func
        self.dropout_p = dropout_p
        self.lambda_list = lambda_list
        self.initialize_method = initialize_method

        if size_list is not None and act_func is not None:
            self.layers = []
            nums_hidden_layers = len(size_list) - 2
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1], initialize_method=self.initialize_method)
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                self.layers.append(layer)
                
                if i < len(size_list) - 2:
                    if act_func == 'Logistic':
                        raise NotImplementedError
                    elif act_func == 'ReLU':
                        layer_f = ReLU()   
                        self.layers.append(layer_f)
                    if dropout_p is not None and i < len(dropout_p):
                        layer_d = Dropout(p=dropout_p[i])
                        self.layers.append(layer_d)

    def training(self):
        """
        Set the model to training mode.
        """
        for layer in self.layers:
            layer.train()

    def evaling(self):
        """
        Set the model to evaluation mode.
        """
        for layer in self.layers:
            layer.eval()

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def get_l2_loss(self):
        l2_loss = 0.0
        for layer in self.layers:
            if isinstance(layer, Linear) and layer.weight_decay:
                W = layer.params['W']
                l2_loss += 0.5 * layer.weight_decay_lambda * (W ** 2).sum()
        return l2_loss
    
    def load_model(self, param_path):
        with open(param_path, 'rb') as f:
            param_list = pickle.load(f)

        self.size_list = param_list[0]
        self.act_func = param_list[1]
        self.layers = []

        in_out_sizes = list(zip(self.size_list[:-1], self.size_list[1:]))
        param_idx = 2
        size_idx = 0  # 对应 Linear 层的 (in_dim, out_dim) 使用索引

        while param_idx < len(param_list):
            param_dict = param_list[param_idx]

            if 'W' in param_dict:
                # Linear 层
                in_dim, out_dim = in_out_sizes[size_idx]
                size_idx += 1
                layer = Linear(in_dim=in_dim, out_dim=out_dim)
                layer.W = param_dict['W']
                layer.b = param_dict['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_dict['weight_decay']
                layer.weight_decay_lambda = param_dict['lambda']
                self.layers.append(layer)
                param_idx += 1

                # 激活函数
                if self.act_func == 'Logistic':
                    raise NotImplementedError("Logistic not implemented")
                elif self.act_func == 'ReLU' and size_idx < len(self.size_list) - 1:
                    self.layers.append(ReLU())

            elif 'p' in param_dict:
                # Dropout 层
                self.layers.append(Dropout(p=param_dict['p']))
                param_idx += 1

            else:
                raise ValueError(f"Unknown layer parameters at index {param_idx}: {param_dict}")



    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for i in range(len(self.layers)):
            if isinstance(self.layers[i], Linear):
                layer = self.layers[i]
                param_list.append({
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda
                })
            if isinstance(self.layers[i], Dropout):
                param_list.append({
                    'p': self.layers[i].p
                })
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self, conv_layer_config=None, fully_connected_config=None, activation_function='ReLU', apply_global_avg_pool=False, initialize_method = 'He'):
        """
        Initializes a Convolutional Neural Network (CNN) model with configurable layers and activation functions.

        Args:
            conv_layer_config: List of dictionaries specifying the configuration for convolution and pooling layers.
            fully_connected_config: List of tuples (input_dim, output_dim) for fully connected layers.
            activation_function: The activation function to be applied ('ReLU' by default).
            apply_global_avg_pool: Boolean flag indicating whether to apply global average pooling before the fully connected layers.
            dropout_config: Dictionary specifying the dropout layers' parameters (optional).
        """
        super().__init__()
        self.layers = []
        self.optimizable = True
        self.initialized = False
        self.conv_layer_config = conv_layer_config
        self.fully_connected_config = fully_connected_config
        self.activation_function = activation_function
        self.apply_global_avg_pool = apply_global_avg_pool
        self.initialize_method = initialize_method

        if conv_layer_config and fully_connected_config:
            self.initialized = True
            for layer_config in conv_layer_config:
                layer_type = layer_config.get('layer_type', 'conv')

                if layer_type == 'conv':
                    in_channels = layer_config['in_channels']
                    out_channels = layer_config['out_channels']
                    kernel_size = layer_config['kernel_size']
                    stride = layer_config.get('stride', 1)
                    padding = layer_config.get('padding', 0)
                    weight_decay = layer_config.get('weight_decay', False)
                    weight_decay_lambda = layer_config.get('weight_decay_lambda', 1e-8)

                    conv_layer = conv2D(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        weight_decay=weight_decay,
                        weight_decay_lambda=weight_decay_lambda,
                        initialize_method=self.initialize_method
                    )
                    self.layers.append(conv_layer)

                    if activation_function == 'ReLU':
                        self.layers.append(ReLU())
                    else:
                        raise NotImplementedError(f"No activation function {activation_function}")

                elif layer_type == 'pool':
                    pooling_type = layer_config.get('pooling_type', 'average')
                    kernel_size = layer_config['kernel_size']
                    stride = layer_config.get('stride', kernel_size)
                    padding = layer_config.get('padding', 0)

                    pooling_layer = Pooling(
                        mode=pooling_type,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding
                    )
                    self.layers.append(pooling_layer)
                elif layer_type == 'dropout':
                    p = layer_config.get('p', 0.5)  # default dropout probability p =0.5
                    dropout_layer = Dropout(p)
                    self.layers.append(dropout_layer)

            if apply_global_avg_pool:
                self.layers.append(GlobalAvgPooling())
            else:
                self.layers.append(Flatten())
            #0
            for i, fc_cfg in enumerate(fully_connected_config):
                in_dim = fc_cfg['in_dim']
                out_dim = fc_cfg['out_dim']
                weight_decay = fc_cfg.get('weight_decay', False)
                weight_decay_lambda = fc_cfg.get('weight_decay_lambda', 1e-8)
                
                fc_layer = Linear(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    weight_decay=weight_decay,
                    weight_decay_lambda=weight_decay_lambda,
                    initialize_method=self.initialize_method
                )

                self.layers.append(fc_layer)
                if i < len(fully_connected_config) - 1 and activation_function == 'ReLU':
                    self.layers.append(ReLU())

    def __call__(self, input_data):
        return self.forward(input_data)
    
    def training(self):
        """
        Set the model to training mode.
        """
        for layer in self.layers:
            layer.train()

    def evaling(self):
        """
        Set the model to evaluation mode.
        """
        for layer in self.layers:
            layer.eval()
            
    def forward(self, input_data):
        """
        The forward pass through the CNN network.

        Args:
            input_data: The input data with shape [batch_size, channels, height, width].

        Returns:
            The output after passing through the entire network.
        """
        if not self.initialized:
            raise ValueError("Model doesn't initialize. Please either load a pre-trained model or initialize a new model.")

        output_data = input_data
        for layer in self.layers:
            output_data = layer(output_data)

        return output_data

    def backward(self, gradient_from_loss):
        """
        The backward pass through the network for backpropagation.

        Args:
            gradient_from_loss: The gradient received from the loss function.

        Returns:
            The gradient to be passed to the previous layers.
        """
        gradient = gradient_from_loss
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

        return gradient

    def load_model(self, model_file_path):
        """
        Load the weights and configuration of a pre-trained model.

        Args:
            model_file_path: The path to the file containing the model's saved weights and configuration.
        """
        with open(model_file_path, 'rb') as file:
            model_params = pickle.load(file)

        conv_layer_config = model_params[0]
        fully_connected_config = model_params[1]
        activation_function = model_params[2]
        apply_global_avg_pool = model_params[3]

        self.__init__(
            conv_layer_config=conv_layer_config,
            fully_connected_config=fully_connected_config,
            activation_function=activation_function,
            apply_global_avg_pool=apply_global_avg_pool
        )

        param_idx = 4  # 从 model_params[4] 开始是参数字典
        for layer in self.layers:
            if layer.optimizable:
                params = model_params[param_idx]
                print(f"Loading parameters for layer {param_idx}: W shape: {params['W'].shape}, b shape: {params['b'].shape}")
                print(f"meaning: {params['W'].mean()}, std: {params['W'].std()}")
                layer.params['W'] = params['W']
                layer.params['b'] = params['b']
                layer.weight_decay = params['weight_decay']
                layer.weight_decay_lambda = params['weight_decay_lambda']
                param_idx += 1

        self.initialized = True
        return self

    def save_model(self, save_file_path):
        """
        Save the current model's weights and configuration.

        Args:
            save_file_path: The path where the model will be saved.
        """
        model_params = [self.conv_layer_config, self.fully_connected_config, self.activation_function, self.apply_global_avg_pool]

        for layer in self.layers:
            if layer.optimizable:
                model_params.append({
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'weight_decay_lambda': layer.weight_decay_lambda
                })

        with open(save_file_path, 'wb') as file:
            pickle.dump(model_params, file)

    # def save_model(model, filepath):
    #     """
    #     Save the CNN model to a file.
        
    #     Args:
    #         model: The Model_CNN instance to be saved
    #         filepath: Path to the file where the model will be saved
    #     """
    #     # Collect all parameters from the model
    #     model_data = {
    #         'conv_layer_config': model.conv_layer_config,
    #         'fully_connected_config': model.fully_connected_config,
    #         'activation_function': model.activation_function,
    #         'apply_global_avg_pool': model.apply_global_avg_pool,
    #         'initialize_method': model.initialize_method,
    #         'params': []
    #     }
        
    #     # Save parameters from each layer
    #     for layer in model.layers:
    #         if hasattr(layer, 'params'):
    #             model_data['params'].append(layer.params)
    #         else:
    #             model_data['params'].append(None)
        
    #     # Save to file
    #     with open(filepath, 'wb') as f:
    #         pickle.dump(model_data, f)

    # def load_model(self, filepath):
    #     """
    #     Load a CNN model from a file.
        
    #     Args:
    #         filepath: Path to the file containing the saved model
        
    #     Returns:
    #         The loaded Model_CNN instance
    #     """
    #     with open(filepath, 'rb') as f:
    #         model_data = pickle.load(f)
        
    #     # Create a new model with the saved configuration
    #     model = Model_CNN(
    #         conv_layer_config=model_data['conv_layer_config'],
    #         fully_connected_config=model_data['fully_connected_config'],
    #         activation_function=model_data['activation_function'],
    #         apply_global_avg_pool=model_data['apply_global_avg_pool'],
    #         initialize_method=model_data['initialize_method']
    #     )
        
    #     # Restore parameters for each layer
    #     for layer, params in zip(model.layers, model_data['params']):
    #         if params is not None and hasattr(layer, 'params'):
    #             layer.params = params
        
    #     return model

    def print_weights_statistics(model):
        """
        打印模型中每一层的权重均值和方差
        """
        print("\n模型每层权重参数的均值和方差:")
        for idx, layer in enumerate(model.layers):
            if hasattr(layer, 'params') and 'W' in layer.params:
                w = layer.params['W']
                mean_val = w.mean()
                std_val = w.std()
                print(f"Layer {idx} - {type(layer).__name__} - W: mean={mean_val:.6f}, std={std_val:.6f}")

    def summary(self, input_shape):
        """
        Print the model structure and the output shape of each layer.
        
        Args:
            input_shape: tuple, the shape of input data (e.g., (batch_size, channels, height, width))
        """
        print(f"{'Layer':<30} {'Output Shape':<30}")
        print("=" * 60)

        x = np.random.randn(*input_shape)  # Dummy data
        for layer in self.layers:
            x = layer(x)
            print(f"{layer.__class__.__name__:<30} {str(x.shape):<30}")
        print("=" * 60)
        print(f"Final output shape: {x.shape}")

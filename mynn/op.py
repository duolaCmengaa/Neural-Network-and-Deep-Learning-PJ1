from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True

    def train(self):
        """
        Set the layer to training mode.
        """
        self.train_mode = True
    
    def eval(self):
        """
        Set the layer to evaluation mode.
        """
        self.train_mode = False
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method='He', weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()

        if initialize_method == 'He':
            # He initialization (Kaiming initialization) with normal distribution
            scale = np.sqrt(2.0 / in_dim)
            initializer = lambda size: np.random.normal(loc=0.0, scale=scale, size=size)
        elif initialize_method == 'Xavier':
            # Xavier initialization (Glorot initialization)
            scale = np.sqrt(2.0 / (in_dim + out_dim))
            initializer = lambda size: np.random.normal(loc=0.0, scale=scale, size=size)
        elif initialize_method == 'Normal':
            # Standard normal distribution
            initializer = lambda size: np.random.normal(loc=0.0, scale=1.0, size=size)
        else:
            raise ValueError("Invalid initialization method. Choose 'He', 'Xavier', or 'Normal'.")

        self.W = initializer(size=(in_dim, out_dim))
        self.b = np.zeros((1, out_dim))  # Bias initialized to 0

        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        output = np.matmul(X, self.W) + self.b
        return output

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        batch_size = self.input.shape[0]
        self.grads['W'] = np.matmul(self.input.T, grad) / batch_size
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True) / batch_size

        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W

        output = np.matmul(grad, self.W.T)
        return output
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method='He', weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding



        # Initialization
        if initialize_method == 'He':
            scale = np.sqrt(2.0 / (in_channels * self.kernel_size[0] * self.kernel_size[1]))
            initializer = lambda size: np.random.normal(loc=0.0, scale=scale, size=size)
        elif initialize_method == 'Xavier':
            scale = np.sqrt(2.0 / ((in_channels + out_channels) * self.kernel_size[0] * self.kernel_size[1]))
            initializer = lambda size: np.random.normal(loc=0.0, scale=scale, size=size)
        elif initialize_method == 'Normal':
            initializer = lambda size: np.random.normal(loc=0.0, scale=1.0, size=size)
        else:
            raise ValueError("Invalid initialization method. Choose from 'He', 'Xavier', 'Normal'.")

        # Parameters
        self.params = {
            'W': initializer(size=(out_channels, in_channels, *self.kernel_size)),
            'b': np.zeros(out_channels)
        }
        self.grads = {'W': None, 'b': None}
        self.input = None
        self.padded_img = None
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
        # self.X_col = None

    # Use property manage parameters
    @property
    def W(self):
        return self.params['W']

    @W.setter
    def W(self, value):
        self.params['W'] = value

    @property
    def b(self):
        return self.params['b']

    @b.setter
    def b(self, value):
        self.params['b'] = value


    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """
        self.input = X  #  [B, C, H, W]
        B, C, H, W = X.shape
        KH, KW = self.kernel_size
        stride, pad = self.stride, self.padding

        # ?
        OH = (H + 2 * pad - KH) // stride + 1
        OW = (W + 2 * pad - KW) // stride + 1

        if self.padding > 0:
            # Padding the input image
            self.padded_img = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
        else:
            self.padded_img = X

        # After im2col shape: [B * OH * OW, C * KH * KW]
        X_col = self.im2col(self.padded_img, KH, KW, stride, 0)
        W_col = self.W.reshape(self.out_channels, -1)
        # change
        out = np.matmul(X_col, W_col.T) + self.b
        out = out.reshape(B, OH, OW, self.out_channels).transpose(0, 3, 1, 2)  # [B, OC, OH, OW]
        return out

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """

        if self.grads['W'] is None:
            self.grads['W'] = np.zeros_like(self.W)
        if self.grads['b'] is None:
            self.grads['b'] = np.zeros_like(self.b)

        # B, OC, OH, OW = grads.shape
        B = self.input.shape[0]
        OC = self.out_channels

        KH, KW = self.kernel_size
        stride, pad = self.stride, self.padding
        batch_size = B

        grads_reshaped = grads.transpose(0, 2, 3, 1).reshape(-1, OC)

        X_col = self.im2col(self.padded_img, KH, KW, stride, 0)


        self.grads['W'] = np.matmul(grads_reshaped.T, X_col).reshape(self.W.shape) / batch_size
        self.grads['b'] = np.sum(grads_reshaped, axis=0)/ batch_size

        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W

        W_col = self.W.reshape(OC, -1)
        dX_col = np.matmul(grads_reshaped, W_col)
        dX_padded = self.col2im(dX_col, self.input.shape, KH, KW, stride, pad)

        return dX_padded
    
    def im2col(self, X, KH, KW, stride, padding):
        B, C, H, W = X.shape

        out_H = (H + 2 * padding - KH) // stride + 1
        out_W = (W + 2 * padding - KW) // stride + 1

        X_padded = np.pad(X, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

        cols = np.zeros((B, C, KH, KW, out_H, out_W))
        for y in range(KH):
            y_max = y + stride * out_H
            for x in range(KW):
                x_max = x + stride * out_W
                cols[:, :, y, x, :, :] = X_padded[:, :, y:y_max:stride, x:x_max:stride]

        cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(B * out_H * out_W, -1)
        return cols

    def col2im(self, cols, X_shape, KH, KW, stride, padding):
        B, C, H, W = X_shape
        out_H = (H + 2 * padding - KH) // stride + 1
        out_W = (W + 2 * padding - KW) // stride + 1

        cols_reshaped = cols.reshape(B, out_H, out_W, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)
        X_padded = np.zeros((B, C, H + 2 * padding + stride - 1, W + 2 * padding + stride - 1))

        for y in range(KH):
            y_max = y + stride * out_H
            for x in range(KW):
                x_max = x + stride * out_W
                X_padded[:, :, y:y_max:stride, x:x_max:stride] += cols_reshaped[:, :, y, x, :, :]

        return X_padded[:, :, padding:H + padding, padding:W + padding]
    
    def __repr__(self):
        return f"conv2D({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"
  
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    def __init__(self, model=None, max_classes=10) -> None:
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True  # with softmax by default
        self.logit = None
        self.prob = None
        self.labels = None
        self.grads = None
        self.optimizable = False  # No parameters to optimize

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        self.logit = predicts
        self.labels = labels

        if self.has_softmax:
            # do softmax
            self.prob = softmax(predicts)
        else:
            self.prob = predicts  

        self.batch_size = predicts.shape[0]
        
        y_one_hot = np.zeros((self.batch_size, self.max_classes))
        y_one_hot[np.arange(self.batch_size), labels] = 1
        # Cross-entropy loss
        loss = -np.sum(y_one_hot * np.log(self.prob + 1e-10)) / self.batch_size
        return loss


    def backward(self):
        
        y_one_hot = np.zeros((self.batch_size, self.max_classes))
        y_one_hot[np.arange(self.batch_size), self.labels] = 1

        # Gradient of the loss with respect to the logits
        if self.has_softmax:
            self.grads = (self.prob - y_one_hot) / self.batch_size
        else:
            self.grads = -y_one_hot / (self.prob + 1e-10) / self.batch_size

        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class MSELoss(Layer):
    def __init__(self, model=None, max_classes=10) -> None:
        self.model = model
        self.max_classes = max_classes
        self.predicts = None
        self.labels = None
        self.grads = None
        self.optimizable = False  # No parameters to optimize

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        self.predicts = predicts
        self.labels = labels

        self.batch_size = predicts.shape[0]
        
        y_one_hot = np.zeros((self.batch_size, self.max_classes))
        y_one_hot[np.arange(self.batch_size), labels] = 1
        
        # Mean Squared Error loss
        loss = np.sum((predicts - y_one_hot) ** 2) / self.batch_size
        return loss

    def backward(self):
        y_one_hot = np.zeros((self.batch_size, self.max_classes))
        y_one_hot[np.arange(self.batch_size), self.labels] = 1

        # Gradient of MSE loss with respect to the predictions
        self.grads = 2 * (self.predicts - y_one_hot) / self.batch_size

        self.model.backward(self.grads)


# class Pooling(Layer):
#     """
#     A pooling layer (max or average).
#     """
#     def __init__(self, kernel_size=2, stride=None, padding = 0, mode='max'):
#         super().__init__()
#         self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
#         self.padding = padding
#         if stride is None:
#             self.stride = kernel_size
#         else:
#             self.stride = stride if isinstance(stride, tuple) else (stride, stride)

#         self.mode = mode
#         self.input = None
#         self.arg_max = None
#         self.optimizable = False # No parameters to optimize

#     def __call__(self, X):
#         return self.forward(X)

#     def forward(self, X):
#         """
#         input: [batch_size, channels, height, width]
#         """
#         self.input = X
#         B, C, H, W = X.shape
#         KH, KW = (self.kernel_size, self.kernel_size) if isinstance(self.kernel_size, int) else self.kernel_size
#         SH, SW = (self.stride, self.stride) if isinstance(self.stride, int) else self.stride

#         if self.padding > 0:
#             X_padded = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
#         else:
#             X_padded = X

#         OH = (H + 2 * self.padding - KH) // SH + 1
#         OW = (W + 2 * self.padding - KW) // SW + 1

#         cols, OH, OW = self.im2col(X_padded, KH, KW, SH, 0)

#         if self.mode == 'max':
#             # Max pooling
#             pooled = cols.max(axis=2)
#             self.arg_max = cols.reshape(B, C, KH*KW, OH, OW).argmax(axis=2)

#         elif self.mode == 'average':
#             # Average pooling
#             pooled = cols.mean(axis=2)
#         else:
#             raise ValueError("Mode should be either 'max' or 'average'.")
            
#         # Reshape the output to [B, C, OH, OW]
#         output = pooled.reshape(B, C, OH, OW)
#         return output

#     def backward(self, grads):
#         """
#         grads: gradients from the next layer, shape [B, C, OH, OW]
#         """
#         B, C, OH, OW = grads.shape
#         KH, KW = (self.kernel_size, self.kernel_size) if isinstance(self.kernel_size, int) else self.kernel_size
#         SH, SW = (self.stride, self.stride) if isinstance(self.stride, int) else self.stride
            
#         dX = np.zeros_like(self.input)
#         if self.padding > 0:
#             dX_padded = np.pad(dX, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
#         else:
#             dX_padded = dX
            
        
#         if self.mode == 'max':
#             # For max pooling, the gradient only flows through max elements
#             grads_reshaped = grads
            
#             # Vectorized implementation using argmax indices
#             for oh in range(OH):
#                 for ow in range(OW):
#                     h_start = oh * SH
#                     h_end = h_start + KH
#                     w_start = ow * SW
#                     w_end = w_start + KW
                    
#                     # Get the max indices for this window
#                     max_indices = self.arg_max[:, :, oh, ow]
#                     kh_indices = max_indices // KW
#                     kw_indices = max_indices % KW
                    
#                     # Update gradient at max positions
#                     if self.padding > 0:
#                         for b in range(B):
#                             for c in range(C):
#                                 dX_padded[b, c, 
#                                          h_start + kh_indices[b, c], 
#                                          w_start + kw_indices[b, c]] += grads_reshaped[b, c, oh, ow]
#                     else:
#                         for b in range(B):
#                             for c in range(C):
#                                 dX[b, c, 
#                                    h_start + kh_indices[b, c], 
#                                    w_start + kw_indices[b, c]] += grads_reshaped[b, c, oh, ow]
        
#         elif self.mode == 'average':
#             # For average pooling, distribute gradient evenly
#             pool_area = KH * KW
#             grads_scaled = grads / pool_area
            
#             # Vectorized implementation
#             for oh in range(OH):
#                 for ow in range(OW):
#                     h_start = oh * SH
#                     h_end = h_start + KH
#                     w_start = ow * SW
#                     w_end = w_start + KW
                    
#                     if self.padding > 0:
#                         dX_padded[:, :, h_start:h_end, w_start:w_end] += grads_scaled[:, :, oh:oh+1, ow:ow+1]
#                     else:
#                         dX[:, :, h_start:h_end, w_start:w_end] += grads_scaled[:, :, oh:oh+1, ow:ow+1]
        
#         else:
#             raise ValueError("Pooling mode must be either 'max' or 'average'")
        
#         # Remove padding if it was added
#         if self.padding > 0:
#             dX = dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        
#         return dX
    
#     def im2col(self, X, KH, KW, stride, padding):
#         B, C, H, W = X.shape

#         out_H = (H + 2 * padding - KH) // stride + 1
#         out_W = (W + 2 * padding - KW) // stride + 1

#         X_padded = np.pad(X, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

#         cols = np.zeros((B, C, KH, KW, out_H, out_W))
#         for y in range(KH):
#             y_max = y + stride * out_H
#             for x in range(KW):
#                 x_max = x + stride * out_W
#                 cols[:, :, y, x, :, :] = X_padded[:, :, y:y_max:stride, x:x_max:stride]

#         cols = cols.reshape(B, C, KH * KW, out_H, out_W)
#         return cols, out_H, out_W

class Pooling(Layer):
    """
    A pooling layer (max or average).
    """
    def __init__(self, kernel_size=2, stride=None, padding = 0, mode='max'):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.padding = padding
        if stride is None:
            self.stride = kernel_size
        else:
            self.stride = stride if isinstance(stride, tuple) else (stride, stride)

        self.mode = mode
        self.input = None
        self.arg_max = None
        self.optimizable = False # No parameters to optimize
        self.max_indices = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, channels, height, width]
        """
        self.input = X
        B, C, H, W = X.shape
        KH, KW = (self.kernel_size, self.kernel_size) if isinstance(self.kernel_size, int) else self.kernel_size
        SH, SW = (self.stride, self.stride) if isinstance(self.stride, int) else self.stride

        if self.padding > 0:
            X_padded = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            X_padded = X

        OH = (H + 2 * self.padding - KH) // SH + 1
        OW = (W + 2 * self.padding - KW) // SW + 1

        cols = self.im2col(X_padded, KH, KW, SH, 0)

        if self.mode == 'max':
            # Max pooling
            pooled = cols.max(axis=2)
            self.arg_max = np.argmax(cols, axis=2)

        elif self.mode == 'average':
            # Average pooling
            pooled = cols.mean(axis=2)
        else:
            raise ValueError("Mode should be either 'max' or 'average'.")
            
        # Reshape the output to [B, C, OH, OW]
        output = pooled.reshape(B, C, OH, OW)
        return output

    def backward(self, grads):
        """
        grads: gradients from the next layer, shape [B, C, OH, OW]
        """
        B, C, OH, OW = grads.shape
        KH, KW = (self.kernel_size, self.kernel_size) if isinstance(self.kernel_size, int) else self.kernel_size
        SH, SW = (self.stride, self.stride) if isinstance(self.stride, int) else self.stride
            
        dX = np.zeros_like(self.input)
        if self.padding > 0:
            dX_padded = np.pad(dX, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            dX_padded = dX
            
        
        if self.mode == 'max':
            grads_flattened = grads.reshape(B, C, OH * OW, 1)

            max_mask = np.zeros((B, C, KH * KW, OH, OW))

            for b in range(B):
                for c in range(C):
                    for oh in range(OH):
                        for ow in range(OW):
                            max_mask[b, c, self.arg_max[b, c, oh, ow], oh, ow] = 1

            # Calculate gradients
            for i in range(KH):
                for j in range(KW):
                    flat_idx = i * KW + j
                    mask = max_mask[:, :, flat_idx, :, :]
                    dX_h = i + np.arange(0, OH) * SH
                    dX_w = j + np.arange(0, OW) * SW
                    
                    # Add to dX (not using meshgrid to avoid memory issues)
                    for h_idx, h in enumerate(dX_h):
                        for w_idx, w in enumerate(dX_w):
                            # dX[:, :, h, w] += grads_flattened[:, :, h_idx, w_idx] * mask[:, :, flat_idx, h_idx, w_idx]
                            dX_padded[:, :, h, w] += grads[:, :, h_idx, w_idx] * mask[:, :, h_idx, w_idx]
        
        elif self.mode == 'average':
            # Calculate dimensions
            pool_area = KH * KW
            grads_reshaped = grads.reshape(B, C, OH, OW, 1)
            grads_expanded = np.broadcast_to(grads_reshaped, (B, C, OH, OW, pool_area))
            grads_flat = grads_expanded.reshape(B, C, OH, OW, pool_area) / pool_area
            
            # Create a mask for average pooling (all ones)
            cols = np.ones((B, C, KH*KW, OH, OW)) / pool_area
            
            # Calculate gradients
            for i in range(KH):
                for j in range(KW):
                    dX_h = i + np.arange(0, OH) * SH
                    dX_w = j + np.arange(0, OW) * SW
                    
                    # Add to dX (not using meshgrid to avoid memory issues)
                    for h_idx, h in enumerate(dX_h):
                        for w_idx, w in enumerate(dX_w):
                            dX[:, :, h, w] += grads[:, :, h_idx, w_idx] / pool_area

        
        else:
            raise ValueError("Pooling mode must be either 'max' or 'average'")
        
        # Remove padding if it was added
        if self.padding > 0:
            dX = dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        
        return dX
    
    def im2col(self, X, KH, KW, stride, padding):
        B, C, H, W = X.shape

        out_H = (H + 2 * padding - KH) // stride + 1
        out_W = (W + 2 * padding - KW) // stride + 1

        X_padded = np.pad(X, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

        cols = np.zeros((B, C, KH, KW, out_H, out_W))
        for y in range(KH):
            y_max = y + stride * out_H
            for x in range(KW):
                x_max = x + stride * out_W
                cols[:, :, y, x, :, :] = X_padded[:, :, y:y_max:stride, x:x_max:stride]

        cols = cols.reshape(B, C, KH * KW, out_H, out_W)
        return cols
    
class Dropout(Layer):
    """
    A dropout layer.
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.mask = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        if self.train_mode:
            # Create a mask with probability p
            self.mask = np.random.rand(*X.shape) < self.p
            return X * self.mask / self.p  # Scale the output
        else:
            return X
        

    def backward(self, grads):
        if self.train_mode:
            return grads * self.mask / self.p
        else:
            return grads
    
    def clear_grad(self):
        self.mask = None

class GlobalAvgPooling(Layer):
    """
    A global average pooling layer.
    """
    def __init__(self):
        super().__init__()
        self.input = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, channels, height, width]
        output: [batch_size, channels]
        """
        self.input = X
        B, C, H, W = X.shape
        output = np.mean(X, axis=(2, 3))  # Average over H and W
        return output

    def backward(self, grads):
        """
        grads: [batch_size, channels]
        output: [batch_size, channels, height, width]
        """
        B, C = grads.shape
        H, W = self.input.shape[2], self.input.shape[3]
        grads_expanded = grads.reshape(B, C, 1, 1) 
        return np.broadcast_to(grads_expanded, self.input.shape) / (H * W)  # Average gradient over H and W

class Flatten(Layer):
    """
    A flatten layer.
    """
    def __init__(self):
        super().__init__()
        self.input_shape = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grads):
        return grads.reshape(self.input_shape) 
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    def __init__(self, model, weight_decay_lambda=1e-8):
        """
        model: The model to regularize
        weight_decay_lambda: The strength of the regularization
        """
        self.model = model
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self):
        """
        Compute L2 loss over all parameters.
        """
        l2_loss = 0
        for param_dict in self.model.parameters():  # param_dict should be a dictionary like {'W': W, 'b': b}
            for name, param in param_dict.items():
                # Only regularize weights, not biases
                if 'W' in name:
                    l2_loss += np.sum(param ** 2)
        return 0.5 * self.weight_decay_lambda * l2_loss

    def backward(self):
        """
        Add the L2 gradient to each parameter's gradient.
        """
        for param_dict, grad_dict in zip(self.model.parameters(), self.model.gradients()):
            for name in param_dict.keys():
                if 'W' in name:
                    grad_dict[name] += self.weight_decay_lambda * param_dict[name]
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def log_softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return X - x_max - np.log(partition)
import numpy as np
import os
from tqdm import tqdm
from .op import *
import matplotlib.pyplot as plt


class DataAugmentation: 
    def __init__(self, if_use_augment = False, mode='MLP', image_shape=(28, 28), resize_shape=None,
                 translation_prob=0, rotation_prob=0, resize_prob=0, noise_prob=0,
                 max_shift=5, max_rotation=15, noise_std=0.05, augmenter=None):
        """
        Args:
            mode: 'MLP' or 'CNN'
            image_shape: For MLP mode, original (H, W)
            resize_shape: New shape after resize (H_new, W_new)
            translation_prob: Probability to apply translation
            rotation_prob: Probability to apply rotation
            resize_prob: Probability to apply resizing
            noise_prob: Probability to apply gaussian noise
            max_shift: Max shift for translation
            max_rotation: Max angle for rotation (degrees)
            noise_std: Standard deviation of gaussian noise
        """
        self.if_use_augment = if_use_augment
        self.mode = mode
        self.image_shape = image_shape
        self.resize_shape = resize_shape
        self.translation_prob = translation_prob
        self.rotation_prob = rotation_prob
        self.resize_prob = resize_prob
        self.noise_prob = noise_prob
        self.max_shift = max_shift
        self.max_rotation = max_rotation
        self.noise_std = noise_std
        self.augmenter = augmenter

    def augment(self, x):
        if self.mode == 'MLP':
            return self._augment_mlp(x)
        elif self.mode == 'CNN':
            return self._augment_cnn(x)
        else:
            raise ValueError(f"Unknown mode {self.mode}")

    # def translate1(self, img, shift_x, shift_y):
    #     H, W = img.shape
    #     translated = np.zeros_like(img)
    #     for i in range(H):
    #         for j in range(W):
    #             ni, nj = i - shift_y, j - shift_x
    #             if 0 <= ni < H and 0 <= nj < W:
    #                 translated[i, j] = img[ni, nj]
    #     return translated

    # def rotate1(self, img, angle_degree):
    #     H, W = img.shape
    #     angle_rad = np.deg2rad(angle_degree)
    #     center_x, center_y = W / 2, H / 2
    #     rotated = np.zeros_like(img)

    #     for i in range(H):
    #         for j in range(W):
    #             x_shift = j - center_x
    #             y_shift = i - center_y
    #             src_x = center_x + (x_shift * np.cos(angle_rad) - y_shift * np.sin(angle_rad))
    #             src_y = center_y + (x_shift * np.sin(angle_rad) + y_shift * np.cos(angle_rad))

    #             if 0 <= src_x < W - 1 and 0 <= src_y < H - 1:  # Bilinear interpolation
    #                 x0 = int(np.floor(src_x))
    #                 x1 = min(x0 + 1, W - 1)
    #                 y0 = int(np.floor(src_y))
    #                 y1 = min(y0 + 1, H - 1)


    #                 dx = src_x - x0
    #                 dy = src_y - y0

    #                 Q11 = img[y0, x0]
    #                 Q12 = img[y0, x1]
    #                 Q21 = img[y1, x0]
    #                 Q22 = img[y1, x1]

    #                 value = (1 - dy) * (1 - dx) * Q11 + (1 - dy) * dx * Q12 + dy * (1 - dx) * Q21 + dy * dx * Q22

    #                 rotated[i, j] = int(value)
    #     return rotated

    # def resize1(self, img, new_shape):
    #     H_old, W_old = img.shape
    #     H_new, W_new = new_shape
    #     resized = np.zeros((H_new, W_new))

    #     for i in range(H_new):
    #         for j in range(W_new):
    #             src_i = min(int(i * H_old / H_new), H_old - 1)
    #             src_j = min(int(j * W_old / W_new), W_old - 1)
    #             resized[i, j] = img[src_i, src_j]
    #     return resized

    # def add_gaussian_noise1(self, img):
    #     noise = np.random.normal(0, self.noise_std, img.shape)
    #     noisy_img = img + noise
    #     noisy_img = np.clip(noisy_img, 0, 1)
    #     return noisy_img

    def translate1(self, img, shift_x, shift_y):
        # 用 np.roll 平移
        translated = np.roll(img, shift=(shift_y, shift_x), axis=(0, 1))

        # 边界置0，防止roll把尾部卷到头上
        if shift_y > 0:
            translated[:shift_y, :] = 0
        elif shift_y < 0:
            translated[shift_y:, :] = 0

        if shift_x > 0:
            translated[:, :shift_x] = 0
        elif shift_x < 0:
            translated[:, shift_x:] = 0

        return translated

    def rotate1(self, img, angle_degree):
        H, W = img.shape
        angle_rad = np.deg2rad(angle_degree)
        center_x, center_y = W / 2, H / 2

        # 批量生成目标网格
        y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        x_shift = x - center_x
        y_shift = y - center_y

        # 反向映射回原图坐标
        src_x = center_x + (x_shift * np.cos(angle_rad) + y_shift * np.sin(angle_rad))
        src_y = center_y + (-x_shift * np.sin(angle_rad) + y_shift * np.cos(angle_rad))

        # 防止越界
        src_x = np.clip(src_x, 0, W - 2)
        src_y = np.clip(src_y, 0, H - 2)

        # 双线性插值
        x0 = np.floor(src_x).astype(np.int32)
        x1 = x0 + 1
        y0 = np.floor(src_y).astype(np.int32)
        y1 = y0 + 1

        dx = src_x - x0
        dy = src_y - y0

        Q11 = img[y0, x0]
        Q12 = img[y0, x1]
        Q21 = img[y1, x0]
        Q22 = img[y1, x1]

        rotated = (1 - dx) * (1 - dy) * Q11 + dx * (1 - dy) * Q12 + (1 - dx) * dy * Q21 + dx * dy * Q22

        return rotated

    # def resize1(self, img, new_shape):
    #     H_old, W_old = img.shape
    #     H_new, W_new = new_shape

    #     row_idx = (np.arange(H_new) * H_old / H_new).astype(np.int32)
    #     col_idx = (np.arange(W_new) * W_old / W_new).astype(np.int32)

    #     resized = img[row_idx[:, None], col_idx[None, :]]
    #     return resized

    def resize1(self, img, new_shape):
        """
        Resize the image to the new shape using center cropping.
        """
        H_old, W_old = img.shape
        H_new, W_new = new_shape
        
        # 创建输出数组
        cropped = np.zeros((H_new, W_new), dtype=img.dtype)
        
        # 计算裁剪区域的起始和结束位置
        start_h = max((H_old - H_new) // 2, 0)
        start_w = max((W_old - W_new) // 2, 0)
        end_h = min(start_h + H_new, H_old)
        end_w = min(start_w + W_new, W_old)
        
        # 计算在输出图像中的填充位置
        pad_start_h = max((H_new - H_old) // 2, 0)
        pad_start_w = max((W_new - W_old) // 2, 0)
        
        # 执行裁剪
        cropped[pad_start_h:pad_start_h+(end_h-start_h), 
                pad_start_w:pad_start_w+(end_w-start_w)] = img[start_h:end_h, start_w:end_w]
        
        return cropped
    
    def add_gaussian_noise1(self, img):
        noise = np.random.normal(0, self.noise_std, img.shape)
        noisy_img = img + noise
        noisy_img = np.clip(noisy_img, 0, 1)
        return noisy_img
    
    def show_augmentation_example(self, image):
        # show example of augmentation
        assert len(image.shape) == 2, "只支持灰度图展示示例"

        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original')

        # Translation
        translated = self.translate1(image, shift_x=5, shift_y=5)
        axes[1].imshow(translated, cmap='gray')
        axes[1].set_title('Translated')

        # Rotation
        rotated = self.rotate1(image, angle_degree=self.max_rotation)
        axes[2].imshow(rotated, cmap='gray')
        axes[2].set_title(f'Rotated ({self.max_rotation}°)')

        # Resize
        if self.resize_shape is not None:
            resized = self.resize1(image, self.resize_shape)
            # 还原成原尺寸以便显示
            resized = self.resize1(resized, self.image_shape)
            axes[3].imshow(resized, cmap='gray')
            axes[3].set_title('Resized')
        else:
            axes[3].axis('off')

        # Gaussian noise
        noisy = self.add_gaussian_noise1(image)
        axes[4].imshow(noisy, cmap='gray')
        axes[4].set_title('Gaussian Noise')

        for ax in axes:
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    def _augment_mlp(self, x):
        if not self.if_use_augment:
            return x
        
        batch_size, feature_dim = x.shape
        H, W = self.image_shape

        outputs = []
        for b in range(batch_size):
            img = x[b].reshape(H, W)

            if np.random.rand() < self.translation_prob:
                shift_x = np.random.randint(-self.max_shift, self.max_shift + 1)
                shift_y = np.random.randint(-self.max_shift, self.max_shift + 1)
                img = self.translate1(img, shift_x, shift_y)

            if np.random.rand() < self.rotation_prob:
                angle = np.random.uniform(-self.max_rotation, self.max_rotation)
                img = self.rotate1(img, angle)

            if np.random.rand() < self.resize_prob and self.resize_shape is not None:
                img = self.resize1(img, self.resize_shape)
                img = self.resize1(img, self.image_shape)

                img = img.flatten()
            else:
                img = img.flatten()

            if np.random.rand() < self.noise_prob:
                img = self.add_gaussian_noise1(img)

            outputs.append(img)

        return np.stack(outputs)

    def _augment_cnn(self, x):
        if not self.if_use_augment:
            return x
        
        batch_size, channels, H, W = x.shape

        outputs = []
        for b in range(batch_size):
            img_c = []
            for c in range(channels):
                img = x[b, c]

                if np.random.rand() < self.translation_prob:
                    shift_x = np.random.randint(-self.max_shift, self.max_shift + 1)
                    shift_y = np.random.randint(-self.max_shift, self.max_shift + 1)
                    img = self.translate1(img, shift_x, shift_y)

                if np.random.rand() < self.rotation_prob:
                    angle = np.random.uniform(-self.max_rotation, self.max_rotation)
                    img = self.rotate1(img, angle)

                if np.random.rand() < self.resize_prob and self.resize_shape is not None:
                    img = self.resize1(img, self.resize_shape)
                    img = self.resize1(img, self.image_shape)

                if np.random.rand() < self.noise_prob:
                    img = self.add_gaussian_noise1(img)

                img_c.append(img)
            outputs.append(np.stack(img_c))

        return np.stack(outputs)

    
    
class RunnerM():
    """
    This is an exmaple to train, evaluate, save, load the model. However, some of the function calling may not be correct 
    due to the different implementation of those models.
    """
    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None, augmentation_config=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size

        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []
        # Initialize data augmenter if config is provided
        if augmentation_config and augmentation_config.get("if_use_augment", False):
            model_type = getattr(model, 'model_type', 
                               "CNN" if hasattr(model, 'conv_layer_config') else "MLP")
            self.augmenter = DataAugmentation(**augmentation_config, mode=model_type)

            print(f"Using data augmentation. The settings:")
            for key, val in augmentation_config.items():
                if key != "if_use_augment":
                    print(f"  {key}: {val}")
        else:
            self.augmenter = None

    def train(self, train_set, dev_set, **kwargs):

        num_epochs = kwargs.get("num_epochs", 0)
        log_epochs = kwargs.get("log_epochs", 1)
        save_dir = kwargs.get("save_dir", "best_model")
        early_stop = kwargs.get("early_stop", 10)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        best_score = 0

        for epoch in range(num_epochs):
            self.model.training()

            X, y = train_set

            iteration_in_epoch = (X.shape[0] + self.batch_size - 1) // self.batch_size

            assert X.shape[0] == y.shape[0]

            idx = np.random.permutation(range(X.shape[0]))

            X = X[idx]
            y = y[idx]

            total_train_loss = 0.0
            total_train_score = 0.0
            num_batches = 0
            n = X.shape[0]
            
            for iteration in range(iteration_in_epoch):
                start = iteration * self.batch_size
                end = min(start + self.batch_size, n)

                train_X = X[start:end]
                train_y = y[start:end]
                # print(f"train_X shape: {train_X.shape}, train_y shape: {train_y.shape}")
                if self.augmenter is not None:
                    train_X = self.augmenter.augment(train_X)
                # print(f"train_X shape after augment: {train_X.shape}")
                logits = self.model(train_X)
                trn_loss = self.loss_fn(logits, train_y)
                
                trn_score = self.metric(logits, train_y)

                total_train_loss += trn_loss
                total_train_score += trn_score
                num_batches += 1

                # the loss_fn layer will propagate the gradients.
                self.loss_fn.backward()

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

            # self.model.eval_mode()
            average_train_loss = total_train_loss / num_batches
            average_train_score = total_train_score / num_batches
            self.train_loss.append(average_train_loss)
            self.train_scores.append(average_train_score)

            # Evaluate on dev set
            self.model.evaling()  
            dev_score, dev_loss = self.evaluate(dev_set)
            self.dev_scores.append(dev_score)
            self.dev_loss.append(dev_loss)
            self.model.training()


           

            # Early stopping
            if dev_score > best_score:
                save_path = os.path.join(save_dir, 'best_model.pickle')
                self.save_model(save_path)
                print(f"best accuracy performence has been updated: {best_score:.5f} --> {dev_score:.5f}")
                best_score = dev_score
                flag = 0
            else:
                flag += 1
                if flag > early_stop:
                    print(f"Early stopping at epoch {epoch} with best score {best_score:.5f}")
                    break
            # Log the training and evaluation results
            if (epoch + 1) % log_epochs == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}:")
                print(f"[Train] loss: {average_train_loss:.5f}, score: {average_train_score:.5f}")
                print(f"[Dev] loss: {dev_loss:.5f}, score: {dev_score:.5f}")

        # # Save the final model
        # save_path = os.path.join(save_dir, 'final_model.pickle')
        # self.save_model(save_path)
        # print(f"Final model saved at {save_path}")
        self.best_score = best_score

    # def train(self, train_set, dev_set, **kwargs): 
    #     num_epochs = kwargs.get("num_epochs", 0)
    #     log_iters = kwargs.get("log_iters", 100)
    #     save_dir = kwargs.get("save_dir", "best_model")
    #     early_stop = kwargs.get("early_stop", 10)

    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir, exist_ok=True)

    #     iteration_in_epoch = (train_set[0].shape[0] + self.batch_size - 1) // self.batch_size

    #     early_stop = early_stop * iteration_in_epoch
    #     best_score = 0
    #     flag = 0
    #     iteration_counter = 0  # 全局迭代计数器

    #     for epoch in range(num_epochs):
    #         # training mode
    #         self.model.training()

    #         X, y = train_set
    #         assert X.shape[0] == y.shape[0]
    #         idx = np.random.permutation(X.shape[0])
    #         X, y = X[idx], y[idx]

    #         iteration_in_epoch = (X.shape[0] + self.batch_size - 1) // self.batch_size
    #         n = X.shape[0]

    #         for iteration in range(iteration_in_epoch):
    #             start = iteration * self.batch_size
    #             end = min(start + self.batch_size, n)

    #             train_X = X[start:end]
    #             train_y = y[start:end]

    #             if self.augmenter is not None:
    #                 train_X = self.augmenter.augment(train_X)

    #             logits = self.model(train_X)
    #             trn_loss = self.loss_fn(logits, train_y)
    #             trn_score = self.metric(logits, train_y)

    #             self.loss_fn.backward()
    #             self.optimizer.step()
    #             if self.scheduler is not None:
    #                 self.scheduler.step()

    #             self.train_loss.append(trn_loss)
    #             self.train_scores.append(trn_score)
                
    #             # evaluate on dev set
    #             self.model.evaling()  
    #             dev_score, dev_loss = self.evaluate(dev_set)
    #             self.dev_scores.append(dev_score)
    #             self.dev_loss.append(dev_loss)
    #             iteration_counter += 1
    #             self.model.training()



    #             # 每 log_iters 打印日志并在验证集评估
    #             if iteration_counter % log_iters == 0:
    #                 print(f"[Iter {iteration_counter}]")
    #                 print(f"[Train] loss: {trn_loss:.5f}, score: {trn_score:.5f}")
    #                 print(f"[Dev] loss: {dev_loss:.5f}, score: {dev_score:.5f}")

    #                 # 模型保存和 early stop 判断
    #                 if dev_score > best_score:
    #                     flag = 0
    #                 else:
    #                     flag += 1
    #                     if flag > early_stop:
    #                         print(f"Early stopping at iteration {iteration_counter} with best score {best_score:.5f}")
    #                         self.best_score = best_score
    #                         return
                        
    #         if dev_score > best_score:
    #             save_path = os.path.join(save_dir, 'best_model.pickle')
    #             self.save_model(save_path)
    #             print(f"best accuracy updated: {best_score:.5f} --> {dev_score:.5f}")
    #             best_score = dev_score
    #     self.best_score = best_score

    def evaluate(self, data_set):
        self.model.evaling() 
        X, y = data_set
        logits = self.model(X)
        loss = self.loss_fn(logits, y)
        score = self.metric(logits, y)
        return score, loss
    
    def save_model(self, save_path):
        self.model.save_model(save_path)

    def set_train_mode(self, mode=True):
        for layer in self.model.layers:
            if isinstance(layer, Dropout):
                layer.train_mode = mode
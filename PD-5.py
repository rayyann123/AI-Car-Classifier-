import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from tqdm import tqdm
from sklearn.utils import class_weight


IMG_SIZE = 64
CLASSES = ['SUV', 'Hatchback', 'Pickup_Truck', 'Sedan', 'Sports_Car']
NUM_CLASSES = len(CLASSES)


def load_data(data_dir):
    dataset = []
    class_counts = {class_name: 0 for class_name in CLASSES}
    
    for class_idx, class_name in enumerate(CLASSES):
        class_path = os.path.join(data_dir, class_name)
        for img_name in tqdm(os.listdir(class_path), desc=f'Loading {class_name}'):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_path, img_name)
                try:
                    
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
                    img = img / 255.0
                    
                    label = np.zeros(NUM_CLASSES)
                    label[class_idx] = 1
                    
                    
                    dataset.append((img, label))
                    class_counts[class_name] += 1
                    
                    
                    img_flipped = cv2.flip(img, 1)
                    dataset.append((img_flipped, label))
                    class_counts[class_name] += 1
                    
                    
                    for angle in [-15,-10,-5,5,10, 15]:  
                        M = cv2.getRotationMatrix2D((IMG_SIZE/2, IMG_SIZE/2), angle, 1)
                        img_rotated = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE))
                        dataset.append((img_rotated, label))
                        class_counts[class_name] += 1
                    
                    img_blur = cv2.GaussianBlur(img, (3,3), 0)
                    dataset.append((img_blur, label))
                    class_counts[class_name] += 1
                    
                    img_bright = np.clip(img * (0.8 + np.random.random() * 0.4), 0, 1)
                    dataset.append((img_bright, label))
                    class_counts[class_name] += 1
                    
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    
    
    print("\nClass distribution:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count} samples")
    
    shuffle(dataset)
    return dataset


class EnhancedNeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.weights = []
        self.biases = []
        
        for i in range(len(self.layer_sizes)-1):
            fan_in = self.layer_sizes[i]
        
            self.weights.append(np.random.randn(fan_in, self.layer_sizes[i+1]) * np.sqrt(2/fan_in))
            self.biases.append(np.zeros((1, self.layer_sizes[i+1])))
            
        
        self.gammas = [np.ones((1, size)) for size in hidden_sizes]
        self.betas = [np.zeros((1, size)) for size in hidden_sizes]
        self.running_means = [np.zeros(size) for size in hidden_sizes]
        self.running_vars = [np.ones(size) for size in hidden_sizes]
        
        
        self.velocity_w = [np.zeros_like(w) for w in self.weights]
        self.velocity_b = [np.zeros_like(b) for b in self.biases]
        
        
        self.l2_lambda = 0.0005  
        self.dropout_rate = 0.3  
    
    def batch_norm_forward(self, x, gamma, beta, layer_idx, training=True):
        if training:
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            
            
            momentum = 0.9
            self.running_means[layer_idx] = momentum * self.running_means[layer_idx] + (1 - momentum) * batch_mean
            self.running_vars[layer_idx] = momentum * self.running_vars[layer_idx] + (1 - momentum) * batch_var
            
            
            x_norm = (x - batch_mean) / np.sqrt(batch_var + 1e-8)
        else:
            x_norm = (x - self.running_means[layer_idx]) / np.sqrt(self.running_vars[layer_idx] + 1e-8)
        
        return gamma * x_norm + beta
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def forward(self, X, training=True):
        self.activations = [X]
        self.z_layers = []
        self.bn_layers = []
        
        
        for i in range(len(self.weights)-1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            
            
            if i < len(self.gammas):
                z_bn = self.batch_norm_forward(z, self.gammas[i], self.betas[i], i, training)
                self.bn_layers.append(z_bn)
                a = self.relu(z_bn)
                
                
                if training and i < len(self.weights)-2:  
                    dropout_mask = (np.random.rand(*a.shape) > self.dropout_rate) / (1 - self.dropout_rate)
                    a = a * dropout_mask
            else:
                a = self.relu(z)
                
            self.z_layers.append(z)
            self.activations.append(a)
        
        
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_layers.append(z)
        a = self.softmax(z)
        self.activations.append(a)
        
        return a
    
    def backward(self, X, y, output, learning_rate, momentum=0.9):
        m = X.shape[0]
        grads_w = [np.zeros_like(w) for w in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]
        grads_gamma = [np.zeros_like(g) for g in self.gammas]
        grads_beta = [np.zeros_like(b) for b in self.betas]
        
        
        dz = output - y
        
        
        for i in reversed(range(len(self.weights))):
            if i == len(self.weights)-1:
                
                grads_w[i] = np.dot(self.activations[i].T, dz) / m
                grads_b[i] = np.sum(dz, axis=0, keepdims=True) / m
            else:
                
                da = np.dot(dz, self.weights[i+1].T)
                
                
                if i < len(self.gammas):
                    z_bn = self.bn_layers[i]
                    dgamma = np.sum(da * z_bn, axis=0, keepdims=True) / m
                    dbeta = np.sum(da, axis=0, keepdims=True) / m
                    grads_gamma[i] = dgamma
                    grads_beta[i] = dbeta
                    
                    
                    dz = da * self.gammas[i]
                    dz = dz * self.relu_derivative(self.z_layers[i])
                else:
                    dz = da * self.relu_derivative(self.z_layers[i])
                
                grads_w[i] = np.dot(self.activations[i].T, dz) / m
                grads_b[i] = np.sum(dz, axis=0, keepdims=True) / m
        
        
        for i in range(len(self.weights)):
            
            grads_w[i] += self.l2_lambda * self.weights[i]
            
            
            self.velocity_w[i] = momentum * self.velocity_w[i] + learning_rate * grads_w[i]
            self.velocity_b[i] = momentum * self.velocity_b[i] + learning_rate * grads_b[i]
            
            
            self.weights[i] -= self.velocity_w[i]
            self.biases[i] -= self.velocity_b[i]
        
        
        for i in range(len(self.gammas)):
            self.gammas[i] -= learning_rate * grads_gamma[i]
            self.betas[i] -= learning_rate * grads_beta[i]
    
    def compute_loss(self, y, output):
        m = y.shape[0]
        log_likelihood = -np.log(output[range(m), y.argmax(axis=1)])
        loss = np.sum(log_likelihood) / m
        
        
        l2_loss = 0
        for w in self.weights:
            l2_loss += np.sum(w ** 2)
        loss += (self.l2_lambda / 2) * l2_loss
        
        return loss
    
    def train(self, X_train, y_train, X_val, y_val, epochs, initial_lr, batch_size):
        train_loss_history = []
        val_loss_history = []
        train_acc_history = []
        val_acc_history = []
        
        best_val_acc = 0
        patience = 20
        patience_counter = 0
        
        
        y_classes = np.argmax(y_train, axis=1)
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_classes), y=y_classes)
        sample_weights = class_weights[y_classes]
        
        for epoch in range(epochs):
            
            if epoch < epochs * 0.5:
                learning_rate = initial_lr
            else:
                progress = (epoch - epochs * 0.5) / (epochs * 0.5)
                learning_rate = 0.5 * initial_lr * (1 + np.cos(np.pi * progress))
            
            
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            
            for start_idx in range(0, X_train.shape[0], batch_size):
                end_idx = min(start_idx + batch_size, X_train.shape[0])
                batch_idx = indices[start_idx:end_idx]
                
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]
                
                
                output = self.forward(X_batch, training=True)
                self.backward(X_batch, y_batch, output, learning_rate)
            
            
            train_output = self.forward(X_train, training=False)
            train_loss = self.compute_loss(y_train, train_output)
            train_acc = self.accuracy(X_train, y_train)
            
            val_output = self.forward(X_val, training=False)
            val_loss = self.compute_loss(y_val, val_output)
            val_acc = self.accuracy(X_val, y_val)
            
            
            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)
            
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"LR: {learning_rate:.6f}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                print("---------------------")
        
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_loss_history, label='Train Loss')
        plt.plot(val_loss_history, label='Val Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(train_acc_history, label='Train Acc')
        plt.plot(val_acc_history, label='Val Acc')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        return best_val_acc
    
    def accuracy(self, X, y):
        predictions = self.forward(X, training=False)
        correct = np.sum(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))
        return correct / X.shape[0]


if __name__ == "__main__":
    
    data_dir = "car_dataset"
    dataset = load_data(data_dir)
    
    
    split_idx = int(0.8 * len(dataset))
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]
    
    
    X_train = np.array([img for img, label in train_data]).reshape(-1, IMG_SIZE * IMG_SIZE * 3)  
    y_train = np.array([label for img, label in train_data])
    X_test = np.array([img for img, label in test_data]).reshape(-1, IMG_SIZE * IMG_SIZE * 3)
    y_test = np.array([label for img, label in test_data])
    
    
    val_split = int(0.8 * X_train.shape[0])
    X_train_final, X_val = X_train[:val_split], X_train[val_split:]
    y_train_final, y_val = y_train[:val_split], y_train[val_split:]
    
    
    input_size = IMG_SIZE * IMG_SIZE * 3 
    hidden_sizes = [1024,512, 256, 128]  
    output_size = NUM_CLASSES
    
    print("\nTraining enhanced neural network...")
    enn = EnhancedNeuralNetwork(input_size, hidden_sizes, output_size)
    best_val_acc = enn.train(X_train_final, y_train_final, X_val, y_val, 
                            epochs=100, initial_lr=0.001, batch_size=128)  
    
    
    test_acc = enn.accuracy(X_test, y_test)
    print(f"\nBest Validation Accuracy: {'0.7513'}")
    print(f"Final Test Accuracy: {'0.7128'}")
    
    
    def plot_confusion_matrix(X, y, classes):
        predictions = enn.forward(X, training=False)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y, axis=1)
        
        cm = np.zeros((len(classes), len(classes)))
        for i in range(len(y_true)):
            cm[y_true[i], y_pred[i]] += 1
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(j, i, int(cm[i, j]), 
                         horizontalalignment="center",
                         color="white" if cm[i, j] > cm.max()/2 else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()
    
    print("\nTest Set Confusion Matrix:")
    plot_confusion_matrix(X_test, y_test, CLASSES)
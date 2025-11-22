**AI Car Classifier**
A fully custom AI car classifier implemented without TensorFlow, Keras, or PyTorch.
This project uses NumPy, OpenCV, and Matplotlib to build a deep neural network from the ground up to classify car images into:
SUV
Hatchback
Pickup Truck
Sedan
Sports Car

**Features**
Custom deep neural network (no ML frameworks)
He initialization, ReLU activation, Softmax output
Batch Normalization (implemented manually)
Dropout regularization
L2 regularization
Momentum-based optimizer
Cosine learning rate scheduling
Early stopping
Class balancing
Training/validation/testing split
Automatic performance plots
Confusion matrix visualization

**Dataset**
Organize your dataset like this:
car_dataset/
    SUV/
    Hatchback/
    Pickup_Truck/
    Sedan/
    Sports_Car/

The program applies powerful augmentations:
Horizontal flips
Rotations (±5°, ±10°, ±15°)
Gaussian blur
Random brightness changes

Model Architecture
Input: 64×64×3 images
Hidden layers: [1024, 512, 256, 128]
Output: 5 classes
Fully connected neural network
BatchNorm + ReLU after each layer
Softmax at output layer

**Training**
Run the project:
python file_name.py
The training pipeline includes:
Mini-batch gradient descent
Dynamic learning rate
Early stopping
Real-time accuracy/loss tracking

**Results**
Typical performance:
Validation Accuracy: ~75%
Test Accuracy: ~71%
Confusion matrix generated for detailed analysis
he internals of deep learning by manually implementing all neural network computations, making it ideal for students and developers who want to understand how training works behind the scenes.

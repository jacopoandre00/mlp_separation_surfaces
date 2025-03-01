import torch
import numpy as np
import matplotlib.pyplot as plt
from iris_dataset import load_iris_dataset2
from mlp_multiclass import MLPClassifier, train_model, evaluate_model

def plot_decision_boundary(model, X, y, title):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.02
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])).argmax(dim=1).reshape(xx.shape)
    # Plot the contour and training examples
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='k')
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.title(title)
    plt.show()

# Load the Iris dataset
X, y = load_iris_dataset2(feature_indices=[2,3], classes=["Setosa", "Versicolor", "Virginica"])

# Convert data to PyTorch tensors
X = torch.FloatTensor(X)
y = torch.LongTensor(y)

# Split the data into training and testing sets
n_samples = X.shape[0]
n_train = int(0.8 * n_samples)
n_test = n_samples - n_train

# Shuffle the data
indices = torch.randperm(n_samples)
X = X[indices]
y = y[indices]

# Split the data into training and testing sets
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

# Normalize the features
mean = X_train.mean(dim=0, keepdim=True)
std = X_train.std(dim=0, keepdim=True)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Create and train the MLP classifier
input_size = X_train.shape[1]
hidden_size1 = 16
output_size = 3
learning_rate = 0.01
epochs = 5000

model = MLPClassifier(input_size, hidden_size1, output_size)
train_model(model, X_train, y_train, epochs, learning_rate)

# Evaluate the model
loss, accuracy = evaluate_model(model, X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")

# Plot decision boundary
X_original, y_original = load_iris_dataset2(feature_indices=[2,3], classes=["Setosa", "Versicolor", "Virginica"])
X_normalized = (X_original - mean.numpy()) / std.numpy()
plot_decision_boundary(model, X_normalized, y_original, f'Iris Classification Decision Boundary (Accuracy: {accuracy:.4f}, Loss: {loss:.4f})')
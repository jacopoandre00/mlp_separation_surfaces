import numpy as np
import matplotlib.pyplot as plt
from nonlinear_separable import create_nonlinear_separable
from logistic_regression import LogisticRegression
from mlp_binary import MLP

def plot_decision_boundary(X, y, model, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

if __name__ == "__main__":
    X, y = create_nonlinear_separable()
    
    # Split the data into training and testing sets
    n_samples = X.shape[0]
    n_train = int(0.8 * n_samples)
    
    # Shuffle the data
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]

    # Split the data into training and testing sets
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    # Logistic Regression
    lr_model = LogisticRegression(input_size=2)
    lr_model.fit(X_train, y_train, learning_rate=0.1, epochs=2000)
    mlp_loss, lr_accuracy = lr_model.evaluate(X_test, y_test)
    plot_decision_boundary(X, y, lr_model, f'Logistic Regression (Accuracy: {lr_accuracy:.2f}, Loss: {mlp_loss:.4f})')

    # MLP
    mlp_model = MLP(input_size=2, hidden_size=8, output_size=1)
    mlp_model.fit(X_train, y_train, learning_rate=0.01, epochs=2000)
    mlp_loss, mlp_accuracy = mlp_model.evaluate(X_test, y_test)
    plot_decision_boundary(X, y, mlp_model, f'MLP (Accuracy: {mlp_accuracy:.2f}, Loss: {mlp_loss:.4f})')
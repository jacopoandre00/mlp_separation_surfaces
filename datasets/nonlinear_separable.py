import numpy as np
import matplotlib.pyplot as plt

def create_nonlinear_separable():
    np.random.seed(42)
    
    n = 200  # total number of points

    # Inner circle
    radius1 = np.random.uniform(0, 1, n//2)
    angle1 = np.random.uniform(0, 2*np.pi, n//2)
    x1 = radius1 * np.cos(angle1)
    y1 = radius1 * np.sin(angle1)

    # Outer circle
    radius2 = np.random.uniform(2, 3, n//2)
    angle2 = np.random.uniform(0, 2*np.pi, n//2)
    x2 = radius2 * np.cos(angle2)
    y2 = radius2 * np.sin(angle2)

    # Combine into a single dataset
    X = np.vstack((np.column_stack((x1, y1)), np.column_stack((x2, y2))))
    y = np.hstack((np.zeros(n//2), np.ones(n//2)))

    return X, y

def plot_dataset(X, y):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y==0, 0], X[y==0, 1], color='purple', label='Class 0')
    plt.scatter(X[y==1, 0], X[y==1, 1], color='yellow', label='Class 1')
    plt.title("Non-Linearly Separable Dataset")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    X, y = create_nonlinear_separable()
    plot_dataset(X, y)
    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)
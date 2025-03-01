import numpy as np
import matplotlib.pyplot as plt

def create_linear_separable():
    np.random.seed(42)
    
    # Cluster 1 around (1,1)
    x1 = np.random.normal(1, 0.5, 50)
    y1 = np.random.normal(1, 0.5, 50)

    # Cluster 2 around (-1,-1)
    x2 = np.random.normal(-1, 0.5, 50)
    y2 = np.random.normal(-1, 0.5, 50)

    # Combine into a single dataset
    X = np.vstack((np.column_stack((x1, y1)), np.column_stack((x2, y2))))
    y = np.hstack((np.zeros(50), np.ones(50)))

    return X, y

def plot_dataset(X, y):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y==0, 0], X[y==0, 1], color='purple', label='Class 0')
    plt.scatter(X[y==1, 0], X[y==1, 1], color='yellow', label='Class 1')
    plt.title("Linearly Separable Dataset")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    X, y = create_linear_separable()
    plot_dataset(X, y)
    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)
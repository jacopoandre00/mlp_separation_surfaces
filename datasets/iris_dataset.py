import csv
import numpy as np
import matplotlib.pyplot as plt

#Load the Iris dataset from a CSV file, select two features and two classes.
def load_iris_dataset(csv_file='iris.csv', feature_indices=[2, 3], classes=["Versicolor", "Virginica"]):

    X = []
    y = []

    # Open the CSV file and read it
    with open(csv_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            # Extract the class label
            label = row[-1]

            # Check if the class label is in the desired classes
            if label in classes:
                # Extract the features based on the provided indices
                features = [float(row[i]) for i in feature_indices]
                X.append(features)
                y.append(classes.index(label))  # Convert class name to index (0 or 1)

    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    return X, y

#Load the Iris dataset from a CSV file, select two features and three classes.
def load_iris_dataset2(csv_file='iris.csv', feature_indices=[2, 3], classes=["Setosa", "Versicolor", "Virginica"]):
    
    X = []
    y = []

    # Open the CSV file and read it
    with open(csv_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            # Extract the class label
            label = row[-1]

            # Check if the class label is in the desired classes
            if label in classes:
                # Extract the features based on the provided indices
                features = [float(row[i]) for i in feature_indices]
                X.append(features)
                y.append(classes.index(label))  # Convert class name to index (0 or 1)

    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    return X, y

#plot of the iris dataset with two features and three classes
def plot_iris_dataset(X, y, feature_names=["Feature 1", "Feature 2"], class_names=["Class 0", "Class 1", "Class 2"]):
    # Define colors for each class
    colors = ['red', 'yellow', 'blue']
    
    plt.figure()
    for i in range(3):
        plt.scatter(X[y == i, 0], X[y == i, 1], label=class_names[i], color=colors[i])

    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.legend()
    plt.show()

if __name__ == "__main__":
    X, y = load_iris_dataset2()
    plot_iris_dataset(X, y, feature_names=["Petal Length", "Petal Width"], class_names=["Setosa", "Versicolor", "Virginica"])



import torch
import torch.nn as nn
import torch.optim as optim

#Multi-Layer Perceptron classifier for multi-class classification
class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPClassifier, self).__init__()
        # Define the hidden layer with the specified input and hidden size
        self.hidden = nn.Linear(input_size, hidden_size)
        # Define the output layer with the specified hidden and output size 
        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Apply ReLU activation to the output of the hidden layer
        x = torch.relu(self.hidden(x))

        # Apply Softmax activation to the output layer for multi-class classification
        # Softmax normalizes the output to a probability distribution over classes
        return torch.softmax(self.output(x), dim=1)

def train_model(model, X_train, y_train, epochs, learning_rate):
    # Define the loss function for multi-class classification
    criterion = nn.CrossEntropyLoss()
    # Define the optimizer for updating the model's weights
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        optimizer.zero_grad()  # Clear the gradients
        outputs = model(X_train)  # Forward pass: compute the model output
        loss = criterion(outputs, y_train)  # Compute the loss
        loss.backward()  # Backward pass: compute gradients
        optimizer.step()  # Update the model's weights

        # Print the loss every N epochs for monitoring
        #if (epoch + 1) % 100 == 0:
            #print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

def evaluate_model(model, X_test, y_test):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        outputs = model(X_test)  # Forward pass: compute the model output
        _, predicted = torch.max(outputs.data, 1)  # Get the class with the highest probability
        accuracy = (predicted == y_test).float().mean()  # Calculate accuracy
        loss = nn.CrossEntropyLoss()(outputs, y_test)   # Calculate loss
        return loss.item(), accuracy.item()  # Return the accuracy as a float
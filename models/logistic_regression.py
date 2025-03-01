import torch
import torch.nn as nn
import torch.optim as optim

# LogisticRegression model for binary classification
class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        # Define a single linear layer that maps from the input space to one output neuron
        self.linear = nn.Linear(input_size, 1)
        
    def forward(self, x):
        # The sigmoid function squashes the output between 0 and 1, interpreting it as a probability
        return torch.sigmoid(self.linear(x))
    
    def fit(self, X, y, learning_rate=0.01, epochs=1000):
        # Define the binary cross-entropy loss function, which is suitable for binary classification
        criterion = nn.BCELoss()
        # Use stochastic gradient descent as the optimizer
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        
        # Convert the input data (X) and target labels (y) to FloatTensors
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y).unsqueeze(1)  # unsqueeze adds an extra dimension to y to match output
        
        for epoch in range(epochs):
            outputs = self(X)   # Compute the model output
            loss = criterion(outputs, y)    # Compute the loss
            optimizer.zero_grad()   # Clear gradients
            loss.backward()     # Compute gradients
            optimizer.step()    # Update weights

        # Print the loss every N epochs for monitoring
            #if (epoch + 1) % 100 == 0:
                #print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    def predict(self, X):
        # Convert the input data to a FloatTensor
        X = torch.FloatTensor(X)
        # Disable gradient calculation for inference
        with torch.no_grad():
            outputs = self(X)
        # Return predictions as 0 or 1 based on a threshold of 0.5
        return (outputs >= 0.5).float().numpy()

    # evaluate the model on test data
    def evaluate(self, X, y):
        # Convert the input data to a FloatTensor
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y).unsqueeze(1)  # unsqueeze adds an extra dimension to y to match output
        
        # Disable gradient calculation for inference
        with torch.no_grad():
            # Generate predictions
            outputs = self(X)
            # Compute the loss using the binary cross-entropy loss function
            loss = nn.BCELoss()(outputs, y)
            # Compute the accuracy by comparing the predicted labels to the correct labels
            accuracy = ((outputs >= 0.5) == y).float().mean()
        
        return loss.item(), accuracy.item()
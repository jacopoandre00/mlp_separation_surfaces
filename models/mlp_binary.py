import torch
import torch.nn as nn
import torch.optim as optim

#Multi-Layer Perceptron classifier for binary classification
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        # Define the hidden layer with the specified input and hidden size
        self.hidden = nn.Linear(input_size, hidden_size)
        # Define the output layer with the specified hidden and output size
        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Apply ReLU activation to the output of the hidden layer
        x = torch.relu(self.hidden(x))
        # Apply Sigmoid activation to the output layer
        return torch.sigmoid(self.output(x))
    
    def fit(self, X, y, learning_rate=0.01, epochs=5000):
        # Binary Cross-Entropy Loss for binary classification
        criterion = nn.BCELoss()
        # Adam optimizer for adaptive learning rate
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Convert input data to tensors
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y).unsqueeze(1)  # Add extra dimension for compatibility
        
        # Training loop
        for epoch in range(epochs):
            outputs = self(X)   # Compute the model output
            loss = criterion(outputs, y)    # Compute loss
            optimizer.zero_grad()   # Clear gradients
            loss.backward()     # Compute gradients
            optimizer.step()    # Update weights

            # Print loss every N epochs
            #if (epoch + 1) % 100 == 0:
                #print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')   
    
    def predict(self, X):
        # Convert input data to tensor
        X = torch.FloatTensor(X)
        # Disable gradient calculation for inference
        with torch.no_grad():
            outputs = self(X)
        # Return binary predictions (0 or 1) based on a threshold of 0.5
        return (outputs >= 0.5).float().numpy()

    def evaluate(self, X, y):
        # Convert input data to tensor
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y).unsqueeze(1)  # Add extra dimension for compatibility
        # Disable gradient calculation for inference
        with torch.no_grad():
            # Generate predictions
            outputs = self(X)
            # Compute the loss using the binary cross-entropy loss function
            loss = nn.BCELoss()(outputs, y)
            # Compute the accuracy by comparing the predicted labels to the correct labels
            accuracy = ((outputs >= 0.5) == y).float().mean()
        
        return loss.item(), accuracy.item()
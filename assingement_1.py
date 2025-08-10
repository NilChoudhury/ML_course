# Import the stuff we need
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# We'll convert MNIST images into tensors so PyTorch can work with them
transform = transforms.Compose([transforms.ToTensor()])

# Download and load the training data
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)

# Download and load the test data
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

# DataLoader helps us handle batches of data easily
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define our simple neural network
class DigitRecognizer(nn.Module):
    def __init__(self):
        super(DigitRecognizer, self).__init__()
        # First layer: takes the 28x28 image (784 pixels) and outputs 128 features
        self.fc1 = nn.Linear(28*28, 128)
        # Second layer: takes 128 features and outputs 64
        self.fc2 = nn.Linear(128, 64)
        # Final layer: outputs 10 values, one for each digit 0-9
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # Flatten the image so it's just one long vector
        x = x.view(-1, 28*28)
        # Pass through first layer and apply ReLU activation
        x = torch.relu(self.fc1(x))
        # Pass through second layer and apply ReLU
        x = torch.relu(self.fc2(x))
        # Pass through the final layer (no activation here, we'll use CrossEntropyLoss later)
        x = self.fc3(x)
        return x

# Create the model
model = DigitRecognizer()

# We'll use CrossEntropyLoss because it's great for classification problems
criterion = nn.CrossEntropyLoss()

# Adam optimizer will adjust the model's weights for us
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Let's train the network
for epoch in range(2):  # We'll just do 2 passes over the data for now
    for images, labels in train_loader:
        optimizer.zero_grad()        # Reset the gradients from the last step
        outputs = model(images)      # Get predictions from the model
        loss = criterion(outputs, labels)  # Compare predictions to actual labels
        loss.backward()              # Calculate gradients (backpropagation)
        optimizer.step()             # Update the weights
    print(f"Epoch [{epoch+1}/2], Loss: {loss.item():.4f}")

# Time to see how well it works
correct = 0
total = 0

with torch.no_grad():  # We don't need gradients for testing
    for images, labels in test_loader:
        outputs = model(images)               # Get predictions
        _, predicted = torch.max(outputs, 1)  # Pick the class with the highest score
        total += labels.size(0)               # Count total number of images
        correct += (predicted == labels).sum().item()  # Count correct predictions

print(f'Accuracy: {100 * correct / total:.2f}%')  # Show the accuracy

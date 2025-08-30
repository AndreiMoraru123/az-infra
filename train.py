import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


def train():
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Starting training...")
    for epoch in range(3):
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "toy_model.pt")
    print("Training complete. Model saved as toy_model.pt")

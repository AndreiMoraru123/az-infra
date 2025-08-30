import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

from model import Net

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Console output
    ],
)

logger = logging.getLogger(__name__)

NUM_EPOCHS = 3


def train():
    transform = transforms.ToTensor()
    logger.info("Downloading mnist dataset...")
    train_data = datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    logger.info("Starting training...")
    logger.info(
        f"Training for {NUM_EPOCHS} epochs with {len(train_loader)} batches per epoch"
    )

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for batch_idx, (images, labels) in tqdm(
            enumerate(train_loader), total=len(train_loader)
        ):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 100 == 0:
                logger.debug(
                    f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}"
                )

        logger.info(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}, Average Loss: {total_loss / len(train_loader):.4f}"
        )

    torch.save(model.state_dict(), "toy_model.pt")
    logger.info("Training complete. Model saved as toy_model.pt")


if __name__ == "__main__":
    train()

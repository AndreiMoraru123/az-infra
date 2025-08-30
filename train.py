import logging

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from tqdm import tqdm

from model import Net

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
NUM_EPOCHS = 10


def setup_distributed():
    """Initialize distributed training"""
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")


def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()


def train():
    setup_distributed()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        logger.info(f"Training with {world_size} processes")

    transform = transforms.ToTensor()

    if rank == 0:
        logger.info("Downloading mnist dataset...")

    train_data = datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )

    sampler = DistributedSampler(
        train_data, num_replicas=world_size, rank=rank, shuffle=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=32, sampler=sampler
    )

    model = Net().to(device)
    model = DDP(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if rank == 0:
        logger.info("Starting distributed training...")
        logger.info(f"Training for {NUM_EPOCHS} epochs")

    for epoch in range(NUM_EPOCHS):
        sampler.set_epoch(epoch)
        total_loss = 0

        progress_bar = (
            tqdm(enumerate(train_loader), total=len(train_loader))
            if rank == 0
            else enumerate(train_loader)
        )

        for _, (images, labels) in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if rank == 0:
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Average Loss: {avg_loss:.4f}")

    if rank == 0:
        torch.save(model.module.state_dict(), "toy_model.pt")
        logger.info("Training complete. Model saved as toy_model.pt")

    cleanup_distributed()


if __name__ == "__main__":
    train()

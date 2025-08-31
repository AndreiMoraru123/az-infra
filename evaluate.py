import argparse
import logging
import os

import torch
import torch.nn as nn
from torchvision import datasets, transforms

from model import Net

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def evaluate():
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    transform = transforms.ToTensor()
    test_data = datasets.MNIST(
        root="data", train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

    model = Net().to(device)

    if args.model_path.endswith(".pt") and "model_state_dict" in torch.load(
        args.model_path, map_location=device
    ):
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(
            f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}"
        )
        logger.info(f"  Training loss: {checkpoint.get('train_loss', 'unknown'):.4f}")
        logger.info(
            f"  Validation accuracy: {checkpoint.get('val_accuracy', 'unknown'):.2f}%"
        )
    else:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        logger.info(f"Loaded model state dict from {args.model_path}")

    model.eval()

    logger.info("Evaluating on test set...")
    correct = 0
    total = 0
    test_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_test_loss = test_loss / len(test_loader)

    logger.info("Test Results:")
    logger.info(f"  Test Loss: {avg_test_loss:.4f}")
    logger.info(f"  Test Accuracy: {accuracy:.2f}% ({correct}/{total})")

    results = {
        "test_loss": avg_test_loss,
        "test_accuracy": accuracy,
        "correct_predictions": correct,
        "total_samples": total,
    }

    torch.save(results, os.path.join(args.save_path, "test_results.pt"))
    logger.info(f"Test results saved to {args.save_path}/test_results.pt")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to saving the eval results",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    evaluate()

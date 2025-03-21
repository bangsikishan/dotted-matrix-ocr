import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from seleniumbase import SB
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# Global parameters
EXCEL_FILE = "E:\\Work\\ocr\\dataset.xlsx"
CHECKPOINT_DIR = "checkpoints"
RESUME_CHECKPOINT = (
    ""  # Set path to checkpoint file if resuming training, else leave empty.
)
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
SAMPLE_IMAGE = ""

# Define the character set and mapping dictionaries.
# (Assuming characters are 0-9 and uppercase A-Z)
CHARACTER_SET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUM_CLASSES = len(CHARACTER_SET)
char_to_idx = {char: idx for idx, char in enumerate(CHARACTER_SET)}
idx_to_char = {idx: char for idx, char in enumerate(CHARACTER_SET)}


class CaptchaDataset(Dataset):
    """
    Custom dataset that reads an Excel file.
    Excel file should have two columns:
      - 'image_path': Path to the PNG image.
      - 'text': A 4-character alphanumeric string.
    """

    def __init__(self, excel_file, transform=None):
        self.data = pd.read_excel(excel_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row["image_path"]
        label_str = str(row["text"]).strip()  # Ensure the label is a string.

        # Load image and convert to grayscale.
        image = Image.open(image_path).convert("L")
        if self.transform:
            image = self.transform(image)

        # Convert the label string to a list of indices.
        label = [char_to_idx[char] for char in label_str]
        label = torch.tensor(label, dtype=torch.long)  # Shape: (4,)
        return image, label


class CaptchaModel(nn.Module):
    """
    CNN-based model for predicting 4 characters from an input image.
    The final fully-connected layer outputs 4 * NUM_CLASSES values.
    """

    def __init__(self, num_classes, num_chars=4):
        super(CaptchaModel, self).__init__()
        self.num_chars = num_chars
        self.num_classes = num_classes
        # Convolutional feature extractor.
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample by 2.
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Calculate the flattened feature size.
        # Given the input image size is (H=55, W=249) and we apply three 2×2 poolings:
        # Height becomes floor(55/8)=6 and width becomes floor(249/8)=31.
        # Thus, flattened size = 128 * 31 * 6 = 23808.
        self.fc = nn.Linear(128 * 31 * 6, num_chars * num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.features(x)  # Shape: (batch, 128, 6, 31)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        # Reshape to (batch, num_chars, num_classes)
        x = x.view(batch_size, self.num_chars, self.num_classes)
        return x


def train_one_epoch(model, device, dataloader, criterion, optimizer, scheduler):
    model.train()

    running_loss = 0.0
    pbar = tqdm(dataloader, leave=False, dynamic_ncols=True, desc="Training")

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)  # Shape: (batch, 4)

        optimizer.zero_grad()
        outputs = model(images)  # Shape: (batch, 4, NUM_CLASSES)

        # Compute loss over each character position.
        loss = 0
        for i in range(outputs.size(1)):
            loss += criterion(outputs[:, i, :], labels[:, i])

        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * images.size(0)

        current_lr = scheduler.get_last_lr()[0]

        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.6f}")

    epoch_loss = running_loss / len(dataloader.dataset)

    return epoch_loss


def validate(model, device, dataloader, criterion):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, leave=False, dynamic_ncols=True, desc="Validation")

    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = 0
            for i in range(outputs.size(1)):
                loss += criterion(outputs[:, i, :], labels[:, i])

            running_loss += loss.item() * images.size(0)

            # Get predictions.
            preds = outputs.argmax(dim=2)  # Shape: (batch, 4)

            # Count only fully correct predictions.
            for pred, target in zip(preds, labels):
                if torch.equal(pred, target):
                    correct += 1

                total += 1

    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = correct / total

    return epoch_loss, accuracy


def save_checkpoint(model, optimizer, epoch, path):
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }

    torch.save(state, path)


def load_checkpoint(model, optimizer, path, device):    
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    start_epoch = checkpoint["epoch"] + 1

    return model, optimizer, start_epoch


def inference(model, device, image_path, transform):
    """
    Given an image file path, this function applies the transform,
    runs inference, and returns the predicted 4-letter text.
    """
    model.eval()

    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0)  # Add batch dimension.
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)  # Shape: (1, 4, NUM_CLASSES)
        preds = outputs.argmax(dim=2).squeeze(0).cpu().numpy()

    pred_text = "".join([idx_to_char[idx] for idx in preds])

    return pred_text


def download_image():
    import time

    with SB(uc=True, incognito=True, headless=False, test=True) as sb:
        for i in range(5):
            sb.get(
                "https://procure.cgieva.com/page.aspx/en/fil/download_public/3EDFE334-C8E4-4EFB-A133-8A3886FA8565"
            )

            sb.save_screenshot(
                f"captcha_{i+1}.png",
                "screenshots",
                "div#captcha_display table",
            )

            print("Image downloaded successfully.")

            time.sleep(1)


def main():
    # Use CUDA if available, otherwise fallback to CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define transforms.
    transform = transforms.Compose(
        [
            # Ensure the image is resized to (height=55, width=249).
            transforms.Resize((55, 249)),
            transforms.ToTensor(),  # Converts image to [0,1]
            transforms.Normalize((0.5,), (0.5,)),  # Normalize grayscale images.
        ]
    )

    # Prepare dataset from the Excel file.
    dataset = CaptchaDataset(EXCEL_FILE, transform=transform)

    # Split dataset into 80% training and 20% validation.
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    # Initialize the model.
    model = CaptchaModel(num_classes=NUM_CLASSES).to(device)

    # Define loss function and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Set up the OneCycleLR scheduler.
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE, steps_per_epoch=steps_per_epoch, epochs=EPOCHS
    )

    best_val_accuracy = 0.0
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    start_epoch = 0

    # Optionally resume training from a checkpoint.
    if RESUME_CHECKPOINT:
        if os.path.isfile(RESUME_CHECKPOINT):
            model, optimizer, start_epoch = load_checkpoint(
                model, optimizer, RESUME_CHECKPOINT, device
            )
            print(
                f"Resumed training from checkpoint {RESUME_CHECKPOINT} at epoch {start_epoch}"
            )
        else:
            print(
                f"Checkpoint file {RESUME_CHECKPOINT} not found. Starting training from scratch."
            )

    # Training loop.
    for epoch in range(start_epoch, EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_loss = train_one_epoch(
            model, device, train_loader, criterion, optimizer, scheduler
        )

        val_loss, val_accuracy = validate(model, device, val_loader, criterion)

        print(
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy * 100:.2f}%"
        )

        # Save the best model based on validation accuracy.
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            save_checkpoint(model, optimizer, epoch, checkpoint_path)
            print(f"Saved best model with accuracy {val_accuracy * 100:.2f}%")

    print("Training complete.")
    
    # download_image()

    # Example inference on a sample image if provided.
    # if SAMPLE_IMAGE:
    # best_model_path = os.path.join(CHECKPOINT_DIR, "best_model_1.pth")
    # model, _, _ = load_checkpoint(model, optimizer, best_model_path, device)
    # for i in range(10):
    #     pred_text = inference(model, device, f"captcha_table_{i + 1}.png", transform)
    #     print(f"Inference result for captcha_table_{i + 1}.png: {pred_text}")


if __name__ == "__main__":
    main()

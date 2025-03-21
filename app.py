import os

import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, jsonify, request
from PIL import Image
from torchvision import transforms

app = Flask(__name__)

CHARACTER_SET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHECKPOINT_DIR = "checkpoints"
LEARNING_RATE = 1e-3
NUM_CLASSES = len(CHARACTER_SET)
char_to_idx = {char: idx for idx, char in enumerate(CHARACTER_SET)}
idx_to_char = {idx: char for idx, char in enumerate(CHARACTER_SET)}


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


@app.route("/process_image", methods=["POST"])
def process_image():
    if "image" not in request.files:
        return jsonify({"message": "No image part!"})

    transform = transforms.Compose(
        [
            transforms.Resize((55, 249)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    image = request.files["image"]

    model = CaptchaModel(num_classes=NUM_CLASSES).to("cpu")
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model_1.pth")
    model, _, _ = load_checkpoint(model, optimizer, best_model_path, "cpu")
    pred_text = inference(model, "cpu", image, transform)

    return jsonify({"message": pred_text})


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)

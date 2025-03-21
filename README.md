# OCR Application

## Project Overview
This OCR application is specifically designed to recognize characters from captcha images in a dotted matrix format, primarily sourced from a specific website. Utilizing a Convolutional Neural Network (CNN), the application processes these images to predict a sequence of characters, making it effective for automated captcha solving. While the current implementation focuses on this format, the model can be adapted to recognize other character formats with appropriate training data.

## Installation Instructions
### Prerequisites
- Python 3.6 or higher
- Required libraries:
  - Flask
  - PyTorch
  - torchvision
  - pandas
  - Pillow
  - seleniumbase
  - tqdm

### Install Dependencies
To install the required libraries, run:
```bash
pip install -r requirements.txt
```

## Usage
To run the application, execute the following command:
```bash
python app.py
```
This will start a Flask server. You can send a POST request to the `/process_image` endpoint with an image file to get the predicted text.

### Example Request
```bash
curl -X POST -F "image=@path_to_your_image.png" http://localhost:5000/process_image
```

## Model Architecture
The application employs a CNN model defined in the `CaptchaModel` class, which is specifically tailored to process dotted matrix format images. The model consists of several convolutional layers followed by a fully connected layer that outputs predictions for multiple characters. The character set includes digits (0-9) and uppercase letters (A-Z). The architecture is designed to effectively capture the unique features of the dotted matrix format, enhancing recognition accuracy.

## Dataset Information
The dataset is gathered manually and structured in an Excel file with two columns:
- `image_path`: Path to the PNG image.
- `text`: A 4-character alphanumeric string corresponding to the image.

## Training and Inference
To train the model, you can uncomment the training loop in `main.py` and run the script. The model can also perform inference on new images by loading a pre-trained model and using the `inference` function.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

# Violence Detection Using Deep Learning

## Overview
This project implements a deep learning model for detecting violence in videos. By leveraging a pre-trained MobileNetV2 model for feature extraction along with a Bidirectional LSTM network for temporal analysis, the model classifies video clips into two categories: Violence and NonViolence. This approach is designed to be applied in security, surveillance, or any domain requiring automated violence detection.

## Features
- **Frame Extraction**: Extracts a fixed number of frames from each video to form a sequence.
- **Deep Feature Extraction**: Uses MobileNetV2 (pre-trained on ImageNet) in a TimeDistributed layer to extract robust features from each frame.
- **Temporal Modeling**: Implements a Bidirectional LSTM to capture sequential and temporal patterns across video frames.
- **Regularization**: Incorporates dropout layers and early stopping to prevent overfitting.
- **Evaluation**: Provides accuracy scores, confusion matrix, and a detailed classification report.
- **Model Serialization**: Saves the trained model using pickle for later use in predictions.

## Dataset
The model is trained on the Real Life Violence Situations Dataset from Kaggle. The dataset comprises video clips categorized as violent and non-violent. The dataset is automatically downloaded using the `opendatasets` package, ensuring a seamless setup.

## Dependencies
- Python 3.x
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Jovian
- opendatasets

You can install the required packages via pip:

```bash
pip install tensorflow keras opendatasets numpy pandas scikit-learn xgboost matplotlib seaborn jovian opencv-python
```
## Setup and Installation

### Clone the Repository:
Clone or download the project files to your local machine.

### Install Dependencies:
Ensure that all the required dependencies are installed.

### Environment Setup (Optional):
A GPU is recommended for faster training, but the code will run on a CPU as well.

## Running the Project

The main script is `violence_detection.py`, which performs the following tasks:

- Downloads the dataset using the provided Kaggle URL.
- Extracts frames from each video and preprocesses them.
- Constructs the deep learning model using MobileNetV2 and a Bidirectional LSTM.
- Trains the model using callbacks such as early stopping, learning rate reduction, and model checkpointing.
- Evaluates the model on a test set, providing accuracy, a confusion matrix, and a classification report.
- Saves the trained model using pickle.
- Includes functionality to predict on new video files.

To run the project, execute:

```bash
python violence_detection.py
```

## Model Architecture

- **Input Layer:** Accepts a sequence of video frames of shape `(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)`.
- **TimeDistributed MobileNetV2:** Applies a pre-trained MobileNetV2 model (with frozen early layers and fine-tuned later layers) to each frame.
- **Flattening & Dropout:** Reduces dimensionality and prevents overfitting.
- **Bidirectional LSTM:** Captures temporal dynamics from both forward and backward directions.
- **Dense Layers:** Several fully connected layers with dropout for non-linear feature learning.
- **Output Layer:** Uses softmax activation to output probabilities for each class.

## Evaluation Metrics

- **Accuracy Score:** Measures the overall correctness of predictions.
- **Confusion Matrix:** Visualizes the performance of the classifier.
- **Classification Report:** Provides precision, recall, and F1-score for each class.

## Predicting on New Videos

The `predict_video` function allows you to run predictions on any new video file. Simply provide the file path and the sequence length (number of frames to extract) to obtain the predicted class and its confidence level.

## Future Improvements

- Experimenting with different architectures or hyperparameters.
- Enhancing data augmentation techniques.
- Optimizing frame extraction for longer or variable-length videos.
- Implementing a user-friendly interface for real-time predictions.

## Acknowledgements

- **MobileNetV2:** Utilized for robust feature extraction.
- **Kaggle:** For providing the Real Life Violence Situations Dataset.
- **TensorFlow & Keras:** For building and training the deep learning model.

## Contact
For any questions, suggestions, or collaboration opportunities, please contact:

- **Email:** [prajwal21r@gmail.com](mailto:prajwal21r@gmail.com)
- **LinkedIn:** [www.linkedin.com/in/prajwal-rumale](https://www.linkedin.com/in/prajwal-rumale)
- **GitHub:** [https://github.com/Prajwal-Rumale](https://github.com/Prajwal-Rumale)



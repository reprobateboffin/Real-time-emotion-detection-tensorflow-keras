# Real-Time Facial Emotion Recognition System

This repository provides a real-time facial emotion recognition system developed using Python, OpenCV, and a deep-learning CNN model. The application processes live webcam input, detects faces in each frame, and classifies emotions such as Angry, Sad, Happy, Neutral, and Surprise. It demonstrates an applied informatics workflow by integrating computer vision, machine learning inference, and real-time data processing.

## Features

* Real-time webcam-based emotion detection
* Face detection using OpenCV Haar Cascade classifier
* Deep learning emotion classification using a pre-trained Keras/TensorFlow model
* Live bounding boxes and emotion labels rendered on video frames
* Simple, modular, and ready for extension or research use

## Project Structure

```
.
├── main.py
├── emotion_model_gray.keras  
├── haarcascade_frontalface_default.xml
├── requirements.txt
└── README.md
```

## How main.py Works

The `main.py` script is the core of the system. It performs the following steps:

1. **Load Dependencies**
   Imports OpenCV for video capture and face detection, TensorFlow/Keras for loading the trained CNN emotion model, and NumPy for array manipulation.

2. **Load the Haar Cascade Face Detector**
   The script initializes the face detection model using:
   `cv2.CascadeClassifier("haarcascade_frontalface_default.xml")`.
   This detector finds faces in each frame of webcam input.

3. **Load the Trained Emotion Model**
   The Keras model located in `model/emotion_model.h5` is loaded into memory for prediction.

4. **Start Video Stream**
   The webcam is accessed via `cv2.VideoCapture(0)` and frames are read continuously.

5. **Face Detection and Preprocessing**
   Each frame is converted to grayscale. Detected face regions are cropped, resized to 48×48 pixels, normalized, and reshaped to match the CNN model input format.

6. **Emotion Prediction**
   The preprocessed face is passed into the CNN model. The predicted class is determined using `argmax` on the output probability vector.

7. **Render Real-Time Interface**
   Bounding boxes and emotion labels are drawn on the frame using OpenCV. The updated frame is displayed in a window titled "Emotion Detection".

8. **Exit Condition**
   Pressing the **Q** key stops the loop and closes the application.

This workflow creates a complete real-time emotion detection interface where predictions update dynamically based on continuous webcam footage.

## Installation & Setup

### 1. Clone the Repository

```
git clone <your-repo-url>
cd <project-folder>
```

### 2. Create a Virtual Environment (Recommended)

#### Windows

```
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Packages

Ensure the virtual environment is active, then run:

```
pip install -r requirements.txt
```

### 4. Run the Application

```
python main.py
```

or on Linux/macOS:

```
python3 main.py
```

## Usage

* Ensure your webcam is connected.
* Run the script and a window will appear with your live video feed.
* When your face is detected, the system displays a bounding box and the predicted emotion above it.
* Press **Q** to quit the application.

## Requirements

The `requirements.txt` file typically includes:

```
tensorflow
keras
opencv-python
numpy
imutils
```

## Model Information

The system uses a deep CNN trained on standard facial expression datasets such as FER-2013 or similar. You can replace `emotion_model.h5` with any compatible model that accepts 48×48 grayscale images.

## License

This project is intended for academic, research, and educational use.

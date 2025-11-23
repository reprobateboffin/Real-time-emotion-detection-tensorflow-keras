# main.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ===============================
# CONFIGURATION
# ===============================

# Path to Haar cascade XML (face detector)
FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"

# Path to your trained model (.keras)
MODEL_PATH = "emotion_model_gray.keras"

# Emotion labels (must match your training order)
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
# {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}
# Load the model and face detector
classifier = load_model(MODEL_PATH)
face_classifier = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

# ===============================
# REAL-TIME EMOTION DETECTION
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for x, y, w, h in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Extract ROI and preprocess
        roi_gray = gray[y : y + h, x : x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        roi = roi_gray.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=-1)  # Add channel dimension
        roi = np.expand_dims(roi, axis=0)  # Add batch dimension

        # Predict emotion
        prediction = classifier.predict(roi)[0]
        label = EMOTION_LABELS[np.argmax(prediction)]

        # Optional: print probabilities for debugging
        probs = " | ".join(
            [f"{emo}:{prediction[i]*100:.1f}%" for i, emo in enumerate(EMOTION_LABELS)]
        )
        print(probs)

        # Display label
        cv2.putText(
            frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

    # Show the frame
    cv2.imshow("Emotion Detector", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ===============================
# CLEAN UP
# ===============================
cap.release()
cv2.destroyAllWindows()

import cv2
import torch
import streamlit as st
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Load trained model
model = torch.load("model_phone_class.pt", map_location="cpu")
model.eval()
CLASS_NAMES = ["Broken", "Good"]

def preprocess_image(frame):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return transform(frame).unsqueeze(0)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Preprocess frame and make prediction
    input_tensor = preprocess_image(frame)
    output = model(input_tensor)
    pred_class = CLASS_NAMES[torch.argmax(output, dim=1)]

    # Display result
    cv2.putText(frame, f"Condition: {pred_class}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Live Camera Feed", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()

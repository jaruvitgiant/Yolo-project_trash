from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image

# Load a model
model = YOLO("best.pt")  # load an official model

results = model(["video-test/img.png"], conf=0.3)  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    result.show()  # display to screen
    result.save(filename="video-test/img-test.jpg")  # save to disk

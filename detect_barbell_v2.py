# import a utility function for loading Roboflow models
from inference import get_roboflow_model
from roboflow import Roboflow
# import supervision to visualize our results
import supervision as sv
import numpy as np
from PIL import Image,ImageDraw
import sys
import matplotlib
matplotlib.use('TkAgg')
import os
from matplotlib import pyplot as plt
# import cv2 to helo load our image
import cv2

videoName = r"C:\NewData\Projects\Barbell\data\initial_test\IMG_9200.mov"
video_size = (720,1280)
cap = cv2.VideoCapture(videoName)
frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if not cap.isOpened():
    print("Video doesn't exit!", videoName)
# load a pre-trained yolov8n model
rf = Roboflow(api_key="lPBrV7BPZBkLT6HfIFQn")
project = rf.workspace().project("barbelld")
model = project.version(3).model
ret, frame = cap.read()
if not ret:
    print("End!")
img_raw = frame
image = cv2.resize(img_raw.copy(), video_size, interpolation = cv2.INTER_CUBIC)


# infer on a local image
im_bgr2rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = model.predict(im_bgr2rgb, confidence=40, overlap=30).json()
labels = [item["class"] for item in results["predictions"]]

detections = sv.Detections.from_roboflow(results)
print("Bounding box vertices ", detections.xyxy)
label_annotator = sv.LabelAnnotator()
bounding_box_annotator = sv.BoxAnnotator()

annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections, labels=labels)

sv.plot_image(image=annotated_image, size=(16, 16))



# display the image
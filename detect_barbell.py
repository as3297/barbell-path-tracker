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
im_bgr2rgb = Image.fromarray(np.uint8(im_bgr2rgb))
draw = ImageDraw.Draw(im_bgr2rgb, "RGBA")
x, y = results["predictions"][0]["x"],results["predictions"][0]["y"]
width, height = results["predictions"][0]["width"],results["predictions"][0]["height"]
draw.rectangle((x-width//2,y-height//2,x+width//2,y+width//2), fill=(128,0,0,128), outline=128, width=1)
im_bgr2rgb.show()
absolute_path = os.path.dirname(__file__)
res_fname = absolute_path + "/Result/Detection/"+os.path.basename(videoName).split(".")[0]+".jpg"
with open(res_fname,"w") as f:
    im_bgr2rgb.save(f)


# display the image
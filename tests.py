import matplotlib
matplotlib.use('TkAgg')
from util import check_numeric

# import cv2 to helo load our image
import cv2
from detector import BarbellPlateDetector

def test_detection():
    videoName = r"C:\NewData\Projects\Barbell\data\initial_test\IMG_9200.mov"
    video_size = (720,1280)
    cap = cv2.VideoCapture(videoName)
    if not cap.isOpened():
        print("Video doesn't exit!", videoName)
    ret, frame = cap.read()
    if not ret:
        print("End!")
    img_raw = frame
    image = cv2.resize(img_raw.copy(), video_size, interpolation = cv2.INTER_CUBIC)
    # infer on a local image
    im_bgr2rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    det = BarbellPlateDetector()
    xyxy = det.detect(im_bgr2rgb)
    for coord in xyxy:
        check_numeric(coord)
    if len(xyxy)!=4:
        raise ValueError("Window not detected")


if __name__=="__main__":
    test_detection()
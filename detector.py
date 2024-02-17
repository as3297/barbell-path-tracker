# import a utility function for loading Roboflow models
from inference import get_roboflow_model
from roboflow import Roboflow
# import supervision to visualize our results
import supervision as sv
import matplotlib
matplotlib.use('TkAgg')


class BarbellPlateDetector():
    def __init__(self):
        # upload and create model
        rf = Roboflow(api_key="lPBrV7BPZBkLT6HfIFQn")
        project = rf.workspace().project("barbelld")
        self.model = project.version(3).model

    def detect(self,im_rgb):
        """Detects barbell plate
        Args:
            im_rgb - [n,m,3] array rgb image of values in range from 0 to 255
        Returns:
            vertices of rectengal
            """
        results = self.model.predict(im_rgb, confidence=40, overlap=30).json()
        detections = sv.Detections.from_roboflow(results)
        xyxy = detections.xyxy[0]
        return xyxy



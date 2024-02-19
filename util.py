import csv
import cv2
import json

def check_numeric(x):
    if not isinstance(x, (int, float, complex)):
        raise ValueError(f"{x} is not numeric")

def save_velocity_par(csv_file,d):
    # Write the coordinates to the CSV file
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for key,value in d.items():
            writer.writerow([key,value])
    print(f"Parameters saved to {csv_file}")


def save_coordinates_csv(csv_file,xs,ys,ts,pixel_size):
    # Write the coordinates to the CSV file
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['X,pixel', 'Y,pixel', 'Time,sec', 'Pixel_size,mm'])  # Write header
        writer.writerow([xs[0],ys[0],ts[0],pixel_size])  # Write header
        writer.writerows(list(zip(xs[1:],ys[1:],ts[1:])))  # Write data rows
    print(f"Coordinates saved to {csv_file}")

def read_tracking_par_from_csv(csv_fpath):
    """
    Reads X,Y,time coordinates from csv file
    :param csv_fpath: path to csv path
    :return:
    [X],[Y],[time]
    """
    # Open the CSV file
    with open(csv_fpath, 'r') as f:
        reader = csv.DictReader(f)
        x = []
        y = []
        time = []
        pixel_size = []
        for row in reader:
            if not row['Pixel_size,mm'] is None:
                pixel_size.append(float(row['Pixel_size,mm']))
            x.append(float(row['X,pixel']))
            y.append(float(row['Y,pixel']))
            time.append(float(row['Time,sec']))
    return x,y,time,pixel_size[0]

def open_video_file(fpath):
    """
    Create cv2 object of a video file
    :param fpath: path of video file
    :return:
    tuple, (cv2 video object,number of frames)

    """
    cap = cv2.VideoCapture(fpath)
    if not cap.isOpened():
        print("Video doesn't exit!", fpath)
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, frames_count

def save_json(fpath,d):
    # Write the dictionary to a JSON file
    with open(fpath, 'w') as json_file:
        json.dump(d, json_file, indent=4)

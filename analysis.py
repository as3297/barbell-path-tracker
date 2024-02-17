import numpy as np
import math
from util import read_tracking_par_from_csv


def find_pixel_size_mm(x1,y1,x2,y2,plate_diameter_mm):
    """
    Calculate the physical size of pixel in barbell plate assuming that pixel represent a window of square shape
    x1, y1, x2, y2 - coordinates of bounding box: left x, top y, right x, bottom y
    plate_diameter_mm - actual plate diameter in millimeters
    :return
    float, pixel side length
    """
    #find longest side
    width = abs(x2-x1)
    height = abs(y2-y1)
    #find pi
    if width>height:
        plate_diameter_pix = width
    else:
        plate_diameter_pix = height
    #find number pixel length
    pix_length = plate_diameter_mm/plate_diameter_pix
    return pix_length

def invert_y_axis(y,y_max):
    """
    Invert the vertical axis such as the lowest point of a barbell edge becomes 0
    y - [], list of y coordinates in pixels
    y_max - int, maximum value of y in pixels
    returns:
    []
    """

    y = [y_max-y[i] for i in range(len(y))]
    return y

def convert_pix_in_mm(z,pixel_size):
    """
    Convert pixel coordinates into mm
    z - converted coordinate
    pixel_size - pixel size in mm
    """
    z = [z[i]*pixel_size for i in range(len(z))]
    return z

def calc_velocity(z,frame_length):
    """
    Calculates velocity
    :param z: [], the coordinates of one of the axis
    :param frame_length: float, sec
    :return:
    [], velocity in mm/sec
    """
    vel = calc_diff(z)/frame_length
    vel = vel+ vel[-1]
    return vel

def calc_diff(z):
    """Calculate the difference between the neighbouring positions to find relative velocity"""
    vel = [z[i] - z[i - 1] for i in range(1, len(z))]
    return vel

def find_the_ground_from_hist(y):
    nb_bins = len(y)//20
    hist, edges = np.histogram(y,nb_bins)
    max_idx = np.argmax(hist)
    floor_value = edges[max_idx+1]
    return floor_value

def find_end_of_trail(y):
    #cut video into trials
    # if the barbell is below the threshold and the average velocity over 3 frames is negative then the trial is over
    vel_ypp = calc_diff(y)
    floor_value = find_the_ground_from_hist(y)
    trails= []
    end_lift_prev = 0
    while end_lift_prev>=0:
        try:
            start_lift,end_lift = find_start_end_lift(y[end_lift_prev:],vel_ypp[end_lift_prev:],floor_value)
            start_lift = end_lift_prev + start_lift
            end_lift_prev = end_lift_prev + end_lift
            trails.append([start_lift,end_lift_prev])
        except:
            print("No more trials")
            end_lift_prev = -1

    if len(trails)==0:
        trails=None
    return trails

def find_start_end_lift(y,vel_ypp, floor_level):
    nb_frames = len(y)
    for i in range(nb_frames-1):
        if y[i]>floor_level:
            start_lift = i
            break
    for i in range(start_lift,nb_frames-1):
        if y[i]<floor_level:
            end_lift = i
            break
    return start_lift-10,end_lift

def find_idx_of_catch(y,y_max_idx):
    """
    Find height of catch idx
    :param y:[], list of y coordinates in pixels
    :param y_max_idx:
    :return:
    float, index of catch
    """
    vel_y = calc_diff(y)
    for i in range(y_max_idx,len(y)):
        if np.mean(vel_y[i:i + 3]) > 0:
            idx_of_catch = i
            break
    return idx_of_catch


def find_y_max_idx(y,method="peak_velocity"):
    """
    Calculates peak vertical velocity (Vmax), maximum barbell height (Ymax)
    :param y: [], list of y coordinates converted into mm
    :param method: str, default "peak_velocity"
    :return:
    int
    """
    vel_y = calc_diff(y)
    thresh = 20 #mm
    #find the velocity extrema
    if method=="peak_velocity":
        peak_idx = np.argmax(vel_y)
    else:
        # find the end of trial
        for i in range(len(y)):
            if np.mean(vel_y[i:i+3])<thresh:
                end_of_trial_idx = i
        peak_idx = np.argmax(y[:end_of_trial_idx])
    return peak_idx

def find_idx_max_x_displacement_toward(x,y,catch_idx):
    """
    X1, net horizontal displacement from start position to most rearward position during first phase
    of displacement toward the lifter
    :param x:
    :param y:
    :param catch_idx:
    :return:
    """
    #cut all values after Y_catch height
    y_catch_height = y[catch_idx]
    i=0
    while y[i]<y_catch_height:
        i+=1
    # x coordinates till catch
    x_tc = x[:i]
    idx_x_max_toward = np.argmax(x_tc)
    return idx_x_max_toward

def find_idx_most_anterior_x(x,idx_y_max,idx_x1):
    """
    Finds index of most anterior position between X1 and Ymax in horizontal direction
    :param x:
    :param idx_y_max:
    :param idx_x1:
    :return:
    int, index
    """
    #get portion of x between X1 and Y_max
    x_bmt = x[idx_x1:idx_y_max]
    idx_x2 = np.argmax(x_bmt)
    return idx_x2

def calc_parameters_of_lift(x,y,pixel_size,trial_idx,trial_start,trial_end):
    """

    :return:
    dictionary,
    """
    d={"trial_idx":trial_idx,"trial_start":trial_start, "trial_end":trial_end}
    #invert y measurments to have minimum at the plate ground level
    y_max_non_inverted = np.amax(y)
    d["barbell_initial_height_pix"]=y_max_non_inverted
    y = invert_y_axis(y,y_max=y_max_non_inverted)
    # convert pixel into mm
    y_mm = convert_pix_in_mm(y, pixel_size=pixel_size)
    x_mm = convert_pix_in_mm(x,pixel_size=pixel_size)
    d["pixel_size"]=pixel_size
    #find initial x barbell plate center position
    x_0 = x_mm[0]
    y_0 = 0
    d["barbell_initial_x_coord"]=x_0
    #find Y_max
    y_max_idx = find_y_max_idx(y_mm)
    y_max = y_mm[y_max_idx]
    d["Y_max"]=y_max
    d["Y_max_idx"] = y_max_idx+trial_start
    #find Y_catch
    y_catch_idx = find_idx_of_catch(y,y_max_idx)
    y_catch = y_mm[y_catch_idx]
    d["Y_catch"] = y_catch
    d["Y_catch_idx"] = y_catch_idx+trial_start
    #find X1
    x1_idx = find_idx_max_x_displacement_toward(x,y,y_catch_idx)
    x1 = x_mm[x1_idx]-x_0
    y1 = y_mm[x1_idx]
    d["X1"] = x1
    d["X1_idx"] = x1_idx+trial_start
    #find X2
    x2_idx = find_idx_most_anterior_x(x,y_max_idx,x1_idx)
    x2 = x_mm[x2_idx]-x_0
    d["X2"] = x2
    d["X2_idx"] = x2_idx+trial_start
    #find X_loop
    x_loop = x_mm[y_catch_idx] - x_mm[x2_idx]
    d["x_loop"] = x_loop
    #find Y_drop
    y_drop = y_max - y_catch
    d["y_drop"] = y_drop
    # find angle angle relative to vertical reference line from start position to X1  tg(a)=x_1/y1, a = atan(x1/y1)
    theta = math.atan(x1/y1)
    d["theta_1"] = theta
    d["peak_vertical_velocity"] = None
    d["Y_catch/Y_max"] = y_catch/y_max
    return d

def analyze_csv(fpath):
    """

    :param fpath:
    :return:
    """
    x,y,time = read_tracking_par_from_csv(fpath)
    y_max = np.amax(y)
    y = invert_y_axis(y, y_max=y_max)
    end_of_trial_idxs = find_end_of_trail(y)
    pixel_size = find_pixel_size_mm(x1, y1, x2, y2, plate_diameter_mm)
    print(end_of_trial_idxs)
    for trial_idx,trial_window in enumerate(end_of_trial_idxs):
        d = calc_parameters_of_lift(x,y,pixel_size,trial_idx,trial_window[0],trial_window[1])
        print(d)

if __name__=="__main__":
    fpath = r"C:\NewData\Projects\Barbell\barbell-path-tracker\Result\Tracking\IMG_9200.csv"
    analyze_csv(fpath)
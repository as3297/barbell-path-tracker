import numpy as np
import math
from util import read_tracking_par_from_csv,open_video_file,save_json,save_velocity_par
import cv2
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt



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
    vel = np.array(calc_diff(z))/frame_length
    return vel

def calc_diff(z):
    """Calculate the difference between the neighbouring positions to find relative velocity"""
    vel = [z[i] - z[i - 1] for i in range(1, len(z))]
    return vel

def find_the_ground_from_hist(y):
    """
    Calculates the maximum floor value of barbell from the histogram of barbell center position
    The based on the histogram of y coordinate histogram we assume that the barbell spends the longest time on the ground,so the histogram suppose to have the ground value at ground
    :param y:
    :return:
    int, index of floor value
    """
    nb_bins = len(y)//20
    hist, edges = np.histogram(y,nb_bins)
    max_idx = np.argmax(hist)
    floor_value = edges[max_idx+1]
    return floor_value

def find_end_of_trail(y):
    """
    Find the end of trail based on the y coordinate position, and barbell floor value
    if y if lower then the ground value and than it gets returned to the ground level we assume this to be the start
     and the end of the lift.
    :param y:
    :return:
    [], list of tuples of start and end for each lift in the video
    """
    #cut video into trials
    # if the barbell is below the threshold and the average velocity over 3 frames is negative then the trial is over
    floor_value = find_the_ground_from_hist(y)
    trails= []
    end_lift_prev = 0
    while end_lift_prev>=0:
        try:
            start_lift,end_lift = find_start_end_lift(y[end_lift_prev:],floor_value)
            start_lift = end_lift_prev + start_lift
            end_lift_prev = end_lift_prev + end_lift
            trails.append([start_lift,end_lift_prev])
        except:
            print("No more trials")
            end_lift_prev = -1
    if len(trails)==0:
        trails=None
    return trails

def find_start_end_lift(y,floor_level):
    """
    Find start and end of list, based on the barbell floor level
    :param y: coordinate list
    :param floor_level: floor level of y coordinate in pixels
    :return:
    """
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
    Find height of catch idx, based on the velocity. We consider the moment of catch the momement
    when the velocity goes up after barbell reach Y_max
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
    Calculate maximum barbell height (Ymax), based on maximum velocity.
    Assuming that the maximum velocity is reached moment before barbell reaches the top,
     the moment before the velocity becomes negative if considered to be the top position
    :param y: [], list of y coordinates converted into mm
    :param method: str, default "peak_velocity"
    :return:
    int
    """
    vel_y = calc_diff(y)
    #find the velocity extrema
    if method=="peak_velocity":
        peak_vel_idx = np.argmax(vel_y)
    for i in range(peak_vel_idx,len(y)):
        if np.mean(vel_y[i:i+1])<0:
            peak_idx = i
            break
    return peak_idx

def find_idx_max_x_displacement_toward(x,y,catch_idx):
    """
    X1, net horizontal displacement from start position to most rearward position during first phase
    of displacement toward the lifter. To find X1 position we find the maximum value of X based on the trajectory
    of the first phase of catch before the barbell reaches the catch height.
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
    Finds index of X2, as the most anterior position between X1 and Ymax in horizontal direction
    :param x:
    :param idx_y_max:
    :param idx_x1:
    :return:
    int, index
    """
    #get portion of x between X1 and Y_max
    x_bmt = x[idx_x1:idx_y_max]
    idx_x2 = np.argmin(x_bmt)
    return idx_x2 + idx_x1

def calc_parameters_of_lift(x_org,y_org,pixel_size,trial_start,trial_end, y_max_non_inverted,frame_lengh):
    """
    Calculates the kinematic parameters of the lifting trial based on the barbell trajectory
    :return:
    dictionary,
    """
    y = y_org[trial_start:trial_end]
    x = x_org[trial_start:trial_end]
    d={"trial_start":trial_start, "trial_end":trial_end}
    #invert y measurments to have minimum at the plate ground level

    d["barbell_initial_height_pix"]=y_max_non_inverted
    # convert pixel into mm
    y_mm = convert_pix_in_mm(y, pixel_size=pixel_size)
    x_mm = convert_pix_in_mm(x,pixel_size=pixel_size)
    d["pixel_size,mm"]=pixel_size
    #find initial x barbell plate center position
    x0 = x_mm[0]
    d["barbell_initial_x_coord,mm"]=x0
    #find Y_max
    y_max_idx = find_y_max_idx(y)
    y_max = y_mm[y_max_idx]
    d["Y_max,mm"]=y_max
    d["Y_max_idx"] = y_max_idx+trial_start
    #find Y_catch
    y_catch_idx = find_idx_of_catch(y,y_max_idx)
    y_catch = y_mm[y_catch_idx]
    d["Y_catch,mm"] = y_catch
    d["Y_catch_idx"] = y_catch_idx+trial_start
    #find X1
    x1_idx = find_idx_max_x_displacement_toward(x,y,y_catch_idx)
    x1 = x_mm[x1_idx]-x0
    y1 = y_mm[x1_idx]
    d["X1,mm"] = x1
    d["X1_idx"] = x1_idx+trial_start
    #find X2
    x2_idx = find_idx_most_anterior_x(x,y_max_idx,x1_idx)
    x2 = x_mm[x2_idx]-x_mm[x1_idx]
    d["X2,mm"] = x2
    d["X2_idx"] = x2_idx+trial_start
    #find X_loop
    x_loop = x_mm[y_catch_idx] - x_mm[x2_idx]
    d["x_loop,mm"] = x_loop
    #find Y_drop
    y_drop = y_max - y_catch
    d["y_drop,mm"] = y_drop
    # find angle angle relative to vertical reference line from start position to X1  tg(a)=x_1/y1, a = atan(x1/y1)
    theta = math.degrees(math.atan(x1/y1))
    d["theta"] = theta
    d["peak_vertical_velocity, mm/sec"] = np.max(calc_velocity(y,frame_lengh))
    d["Y_catch/Y_max"] = y_catch/y_max
    return d

def analyse_csv(fpath,fpath_video):
    """
    Calculate the kinematic parameters in a video of each lifting session
    :param fpath:
    :return:
    """
    x,y,time,pixel_size = read_tracking_par_from_csv(fpath)
    y_max = np.amax(y)
    y = invert_y_axis(y, y_max=y_max)
    end_of_trial_idxs = find_end_of_trail(y)
    print(end_of_trial_idxs)
    trail_d = {}
    for trial_idx,trial_window in enumerate(end_of_trial_idxs):
        d = calc_parameters_of_lift(x,y,pixel_size,trial_window[0],trial_window[1],y_max,time[0])
        trail_d["trial_idx_{}".format(trial_idx)] = d
        plot_(x,y,d,trial_idx,trial_window[0],trial_window[1],fpath[:-4])
        print(d)
        draw_on_video(fpath_video,fpath, x, y, d, trial_idx, trial_window[0],trial_window[1])
        save_velocity_par(fpath[:-4]+"_wp_trial{}".format(trial_idx)+ ".csv", d)
    #save_json(fpath[:-4]+".json",trail_d)


def plot_(x,y,d,trial_idx,trial_start,trial_end,fpath):
    """
    Plot the detected key point over one trial trajectory
    :param x:
    :param y:
    :param d:
    :param trial_idx:
    :param trial_start:
    :param trial_end:
    :param fpath:
    :return:
    """
    plt.figure(figsize = (720//90,1280//90))
    plt.title("Trial {}".format(trial_idx))
    plt.plot(x[trial_start:trial_end],y[trial_start:trial_end])
    idx = d["Y_max_idx"]
    plt.text(x[idx],y[idx],"Ymax")
    plt.plot(x[idx],y[idx],marker='o', color='r')
    init_barbell_pos = (trial_end-trial_start)*[x[trial_start]]
    plt.plot(init_barbell_pos, y[trial_start:trial_end],linestyle='--', color='g')
    idx = d["Y_catch_idx"]
    plt.text(x[idx], y[idx], "Ycatch")
    plt.plot(x[idx], y[idx], marker='o', color='r')
    idx = d["X1_idx"]
    plt.text(x[idx], y[idx], "X1")
    plt.plot(x[idx], y[idx], marker='o', color='r')
    idx = d["X2_idx"]
    plt.text(x[idx], y[idx], "X2")
    plt.plot(x[idx], y[idx], marker='o', color='r')
    plt.grid()
    plt.xlabel("X,pixel")
    plt.ylabel("Y,pixel")
    plt.savefig(fpath+"_wp_trial{}".format(trial_idx)+".png")

    plt.show()

def draw_on_video(fpath_video,fpath,x,y,d,trial_idx, start_trial, end_trial):
    """
    Draw key points on video of one trial
    :param fpath_video:
    :param fpath:
    :param x:
    :param y:
    :param d:
    :param trial_idx:
    :param start_trial:
    :param end_trial:
    :return:
    """
    cap, nb_frames = open_video_file(fpath_video)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    frame_width = 720//2#int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = 1280//2#int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x = [int(x[i]) for i in range(len(x))]
    y_max_non_inverted = d["barbell_initial_height_pix"]
    y = [int(y_max_non_inverted-y[i]) for i in range(len(y))]
    video_res_writer = cv2.VideoWriter(fpath[:-4]+"_wp_trial{}".format(trial_idx)+ ".mov",fourcc, 15, (frame_width,frame_height))
    for i in range(nb_frames):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (frame_width,frame_height), interpolation=cv2.INTER_CUBIC)
        if i>start_trial and i<end_trial:
            #start of trial
            cv2.putText(frame, "Xst", (x[start_trial], y[start_trial]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255, 0), 2)
            # coordinate X1
            idx = d["X1_idx"]
            if i>idx:
                cv2.putText(frame, "X1",(x[idx],y[idx]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            # coordinate X2
            idx = d["X2_idx"]
            if i>idx:
                cv2.putText(frame, "X2", (x[idx], y[idx]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            # coordinate Y_max
            idx = d["Y_max_idx"]
            if i>idx:
                cv2.putText(frame, "Y_max", (x[idx], y[idx]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            # coordinate Y_catch
            idx = d["Y_catch_idx"]
            if i>idx:
                cv2.putText(frame, "Y_catch", (x[idx], y[idx]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            # end of trial
            if i>end_trial-1:
                cv2.putText(frame, "Xend", (x[end_trial], y[end_trial]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255,0), 2)
            # Write the processed frame to the video
            for j in range(start_trial,i):
                # path of center point
                cv2.line(frame, (x[j-1],y[j-1]), (x[j],y[j]), (255,0,0), 2)
            video_res_writer.write(frame)
    # Release the VideoWriter and close the output file
    video_res_writer.release()
    cv2.destroyAllWindows()



if __name__=="__main__":
    fpath_csv = r"C:\NewData\Projects\Barbell\barbell-path-tracker\Result\Tracking\IMG_9203.csv"
    fpath_org_video = r"C:\NewData\Projects\Barbell\data\initial_test\IMG_9203.mov"
    analyse_csv(fpath_csv,fpath_org_video)

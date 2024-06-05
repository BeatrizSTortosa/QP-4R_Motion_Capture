# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Beatriz Soriano Tortosa
# 02/06/24
# QP-4R_Motion Capture
# Parts of this code were adapted from: https://github.com/python-dontrepeatyourself/Color-Based-Object-Detection-with-OpenCV-and-Python/blob/main/color_detection_video.py, Accessed: 25/02/24.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


# Import libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.signal import argrelextrema
import matplotlib.collections as mcoll
import matplotlib.path as mpath
from matplotlib import interactive
from matplotlib import rc
import matplotlib as mpl
import math

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ INITIALISATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# Colours to track
capture_green = True
capture_blue = True
capture_red = True
capture_orange = True

# Initialize the video capture object
cap = cv2.VideoCapture('QP-4R_Video.MOV') # Replace with right directory containing video file

# Grab the width, height, and fps of the frames in the video stream.
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize the FourCC and a video writer object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('QP-4R_Video_Output.avi', fourcc, fps, (frame_width, frame_height))    # Replace with directory for output video

# Initialse arrays
x_blue = []
y_blue = []
x_green = []   
y_green = []   
x_red = []
y_red = []
x_orange = []
y_orange = []           
frames = []
green_frames = []
red_frames = []
blue_frames = []
frame_number = 0

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ COLOUR TRACKING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

while True:
    ret, frame = cap.read()
    frame_number += 1
    frames.append(frame_number)
    
    if not ret:
        print("There are no more frames to read, exiting...")
        break

    # Convert from BGR to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define lower and upper limits for each of the colours
    lower_blue = np.array([90, 64, 91])
    upper_blue = np.array([108, 255, 177])
    lower_green = np.array([35, 86, 80])
    upper_green = np.array([77, 224, 161])
    lower_red = np.array([102, 44, 47])
    upper_red = np.array([179, 255, 156])
    lower_orange = np.array([0, 47, 118])
    upper_orange = np.array([18, 237, 245])

    # Create masks to detect the colors
    blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)
    blue_mask = cv2.erode(blue_mask, None, iterations=3)
    blue_mask = cv2.dilate(blue_mask, None, iterations=3)
    green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)
    green_mask = cv2.erode(green_mask, None, iterations=3)
    green_mask = cv2.dilate(green_mask, None, iterations=3)
    red_mask = cv2.inRange(hsv_frame, lower_red, upper_red)
    red_mask = cv2.erode(red_mask, None, iterations=3)
    red_mask = cv2.dilate(red_mask, None, iterations=3)
    orange_mask = cv2.inRange(hsv_frame, lower_orange, upper_orange)
    orange_mask = cv2.erode(orange_mask, None, iterations=3)
    orange_mask = cv2.dilate(orange_mask, None, iterations=3)

    # Find contours and draw the minimum enclosing bounding circles
    if capture_blue:
        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in blue_contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            x_blue.append(int(x))
            y_blue.append(int(y))
            cv2.circle(frame, center, radius, (255, 0, 0), 2)   # Blue 
            blue_frames.append(frame_number)

    if capture_green:
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in green_contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            x_green.append(int(x))
            y_green.append(int(y))
            green_radius = radius
            cv2.circle(frame, center, radius, (0, 255, 0), 2)   # Green 
            green_frames.append(frame_number)

    if capture_red:
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in red_contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            if radius > 7:                          # Ensure detected object is above a certain radius        
                x_red.append(int(x))
                y_red.append(int(y))
                cv2.circle(frame, center, radius, (0, 0, 255), 2)   # Red 
                red_frames.append(frame_number)

    if capture_orange:
        orange_contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in orange_contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            if y - radius <= frame_height * 0.5:    # Ensure detected object is not in the bottom 50% of the screen
                x_orange.append(int(x))
                y_orange.append(int(y))
                cv2.circle(frame, center, radius, (0, 165, 255), 2)   # Orange   

    # Write the frame to the output file
    output.write(frame)
      
    cv2.imshow('frame', cv2.resize(frame, None, fx=1, fy=1))

    if cv2.waitKey(1) == 27:    # Exit if escape is pressed
        break

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ OBJECT SORTING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# When there are multiple objects of the same colour detected, they need to be allocated to the right array.

# Create arrays to sort objects ('1' is the first to appear on-screen)
x_blue_1 = []
y_blue_1 = []
x_blue_2 = []
y_blue_2 = []
x_red_1 = []
y_red_1 = []
x_red_2 = []
y_red_2 = []
x_orange_1 = []
y_orange_1 = [] 
x_orange_2 = []
y_orange_2 = []  

red_1_frames = []
red_2_frames = []
blue_1_frames = []
blue_2_frames = []

# Sort orange colour
x_orange_1.append(x_orange[0])
x_orange_2.append(0)
y_orange_1.append(y_orange[0])
y_orange_2.append(0)
distances = []
for i in range(1, len(x_orange)):
    if abs(x_orange[i] - x_orange[i-1]) > 70 and x_orange_1[i-1] != 0:     # If switch object and previously in 1
        x_orange_2.append(x_orange[i])
        x_orange_1.append(0)
        y_orange_2.append(y_orange[i])
        y_orange_1.append(0)
        distances.append(abs(x_orange[i] - x_orange[i-1]))
    elif abs(x_orange[i] - x_orange[i-1]) > 70 and x_orange_1[i-1] == 0:   # If switch object and previously in 2
        x_orange_1.append(x_orange[i])
        x_orange_2.append(0)
        y_orange_1.append(y_orange[i])
        y_orange_2.append(0)
        distances.append(abs(x_orange[i] - x_orange[i-1]))
    elif abs(x_orange[i] - x_orange[i-1]) < 70 and x_orange_1[i-1] != 0:   # If didn't switch object and previously in 1
        x_orange_1.append(x_orange[i])
        x_orange_2.append(0)
        y_orange_1.append(y_orange[i])
        y_orange_2.append(0)
    elif abs(x_orange[i] - x_orange[i-1]) < 70 and x_orange_1[i-1] == 0:   # If didn't switch object and previously in 2
        x_orange_2.append(x_orange[i])
        x_orange_1.append(0)
        y_orange_2.append(y_orange[i])
        y_orange_1.append(0)

# Remove all elements with value 0 from the arrays
x_orange_1 = [x for x in x_orange_1 if x != 0]
x_orange_2 = [x for x in x_orange_2 if x != 0]
y_orange_1 = [x for x in y_orange_1 if x != 0]
y_orange_2 = [x for x in y_orange_2 if x != 0]

# Sort blue colour
x_blue_1.append(x_blue[0])
x_blue_2.append(0)
y_blue_1.append(y_blue[0])
y_blue_2.append(0)
blue_1_frames.append(blue_frames[0])
for i in range(1, len(x_blue)):
    if abs(x_blue[i] - x_blue[i-1]) > 100 and x_blue_1[i-1] != 0:     # If switch object and previously in 1
        x_blue_2.append(x_blue[i])
        x_blue_1.append(0)
        y_blue_2.append(y_blue[i])
        y_blue_1.append(0)
        blue_2_frames.append(blue_frames[i])
    elif abs(x_blue[i] - x_blue[i-1]) > 100 and x_blue_1[i-1] == 0:   # If switch object and previously in 2
        x_blue_1.append(x_blue[i])
        x_blue_2.append(0)
        y_blue_1.append(y_blue[i])
        y_blue_2.append(0)
        blue_1_frames.append(blue_frames[i])
    elif abs(x_blue[i] - x_blue[i-1]) < 100 and x_blue_1[i-1] != 0:   # If didn't switch object and previously in 1
        x_blue_1.append(x_blue[i])
        x_blue_2.append(0)
        y_blue_1.append(y_blue[i])
        y_blue_2.append(0)
        blue_1_frames.append(blue_frames[i])
    elif abs(x_blue[i] - x_blue[i-1]) < 100 and x_blue_1[i-1] == 0:   # If didn't switch object and previously in 2
        x_blue_2.append(x_blue[i])
        x_blue_1.append(0)
        y_blue_2.append(y_blue[i])
        y_blue_1.append(0)
        blue_2_frames.append(blue_frames[i])

# Remove all elements with value 0 from the arrays
x_blue_1 = [x for x in x_blue_1 if x != 0]
x_blue_2 = [x for x in x_blue_2 if x != 0]
y_blue_1 = [x for x in y_blue_1 if x != 0]
y_blue_2 = [x for x in y_blue_2 if x != 0]

# Sort red colour
x_red_1.append(x_red[0])
x_red_2.append(0)
y_red_1.append(y_red[0])
y_red_2.append(0)
red_1_frames.append(red_frames[0])
for i in range(1, len(x_red)):
    if abs(x_red[i] - x_red[i-1]) > 19.1 and x_red_1[i-1] != 0:     # If switch object and previously in 1
        x_red_2.append(x_red[i])
        x_red_1.append(0)
        y_red_2.append(y_red[i])
        y_red_1.append(0)
        red_2_frames.append(red_frames[i])
    elif abs(x_red[i] - x_red[i-1]) > 19.1 and x_red_1[i-1] == 0:   # If switch object and previously in 2
        x_red_1.append(x_red[i])
        x_red_2.append(0)
        y_red_1.append(y_red[i])
        y_red_2.append(0)
        red_1_frames.append(red_frames[i])
    elif abs(x_red[i] - x_red[i-1]) < 19.1 and x_red_1[i-1] != 0:   # If didn't switch object and previously in 1
        x_red_1.append(x_red[i])
        x_red_2.append(0)
        y_red_1.append(y_red[i])
        y_red_2.append(0)
        red_1_frames.append(red_frames[i])
    elif abs(x_red[i] - x_red[i-1]) < 19.1 and x_red_1[i-1] == 0:   # If didn't switch object and previously in 2
        x_red_2.append(x_red[i])
        x_red_1.append(0)
        y_red_2.append(y_red[i])
        y_red_1.append(0)
        red_2_frames.append(red_frames[i])

# Remove all elements with value 0 from the arrays
x_red_1 = [x for x in x_red_1 if x != 0]
x_red_2 = [x for x in x_red_2 if x != 0]
y_red_1 = [x for x in y_red_1 if x != 0]
y_red_2 = [x for x in y_red_2 if x != 0]
    
# Frames to time conversion
green_times = []
red_1_times = []
red_2_times = []
blue_1_times = []
blue_2_times = []

total_time = frames[-1] / fps
for i in green_frames:
    green_time = i / fps
    green_times.append(green_time)
for i in red_1_frames:
    red_1_time = i / fps
    red_1_times.append(red_1_time)
for i in red_2_frames:
    red_2_time = i / fps
    red_2_times.append(red_2_time)
for i in blue_1_frames:
    blue_1_time = i / fps
    blue_1_times.append(blue_1_time)
for i in blue_2_frames:
    blue_2_time = i / fps
    blue_2_times.append(blue_2_time)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CALIBRATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    
avg_distance = sum(distances)/len(distances)            # Distance between the orange hip joints
calibration_distance = 104                              # Actual distance between the joints in cm
calibration_factor = calibration_distance/avg_distance  # Calibration factor

x_green = [i * calibration_factor for i in x_green]
y_green = [i * calibration_factor for i in y_green]
x_blue_1 = [i * calibration_factor for i in x_blue_1]
y_blue_1 = [i * calibration_factor for i in y_blue_1]
x_blue_2 = [i * calibration_factor for i in x_blue_2]
y_blue_2 = [i * calibration_factor for i in y_blue_2]
x_red_1 = [i * calibration_factor for i in x_red_1]
y_red_1 = [i * calibration_factor for i in y_red_1]
x_red_2 = [i * calibration_factor for i in x_red_2]
y_red_2 = [i * calibration_factor for i in y_red_2]
x_orange_1 = [i * calibration_factor for i in x_orange_1]
y_orange_1 = [i * calibration_factor for i in y_orange_1] 
x_orange_2 = [i * calibration_factor for i in x_orange_2]
y_orange_2 = [i * calibration_factor for i in y_orange_2]  

# Arrays with (0, 0) at their start point
origin_y_green = [-(y - y_green[0]) for y in y_green]
origin_y_red_1 = [-(y - y_red_1[0]) for y in y_red_1]
origin_y_blue_1 = [-(y - y_blue_1[0]) for y in y_blue_1]
origin_y_red_2 = [-(y - y_red_2[0]) for y in y_red_2]
origin_y_blue_2 = [-(y - y_blue_2[0]) for y in y_blue_2]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DETRENDING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def detrend_data(color_times, x_values, y_values):
    trend_x = np.polyfit(color_times, x_values, 4)
    trend_line_x = np.polyval(trend_x, color_times)
    x_values_detrended = x_values - trend_line_x
    return x_values_detrended, trend_line_x

x_green_detrended, trend_line_x_green = detrend_data(green_times, x_green, origin_y_green)
x_red_1_detrended, trend_line_x_red_1 = detrend_data(red_1_times, x_red_1, origin_y_red_1)
x_red_2_detrended, trend_line_x_red_2 = detrend_data(red_2_times, x_red_2, y_red_2)
x_blue_1_detrended, trend_line_x_blue_1 = detrend_data(blue_1_times, x_blue_1, origin_y_blue_1)
x_blue_2_detrended, trend_line_x_blue_2 = detrend_data(blue_2_times, x_blue_2, y_blue_2)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PLOTTING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

interactive(True)   

# Text formatting
plt.rc('text', usetex=True)
plt.rcParams.update({
    'text.usetex' : True
})
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Computer Modern",
    "font.size": 10
})
plt.rc('text.latex', preamble=r'\usepackage{cmbright}')

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SCATTER PLOT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# Scatter plot of x-y coordinates
plt.scatter(x_blue_1, y_blue_1, color='blue', marker='o', s=10, label='Blue 1')
plt.scatter(x_blue_2, y_blue_2, color='blue', marker='o', s=10, facecolors='white', edgecolors='blue', label='Blue 2')
plt.scatter(x_green, y_green, color='green', marker='o', s=10, label='Green')
plt.scatter(x_red_1, y_red_1, color='red', marker='o', s=10, label='Red 1')
plt.scatter(x_red_2, y_red_2, color='red', marker='o', s=10, facecolors='white', edgecolors='red', label='Red 2')
plt.scatter(x_orange_1, y_orange_1, color='orange', marker='o', s=10, label='Orange 1')
plt.scatter(x_orange_2, y_orange_2, color='orange', marker='o', s=10, facecolors='white', edgecolors='orange', label='Orange 2')
plt.title('Scatter Plot of Motion Tracking')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.gca().invert_yaxis() 
plt.legend()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  LINE PLOT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# Line plot of x-y coordinates
plt.figure(figsize=(7,2.8))
plt.plot(x_green, y_green, color='#1cb01a', linewidth=1, label='Crank Joint')
plt.plot(x_orange_1, y_orange_1, color='#fb922b', linewidth=1, label='Hip Joint 1')  
plt.plot(x_orange_2, y_orange_2, color='#fb922b', linestyle='--', linewidth=1, label='Hip Joint 2')  
plt.plot(x_blue_1, y_blue_1, color='#2578dd', linewidth=1, label='Rocker Joint 1')
plt.plot(x_blue_2, y_blue_2, color='#2578dd', linewidth=1, linestyle='--', label='Rocker Joint 2')  
plt.plot(x_red_1, y_red_1, color='#ec0604', linewidth=1, label='Foot 1')
plt.plot(x_red_2, y_red_2, color='#ec0604', linewidth=1, linestyle='--', label='Foot 2')  
plt.title('Line Plot of Motion Tracking')
plt.xlabel('Horizontal Position (mm)')
plt.ylabel('Vertical Position (mm)')
plt.gca().invert_yaxis()
plt.legend(loc ="lower left", fontsize="8")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  TIME SERIES PLOTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# Green
fig, axs = plt.subplots(3, 1, figsize=(8, 6))
axs[0].plot(green_times, x_green, color='#1cb01a', label='Green (All Data)')
axs[0].plot(green_times, trend_line_x_green, color='black', linestyle='--', label='Trend Line (Green)')
axs[0].set_title('Crank Joint Detrending Timeseries')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Horizontal Position (mm)')
axs[0].set_xlim(0, total_time)  
axs[1].plot(green_times, origin_y_green, color='#1cb01a', label='Green (All Data)')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Vertical Position (mm)')
axs[1].set_xlim(0, total_time)
axs[1].invert_yaxis()
axs[2].plot(green_times, x_green_detrended, color='#1cb01a', label='Green (All Data)')
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('Detrended Horizontal Position (mm)')
axs[2].set_xlim(0, total_time)  
plt.tight_layout()

# Red
fig, axs = plt.subplots(3, 1, figsize=(8, 6))
axs[0].plot(red_1_times, x_red_1, color='#ec0604', label='Red 1')
axs[0].set_title('Foot Detrending Timeseries')
# axs[0].plot(red_2_times, x_red_2, color='#ec0604', linestyle='--', label='Red 2')
axs[0].plot(red_1_times, trend_line_x_red_1, color='black', linestyle='--', label='Trend Line (Red 1)')
# axs[0].plot(red_2_times, trend_line_x_red_2, color='black', linestyle='--', label='Trend Line (Red 2)')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Horizontal Position (mm)')
axs[0].set_xlim(0, total_time)  
axs[1].plot(red_1_times, origin_y_red_1, color='#ec0604', label='Red 1')
# axs[1].plot(red_2_times, y_red_2, color='#ec0604', linestyle='--', label='Red 2')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Vertical Position (mm)')
axs[1].set_xlim(0, total_time)
axs[1].invert_yaxis()
axs[2].plot(red_1_times, x_red_1_detrended, color='#ec0604', label='Red 1')
# axs[2].plot(red_2_times, x_red_2_detrended, color='#ec0604', linestyle='--', label='Red 2')
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('Detrended Horizontal Position (mm)')
axs[2].set_xlim(0, total_time)  
plt.tight_layout()

# Blue
fig, axs = plt.subplots(3, 1, figsize=(8, 6))
axs[0].plot(blue_1_times, x_blue_1, color='#2578dd', label='Blue 1')
axs[0].set_title('Rocker Joint Detrending Timeseries')
# axs[0].plot(blue_2_times, x_blue_2, color='#2578dd', label='Blue 2')
axs[0].plot(blue_1_times, trend_line_x_blue_1, color='black', linestyle='--', label='Trend Line (Blue 1)')
# axs[0].plot(blue_2_times, trend_line_x_blue_2, color='black', linestyle='--', label='Trend Line (Blue 2)')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Horizontal Position (mm)')
axs[0].set_xlim(0, total_time)  
axs[1].plot(blue_1_times, origin_y_blue_1, color='#2578dd', label='Blue 1')
# axs[1].plot(blue_2_times, y_blue_2, color='#2578dd', linestyle='--', label='Blue 2')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Vertical Position (mm)')
axs[1].set_xlim(0, total_time)
axs[1].invert_yaxis()
axs[2].plot(blue_1_times, x_blue_1_detrended, color='#2578dd', label='Blue 1')
# axs[2].plot(blue_2_times, x_blue_2_detrended, color='#2578dd', linestyle='--', label='Blue 2')
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('Detrended Horizontal Position (mm)')
axs[2].set_xlim(0, total_time)  
plt.tight_layout()

# Overlayed detrending plot for the front foot
fig = plt.figure(figsize=(5, 2.5))  
ax1 = fig.add_subplot(111)
ax1.plot(red_1_times, x_red_1, color='black', label='Original Data')
ax1.plot(red_1_times, trend_line_x_red_1, color='black', linestyle='--', label='Quartic Trend')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Horizontal Position (mm)')
ax1.plot(red_1_times, x_red_1_detrended, color='black', linestyle=':', label='Detrended Data')
ax1.set_xlim(0, total_time) 
plt.legend(loc="upper left")
ax1.set_xlim(1.43, 16.2)
ax1.set_ylim(-105, 1000)
plt.tight_layout()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  DETRENDED PLOTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# Colour gradients in plot
def colorline(
    x, y, z=None, cmap=plt.get_cmap('binary'),
        linewidth=2):
    z = np.linspace(0.0, 1.0, len(x))
    z = np.asarray(z)
    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, linewidth=linewidth, alpha=1)
    ax = plt.gca()
    ax.add_collection(lc)
    return lc
def make_segments(x, y):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

# Green
fig = plt.figure()
ax1 = fig.add_subplot(111) 
ax1.set_xlim(-75, 60)
ax1.set_ylim(-20, 70)
ax1.set_aspect('equal')
cbar = plt.colorbar(colorline(x_green_detrended,origin_y_green, cmap="Greens", linewidth=2,), shrink = 0.65)
cbar.set_ticks(ticks=[0, 1], labels=['Start', 'End'], fontsize="15")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.yticks(np.arange(-20, 80, 20))
plt.title('Detrended XY-coordinate Plot for Crank')
plt.xlabel('Detrended Horizontal Position (mm)', fontsize="15")
plt.ylabel('Vertical Position (mm)', fontsize="15")

# Red 1
fig = plt.figure()
ax1 = fig.add_subplot(111) 
ax1.set_xlim(-90, 85)
ax1.set_ylim(-40, 60)
ax1.set_aspect('equal')
cbar = plt.colorbar(colorline(x_red_1_detrended,origin_y_red_1, cmap="Reds", linewidth=2), shrink = 0.65)
cbar.set_ticks(ticks=[0, 1], labels=['Start', 'End'], fontsize="15")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.yticks(np.arange(-40, 80, 20))
plt.title('Detrended XY-coordinate Plot for Front Foot')
plt.xlabel('Detrended Horizontal Position (mm)', fontsize="15")
plt.ylabel('Vertical Position (mm)', fontsize="15")

# Red 2
fig = plt.figure()
ax1 = fig.add_subplot(111) 
ax1.set_xlim(-90, 85)
ax1.set_ylim(-40, 60)
ax1.set_aspect('equal')
cbar = plt.colorbar(colorline(x_red_2_detrended,origin_y_red_2, cmap="Reds", linewidth=2), shrink = 0.65)
cbar.set_ticks(ticks=[0, 1], labels=['Start', 'End'], fontsize="15")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.yticks(np.arange(-40, 80, 20))
plt.title('Detrended XY-coordinate Plot for Hind Foot')
plt.xlabel('Detrended Horizontal Position (mm)', fontsize="15")
plt.ylabel('Vertical Position (mm)', fontsize="15")

# Blue 1
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_xlim(-80, 60)
ax1.set_ylim(-40, 40)
ax1.set_aspect('equal')
cbar = plt.colorbar(colorline(x_blue_1_detrended,origin_y_blue_1, cmap="Blues", linewidth=2), shrink = 0.65)
cbar.set_ticks(ticks=[0, 1], labels=['Start', 'End'], fontsize="15")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.yticks(np.arange(-40, 60, 20))
plt.title('Detrended XY-coordinate Plot for Front Rocker')
plt.xlabel('Detrended Horizontal Position (mm)', fontsize="15")
plt.ylabel('Vertical Position (mm)', fontsize="15")

# Blue 2
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_xlim(-80, 60)
ax1.set_ylim(-40, 40)
ax1.set_aspect('equal')
cbar = plt.colorbar(colorline(x_blue_2_detrended,origin_y_blue_2, cmap="Blues", linewidth=2), shrink = 0.65)
cbar.set_ticks(ticks=[0, 1], labels=['Start', 'End'], fontsize="15")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.yticks(np.arange(-40, 60, 20))
plt.title('Detrended XY-coordinate Plot for Hind Rocker')
plt.xlabel('Detrended Horizontal Position (mm)', fontsize="15")
plt.ylabel('Vertical Position (mm)', fontsize="15")

interactive(False)      
plt.show()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  OBJECT SORTING PLOT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

"""
# This plot was used to verify the object sorting is correct
# To generate this plot first comment out: 1) other plots, 2) calibration, 3) detrending and 4) removal of 0s from arrays

plt.figure(figsize=(10, 6))
# if capture_blue:
#     plt.plot(range(len(x_blue)), x_blue, label='Blue', color='blue', marker='o', markersize=3, linestyle='-')
#     plt.plot(range(len(x_blue_1)), x_blue_1, label='Blue_1', color='red', marker='o', markersize=3, linestyle='')
#     plt.plot(range(len(x_blue_2)), x_blue_2, label='Blue_2', color='green', marker='o', markersize=3, linestyle='')
# if capture_green:
#     plt.plot(range(len(x_green)), x_green, label='Green', color='green', marker='o', markersize=3, linestyle='-')
if capture_red:
    plt.plot(range(len(x_red)), x_red, label='Red', color='red', marker='o', markersize=3, linestyle='-')
    plt.plot(range(len(x_red_1)), x_red_1, label='Red_1', color='blue', marker='o', markersize=3, linestyle='')
    plt.plot(range(len(x_red_2)), x_red_2, label='Red_2', color='green', marker='o', markersize=3, linestyle='')
# if capture_orange:
#     plt.plot(range(len(x_orange)), x_orange, label='Orange', color='orange', marker='o', markersize=3, linestyle='-')
#     plt.plot(range(len(x_orange_1)), x_orange_1, label='Orange_1', color='red', marker='o', markersize=3, linestyle='')
#     plt.plot(range(len(x_orange_2)), x_orange_2, label='Orange_2', color='blue', marker='o', markersize=3, linestyle='')

plt.title('X-coordinate vs Frame Number')
plt.xlabel('Frame Number')
plt.ylabel('X-coordinate')
plt.legend()
plt.grid(True)
interactive(False)      
plt.show()
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  EXIT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

cap.release()
output.release()
cv2.destroyAllWindows()

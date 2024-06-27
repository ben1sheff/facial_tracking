import cv2
from picamera2 import Picamera2
from gpiozero import Motor, Button
import time
import numpy as np

# Params
dispW = 1280
dispH = 720 # Should be able to go to 2592x1944. Others: 1296, 972; 640 x 480 also works. Use standard sizes
# FPS tracker details:
font_size = 0.6
font_pos = (30, 60) # Distance from top left in pixels
font_color = (0, 100, 0)  # BGR Color
font_weight = 2

button = Button(18)
picam = Picamera2()
picam.preview_configuration.main.size = (dispW, dispH)
picam.preview_configuration.main.format = "RGB888"
picam.preview_configuration.controls.FrameRate = 30
picam.preview_configuration.align()  # Forces to standard size for speed
picam.configure("preview")  # Applies the above configurations
picam.start()

# # Let's draw a box
# box_UL = (650, 500)
# box_LR = (750, 125)
# box_color = (255, 0, 255)
# box_thick = 7 # Line width, -1 means solid

# And a circle
circle_cent = [900, 500]
circle_color = (155, 0, 155)
circle_thick = -1
circle_rad = 35
circle_v = [30, 80]
trap_flag = [False, False]
# def update_circle(center, vel, rad, edges, dt):
#     new_center = center.copy()  # [center[0] + vel[0] * dt, center[1] + vel[1] * dt]
#     new_vel = vel.copy()
#     for i in range(len(new_center)):
#         new_center[i] += vel[i] * dt
#         if new_center[i] < rad or new_center[i] >= edges[i] - rad:
#             if not trap_flag[i]:
#                 new_vel[i] = -vel[i]
#                 trap_flag[i] = True
#                 # new_center[i] += 2 * new_vel[i] * dt  # Prevent overshoot
#         else:
#             trap_flag[i] = False

#     return new_center, new_vel
 
# Let's make a little ROI
xRange = [dispW//4, 3*dispW//4]
yRange = [dispH//4, 3*dispH//4]

# How about searching an image based on color?
hue_range = [12, 26]
sat_range = [180, 255]
val_range = [10, 255]
mask_low = lambda : np.array([hue_range[0], sat_range[0], val_range[0]])
mask_high = lambda : np.array([hue_range[1], sat_range[1], val_range[1]])

# Make track bars for better color searches
cv2.namedWindow("Trackbars")
def hue_low(val):
    global hue_range
    hue_range[0] = val
def hue_high(val):
    global hue_range
    hue_range[1] = val
def sat_low(val):
    global sat_range
    sat_range[0] = val
def sat_high(val):
    global sat_range
    sat_range[1] = val
def val_low(val):
    global val_range
    val_range[0] = val
def val_high(val):
    global val_range
    val_range[1] = val
cv2.createTrackbar("Hue Min", "Trackbars", hue_range[0], 179, hue_low)
cv2.createTrackbar("Hue Max", "Trackbars", hue_range[1], 179, hue_high)
cv2.createTrackbar("Saturation Min", "Trackbars", sat_range[0], 255, sat_low)
cv2.createTrackbar("Saturation Max", "Trackbars", sat_range[1], 255, sat_high)
cv2.createTrackbar("Value Min", "Trackbars", val_range[0], 255, val_low)
cv2.createTrackbar("Value Max", "Trackbars", val_range[1], 255, val_high)



# Loop setup stuff
exit_flag = True
fps = 0  # Start
timer = time.time()
# How to leave the loop
def leave_loop():
    global exit_flag
    exit_flag = False
    cv2.destroyAllWindows()
button.when_pressed = leave_loop

while exit_flag:
    image = picam.capture_array()  # Array of pixels, [col(height)][row(width)][B,G,R]
    imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert to HSV for human intuition

    # Add frame rate
    timer2 = time.time()
    dt = timer2 - timer
    fps = 0.9 * fps + 0.1/ dt
    cv2.putText(image, str(int(np.round(fps))) + " FPS",
                font_pos, cv2.FONT_HERSHEY_SIMPLEX, font_size, 
                font_color, font_weight)
    timer = timer2

    # Add a rectangle
    # cv2.rectangle(image, box_UL, box_LR, box_color, box_thick)

    # Add a circle
    # circle_cent, circle_v = update_circle(circle_cent,circle_v, circle_rad,
    #                                       [dispW, dispH], dt)
    # cv2.circle(image, (int(circle_cent[0]), int(circle_cent[1])),
    #                    circle_rad, circle_color, circle_thick)

    # # make and separately display ROI
    # roi = image[yRange[0]:yRange[1], xRange[0]:xRange[1]]
    # cv2.imshow("ROI", roi)

    # make a mask that's white only for points in the color range we've set, then cut out black areas
    mask = cv2.inRange(imageHSV, mask_low(), mask_high()) //255
    obj_of_interest = cv2.bitwise_and(image, image, mask=mask)
    small = cv2.resize(obj_of_interest, (dispW//2, dispH//2))
    cv2.imshow("Mask", small)

    # Make contours out of the masks
    contours, junk = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        biggest_contour_ind = np.argmax([cv2.contourArea(contour) for contour in contours])
        # contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        biggest_contour = contours[biggest_contour_ind]
        cv2.drawContours(image, [biggest_contour], 0, (255, 0, 0), 3)
        x, y, w, h = cv2.boundingRect(biggest_contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255))

    # Display and exit possibility
    cv2.imshow("picam", image)
    if cv2.waitKey(1) == ord("q"):
        leave_loop()



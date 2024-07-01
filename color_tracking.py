import cv2
from picamera2 import Picamera2
from gpiozero import Motor, Button, AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory
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
# Tolerance for centering camera
camera_tolerance = dispW/30
camera_speed = 20
flip_camera = -1  # -1 indicates one horizontal flip

# A button to end our run
button = Button(18)
#  servo for swiveling
servo_pin_factory = PiGPIOFactory()
pan = AngularServo(13,# min_pulse_width = 0.0006, max_pulse_width=0.0023,
                   pin_factory = servo_pin_factory)  # positive angle is anti-clockwise
pan_angle = 0
MAX_SERVO_ANGLE = 90

# And the camera
picam = Picamera2()
picam.preview_configuration.main.size = (dispW, dispH)
picam.preview_configuration.main.format = "RGB888"
picam.preview_configuration.controls.FrameRate = 30
picam.preview_configuration.align()  # Forces to standard size for speed
picam.configure("preview")  # Applies the above configurations
picam.start()

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
servo_flag = False
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
    if flip_camera == -1:
        image = cv2.flip(image, 1)
    imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert to HSV for human intuition

    # Add frame rate
    timer2 = time.time()
    dt = timer2 - timer
    fps = 0.9 * fps + 0.1/ dt
    cv2.putText(image, str(int(np.round(fps))) + " FPS",
                font_pos, cv2.FONT_HERSHEY_SIMPLEX, font_size, 
                font_color, font_weight)
    timer = timer2

    # Detecting faces

    # make a mask that's white only for points in the color range we've set, then cut out black areas
    mask = cv2.inRange(imageHSV, mask_low(), mask_high()) //255
    obj_of_interest = cv2.bitwise_and(image, image, mask=mask)
    small = cv2.resize(obj_of_interest, (dispW//2, dispH//2))
    # cv2.imshow("Mask", small)  # Draw just the found object of interest for tuning

    # Make contours out of the masks
    contours, junk = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        biggest_contour_ind = np.argmax([cv2.contourArea(contour) for contour in contours])
        # contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        biggest_contour = contours[biggest_contour_ind]
        cv2.drawContours(image, [biggest_contour], 0, (255, 0, 0), 3)
        x, y, w, h = cv2.boundingRect(biggest_contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255))

        # Check if box is centered
        mid_x_error = x + w // 2 - dispW // 2
    else:
        mid_x_error = 0

    if cv2.waitKey(1) == ord("w"):
        servo_flag = not servo_flag
    if servo_flag:
        # Moving toward the center
        if np.abs(mid_x_error) > camera_tolerance:
            pan_angle -= mid_x_error * camera_speed // dispW * 2
            if abs(pan_angle) > MAX_SERVO_ANGLE:
                pan_angle = np.sign(pan_angle) * MAX_SERVO_ANGLE
        pan.angle = pan_angle * flip_camera


    # Display and exit possibility
    cv2.imshow("picam", image)
    if cv2.waitKey(1) == ord("q"):
        leave_loop()



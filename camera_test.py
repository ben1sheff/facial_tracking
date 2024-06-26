import cv2
from picamera2 import Picamera2
from gpiozero import Motor, Button
import time
import numpy as np

# Params
dispWH = (1280, 720) # Should be able to go to 2592x1944. Others: 1296, 972; 640 x 480 also works. Use standard sizes
# FPS tracker details:
font_size = 0.6
font_pos = (30, 60) # Distance from top left in pixels
font_color = (0, 100, 0)  # BGR Color
font_weight = 2

button = Button(18)
picam = Picamera2()
picam.preview_configuration.main.size = dispWH
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
def update_circle(center, vel, rad, edges, dt):
    new_center = center.copy()  # [center[0] + vel[0] * dt, center[1] + vel[1] * dt]
    new_vel = vel.copy()
    for i in range(len(new_center)):
        new_center[i] += vel[i] * dt
        if new_center[i] < rad or new_center[i] > edges[i] - rad:
            new_vel[i] = -vel[i]
            # new_center[i] += 2 * new_vel[i] * dt  # Prevent overshoot
    return new_center, new_vel

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
    image = picam.capture_array()  # Array of pixels, [row(width)][col(height)][B,G,R]

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

    circle_cent, circle_v = update_circle(circle_cent,
                                          circle_v, circle_rad, dispWH, dt)
    cv2.circle(image, (int(circle_cent[0]), int(circle_cent[1])),
                       circle_rad, circle_color, circle_thick)

    # Display and exit possibility
    cv2.imshow("picam", image)
    if cv2.waitKey(1) == ord("q"):
        leave_loop()



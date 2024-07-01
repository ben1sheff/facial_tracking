import cv2
import time
import numpy as np
from gpiozero import Button
from picamera2 import Picamera2

# Note, to make tensorflow lite work on bullfrog, have to use python -m pip install --upgrade tflite-support==0.4.3
from tflite_support.task import core, processor, vision
import utils

models = "efficientdet_lite0.tflite"
# Parameters
num_threads = 4
# Display parameters
dispW = 1280  # 640 # 
dispH = 720  # 480  #
# fps label
font_pos = (20, 60)
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 0.7
font_weight = 2
font_color = (0, 155, 0)
# Setting up object detection
base_options = core.BaseOptions(file_name=models, use_coral=False, num_threads=num_threads)
detection_options = processor.DetectionOptions(max_results=8, score_threshold = 0.3) # how many objects, how sure to be
options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
detector = vision.ObjectDetector.create_from_options(options)


button = Button(18)
picam = Picamera2()
picam.preview_configuration.main.size = (dispW, dispH)
picam.preview_configuration.main.format = "RGB888"
picam.preview_configuration.controls.FrameRate = 30
picam.preview_configuration.align()  # Forces to standard size for speed
picam.configure("preview")  # Applies the above configurations
picam.start()

webcam = "/dev/video1"
cam = cv2.VideoCapture(webcam)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, dispW)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, dispH)
cam.set(cv2.CAP_PROP_FPS, 30)



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
    ret, image = cam.read()
    # image = picam.capture_array()
    image = cv2.flip(image, 1)

    # Add frame rate
    timer2 = time.time()
    dt = timer2 - timer
    fps = 0.9 * fps + 0.1/ dt
    cv2.putText(image, str(int(np.round(fps))) + " FPS",
                font_pos, cv2.FONT_HERSHEY_SIMPLEX, font_size, 
                font_color, font_weight)
    timer = timer2

    # Tensorflow
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_tensor = vision.TensorImage.create_from_array(image_rgb)
    det_objs = detector.detect(im_tensor)
    image_det = utils.visualize(image, det_objs)

    # Do something with the data
    for det_object in det_objs.detections:
        bounding_box = [det_obj.bounding_box.origin_x,  det_obj.bounding_box.origin_y
                        det_obj.bounding_box.width,  det_obj.bounding_box.height]
        label = det_obj.categories[0].category_name

    # Show the image
    cv2.imshow("Camera", image)

    if cv2.waitKey(1) == ord("q"):
        leave_loop()
cv2.destroyAllWindows()

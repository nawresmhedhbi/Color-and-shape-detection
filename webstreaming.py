# import the necessary packages
from pyimagesearch.motion_detection import singlemotiondetector
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np


# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()
# initialize a flask object
app = Flask(__name__)
# initialize the video stream and allow the camera sensor to
# warmup
#vs = VideoStream(usePiCamera=1).start()
vs = VideoStream(src=0).start()

time.sleep(2.0)

@app.route("/colsha")
def index():
    # return the rendered template
    return render_template("index.html")


def shape(approx):

    # if the shape is a triangle, it will have 3 vertices
    if len(approx) == 3:
        shape = "triangle"
        # if the shape has 4 vertices, it is either a square or
        # a rectangle
    elif len(approx) == 4:
         # compute the bounding box of the contour and use the
         # bounding box to compute the aspect ratio
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
         # a square will have an aspect ratio that is approximately
         # equal to one, otherwise, the shape is a rectangle
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
        # if the shape is a pentagon, it will have 5 vertices
    # elif len(approx) == 5:
     #   shape = "pentagon"
        # otherwise, we assume the shape is a circle
    else:
        shape = "circle"
        # return the name of the shape
    return shape


def make_mask_red(hsv_frame):
    #red mask:
    low_red = np.array([165,167,75])
    high_red = np.array([179,255,255])
    red_mask = cv2.inRange(hsv_frame, low_red, high_red)
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.erode(red_mask, kernel)
  
    return red_mask

def make_mask_yellow(hsv_frame):


    #yellow mask:
    low_yellow = np.array([20,109,73])
    high_yellow = np.array([90,255,228])
    yellow_mask = cv2.inRange(hsv_frame, low_yellow, high_yellow)


 
    kernel = np.ones((5, 5), np.uint8)
   
    yellow_mask = cv2.erode(yellow_mask, kernel)
    return yellow_mask

def make_contour_red(red_mask,frame):
    #for red color:
   cnt_red = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   cnt_red = imutils.grab_contours(cnt_red)
   Text_red= ""


   
   for c in cnt_red:
       area_red= cv2.contourArea(c)
       x, y, w, h = cv2.boundingRect(c)
       if area_red > 5000:
           cv2.drawContours(frame, [approx], -1,(0, 255, 0), 3)
           Text_red="Red" + str(area_red)
           cv2.putText(frame,Text_red , (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255)) 

                
def make_contour_yellow(yellow_mask,frame):

    #for yellow color:
   cnt_yellow = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   cnt_yellow = imutils.grab_contours(cnt_yellow)
   Text_yellow= ""


   
   for c2 in cnt_yellow:

       area_yellow= cv2.contourArea(c2)
       
       x, y, w, h = cv2.boundingRect(c2)
       if area_yellow > 5000: #exact area

           peri = cv2.arcLength(c2, True)
           approx = cv2.approxPolyDP(c2, 0.01 * peri, True)

           cv2.drawContours(frame, [approx], -1,(0, 255, 0), 3)
           s = shape(approx)
           Text_yellow="Yellow"+ s + str(area_yellow)
           
           cv2.putText(frame,Text_yellow , (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255)) 

          



def detect_color_shape(frameCount):
     # grab global references to the video stream, output frame, and
     # lock variables
    global vs, outputFrame, lock
    # initialize the motion detector and the total number of frames
    # read thus far
 
    total = 0
    # loop over frames from the video stream
    while True:

        # read the next frame from the video stream, resize it,
        # convert the frame to grayscale, and blur it
        frame = vs.read()
        #frame = imutils.resize(frame, width=400)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red_mask = make_mask_red(hsv_frame)
        yellow_mask = make_mask_yellow(hsv_frame)
        make_contour_red(red_mask,frame)
        make_contour_yellow(yellow_mask,frame)
    
        total += 1
        # acquire the lock, set the output frame, and release the
        # lock
        with lock:
            outputFrame = frame.copy()


def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock
    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue
            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            # ensure the frame was successfully encoded
            if not flag:
                continue
        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
              bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
                    help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32,
                    help="# of frames used to construct the background model")
    args = vars(ap.parse_args())
    # start a thread that will perform motion detection
    t = threading.Thread(target=detect_color_shape, args=(
        args["frame_count"],))
    t.daemon = True
    t.start()
    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)
    #app.run(host='0.0.0.0', threaded=True)
# release the video stream pointer
vs.stop()
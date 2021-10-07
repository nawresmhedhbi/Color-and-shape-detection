import cv2
import numpy as np
import imutils
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

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
    #elif len(approx) == 5:
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

def make_contour_red(red_mask):
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

                
def make_contour_yellow(yellow_mask):

    #for yellow color:
   cnt_yellow = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   cnt_yellow = imutils.grab_contours(cnt_yellow)
   Text_yellow= ""
   pixel_to_size = None
   

   
   def mdpt(A, B):
       return ((A[0] + B[0]) * 0.5, (A[1] + B[1]) * 0.5)

   for c2 in cnt_yellow:

       area_yellow= cv2.contourArea(c2)
       
       x, y, w, h = cv2.boundingRect(c2)
       if 15000.0<area_yellow < 27000.0: #exact area

           peri = cv2.arcLength(c2, True)
           approx = cv2.approxPolyDP(c2, 0.01 * peri, True)

           cv2.drawContours(frame, [approx], -1,(0, 255, 0), 3)
           s = shape(approx)
           Text_yellow="Yellow"+ s + str(area_yellow)
           
           cv2.putText(frame,Text_yellow , (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255)) 

           bbox = cv2.minAreaRect(c2)
           bbox = cv2.cv.boxPoints(bbox) if imutils.is_cv2() else cv2.boxPoints(bbox)
           bbox = np.array(bbox, dtype="int")
           for (x, y) in bbox:
               
               cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255),-1)
               (tl, tr, br, bl) = bbox
               (tltrX, tltrY) = mdpt(tl, tr)
               (blbrX, blbrY) = mdpt(bl, br)
               (tlblX, tlblY) = mdpt(tl, bl)
               (trbrX, trbrY) = mdpt(tr, br)
               cv2.circle(frame, (int(tltrX), int(tltrY)), 5, (255, 0,0), -1)
               cv2.circle(frame, (int(blbrX), int(blbrY)), 5, (255, 0,0), -1)
               cv2.circle(frame, (int(tlblX), int(tlblY)), 5, (255, 0,0), -1)
               cv2.circle(frame, (int(trbrX), int(trbrY)), 5, (255, 0,0), -1)
               cv2.line(frame, (int(tltrX), int(tltrY)), (int(blbrX),int(blbrY)),(0, 255, 255), 2)
               cv2.line(frame, (int(tlblX), int(tlblY)), (int(trbrX),int(trbrY)),(0, 255, 255), 2)
               dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
               dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
               if pixel_to_size is None:

                   cv2.putText(frame, "{:.1f}in".format(dA),(int(tltrX - 10), int(tltrY - 10)), cv2.FONT_HERSHEY_DUPLEX,0.55, (255, 255, 255), 2)
                   cv2.putText(frame, "{:.1f}in".format(dB),(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_DUPLEX,0.55, (255, 255, 255), 2)
               cv2.drawContours(frame, [bbox.astype("int")], -1, (0, 255, 0), 2)




            
 


    


    
    





while True:
    _, frame = cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_mask = make_mask_red(hsv_frame)
    yellow_mask = make_mask_yellow(hsv_frame)
    make_contour_red(red_mask)
    make_contour_yellow(yellow_mask)
    
    
    cv2.imshow("Frame", frame)
    #cv2.imshow("Red Mask", red)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
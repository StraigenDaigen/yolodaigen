# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 16:24:00 2019

@author: steven
"""


import cv2

img_dir='image_cache\\tmp.jpeg' #bypass

#######################################################
    
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    
while True:
    
    ret, frame = video_capture.read()
    crop_img = frame[0:1920, 0:1080]      
    cv2.imwrite(img_dir, crop_img)
    
#        
#

    img = frame

    

        
    cv2.imshow('Video', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
    

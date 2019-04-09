# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 22:53:24 2018

@author: steven
"""

import cv2
from darkflow.net.build import TFNet
import numpy as np
import time

option = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolov2.weights',
    'threshold': 0.1,
    'gpu': 0.8
}

tfnet = TFNet(option)

#capture = cv2.VideoCapture("videofile_1080_20fps.avi")
capture = cv2.VideoCapture("Recortado_Cafeteria_vacio.mp4")
colors = [tuple(255 * np.random.rand(3)) for i in range(20)]

while (capture.isOpened()):
    stime = time.time()
    ret, frame = capture.read()
    if ret:
        results = tfnet.return_predict(frame)
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            if label != 'person':
                continue
            
            frame = cv2.rectangle(frame, tl, br, color, 7)
            frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow('frame', frame)
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        capture.release()
        cv2.destroyAllWindows()
        break
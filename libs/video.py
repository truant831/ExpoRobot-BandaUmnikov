import cv2
import numpy as np
from threading import Thread

class camera:
    def __init__(self):
        self.pipeline ='v4l2src device=/dev/video0 io-mode=2 ! image/jpeg, width=(int)1280, height=(int)720, framerate=30/1 ! nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink' 
        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        # Подключаем USB камеру
        # self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        # self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        # width = 1280 
        # height = 720
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()
        self.grabbed, self.frame = self.cap.read()
        self.stopped = False
        #self.calibrated = 0
    def start(self):
        Thread(target=self.update, args=()).start()
        return self
    def update(self):
        while True:
            if self.stopped:
                return
            self.grabbed, self.frame = self.cap.read()
            #self.calibrated = cv.remap(self.frame.copy(), map1, map2, interpolation=cv.INTER_CUBIC, borderMode=cv.BORDER_CONSTANT)
    def read(self):
        #print('Resolution: ' + str(self.frame.shape[0]) + ' x ' + str(self.frame.shape[1]))
        return self.frame
    def stop(self):
        self.stopped = True
        self.cap.release()


    
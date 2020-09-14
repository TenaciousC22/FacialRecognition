from threading import Thread
from imutils import face_utils
from imutils.video import VideoStream
import numpy as np
import imutils
import dlib
import cv2

class VideoShow:

	def __init__(self, frame=None):
		self.detector=dlib.get_frontal_face_detector()
		self.predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
		self.frame=frame
		self.stopped=False

	def start(self):
		Thread(target=self.show,args=()).start()
		return self

	def show(self):
		while not self.stopped:
			#self.frame=imutils.resize(self.frame, width=1000)
			gray=cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)
			rects=self.detector(gray,0)
			for rect in rects:
				shape=self.predictor(gray,rect)
				shape=face_utils.shape_to_np(shape)
				shape=shape[48:68]
				cv2.line(self.frame,(shape[0,0],shape[0,1]),(shape[6,0],shape[6,1]),(0,0,255))
				cv2.line(self.frame,(shape[3,0],shape[3,1]),(shape[9,0],shape[9,1]),(0,0,255))
				"""
				for (x,y) in shape:
					cv2.circle(self.frame,(x,y),2,(0,0,255),-1)
					"""
			cv2.imshow("Video",self.frame)
			if cv2.waitKey(1)==ord("q"):
				self.stopped=True

	def stop(self):
		self.stopped=True
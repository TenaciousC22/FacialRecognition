from threading import Thread
from imutils import face_utils
from imutils.video import VideoStream
import numpy as np
import imutils
import dlib
import cv2
import math

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
		t=14 #3, 14
		b=18 #9, 18
		l=12 #0, 12
		r=16 #6, 16
		while not self.stopped:
			gray=cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)
			rects=self.detector(gray,0)
			for rect in rects:
				shape=self.predictor(gray,rect)
				shape=face_utils.shape_to_np(shape)
				shape=shape[48:68]
				
				cv2.line(self.frame,(shape[l,0],shape[l,1]),(shape[r,0],shape[r,1]),(0,0,255))
				cv2.line(self.frame,(shape[t,0],shape[t,1]),(shape[b,0],shape[b,1]),(0,0,255))

				#Find Length of lines
				height=math.sqrt(((shape[t,0]-shape[b,0])**2)+((shape[t,1]-shape[b,1])**2))
				width=math.sqrt(((shape[l,0]-shape[r,0])**2)+((shape[l,1]-shape[r,1])**2))
				#Write text for distance
				cv2.putText(self.frame, "Height: {:.0f}px".format(height),
					(10,415),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(255,255,255))
				cv2.putText(self.frame, "Width: {:.0f}px".format(width),
					(10,380),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(255,255,255))
				if height!=0:
					cv2.putText(self.frame, "W/H Ratio: {:.0f}".format(width/height),
						(10,345),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(255,255,255))
				"""
				for (x,y) in shape:
					cv2.circle(self.frame,(x,y),2,(0,0,255),-1)
				"""
			cv2.imshow("Video",self.frame)
			if cv2.waitKey(1)==ord("q"):
				self.stopped=True

	def stop(self):
		self.stopped=True
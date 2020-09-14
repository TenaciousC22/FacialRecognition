import argparse
import cv2
from CountsPerSec import CountsPerSec
from VideoGet import VideoGet
from VideoShow import VideoShow
from imutils import face_utils
from imutils.video import VideoStream
import numpy as np
import imutils
import dlib
import time

def putIterationsPerSec(frame,iterations_per_sec):
	cv2.putText(frame, "{:.0f} IpS".format(iterations_per_sec),
		(10,450),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(255,255,255))
	return frame

def noThreading(source=0):
	cap = cv2.VideoCapture(source)
	cps = CountsPerSec().start()

	while True:
		(grabbed, frame) = cap.read()
		if not grabbed or cv2.waitKey(1) == ord("q"):
			break

		frame = putIterationsPerSec(frame, cps.countsPerSec())
		cv2.imshow("Video", frame)
		cps.increment()

def threadVideoGet(source=0):
	video_getter=VideoGet(source).start()
	cps=CountsPerSec().start()

	while True:
		if (cv2.waitKey(1)==ord("q")) or video_getter.stopped:
			video_getter.stop()
			break

		frame=video_getter.frame
		frame=putIterationsPerSec(frame,cps.countsPerSec())
		cv2.imshow("Video",frame)
		cps.increment()

def threadVideoShow(source=0):
	cap=cv2.VideoCapture(source)
	(grabbed,frame)=cap.read()
	video_shower=VideoShow(frame).start()
	cps=CountsPerSec().start()

	while True:
		(grabbed,frame)=cap.read()
		if not grabbed or video_shower.stopped:
			video_shower.stop()
			break

		frame=putIterationsPerSec(frame,cps.countsPerSec())
		video_shower.frame=frame
		cps.increment()

def threadBoth(source=0):

	video_getter=VideoGet(source).start()
	video_shower=VideoShow(video_getter.frame).start()
	cps=CountsPerSec().start()

	while True:
		if video_getter.stopped or video_shower.stopped:
			video_shower.stop()
			video_getter.stop()
			break

		frame=video_getter.frame
		"""frame=imutils.resize(frame, width=1000)
		gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		rects=detector(gray,0)
		for rect in rects:
			shape=predictor(gray,rect)
			shape=face_utils.shape_to_np(shape)
			for (x,y) in shape:
				cv2.circle(frame,(x,y),2,(0,0,255),-1)"""
		frame=putIterationsPerSec(frame, cps.countsPerSec())
		video_shower.frame=frame
		cps.increment()
		time.sleep(0.001)

threadBoth()
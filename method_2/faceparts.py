# import the necessary packages
from imutils import face_utils
from imutils.video import VideoStream
import numpy as np
import datetime
import argparse
import imutils
import dlib
import cv2
import time

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Timer to see how long it takes
start=time.time()

# detect faces in the grayscale image
rects = detector(gray, 1)

# loop over the face detections
for (i, rect) in enumerate(rects):
	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)
	
	# convert dlib's rectangle to a OpenCV-style bounding box
	# [i.e., (x, y, w, h)], then draw the face bounding box
	#(x, y, w, h) = face_utils.rect_to_bb(rect)
	#cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
	
	# show the face number
	#cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
		#cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

	# loop over the (x, y)-coordinates for the facial landmarks
	# and draw them on the image
	for (name,(i,j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
		clone=image.copy()
		cv2.putText(clone,name,(10,30),cv2.FONT_HERSHEY_SIMPLEX,
			0.7,(0,0,255),2)
		for (x,y) in shape[i:j]:
			cv2.circle(clone,(x,y),3,(0,0,225),-1)
		
		(x,y,w,h)=cv2.boundingRect(np.array([shape[i:j]]))
		roi=image[y:y+h,x:x+w]
		roi=imutils.resize(roi,width=200,inter=cv2.INTER_CUBIC)
		cv2.imshow("ROI",roi)
		cv2.imshow("Image",clone)
		cv2.waitKey(0)

	output=face_utils.visualize_facial_landmarks(image,shape)
	cv2.imshow("Image",output)
	cv2.waitKey(0)

end=time.time()

# show the output image with the face detections + facial landmarks
print(end-start)
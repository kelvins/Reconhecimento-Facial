
import cv2
import numpy as np
from matplotlib import pyplot as plt

#path1 = "/home/kelvin/Desktop/FaceRecognition/Bases/FaceDataset2/subject_18_1.png"
#path2 = "/home/kelvin/Desktop/FaceRecognition/Bases/FaceDataset2/subject_15_1.png"
path1 = "/home/kelvin/Desktop/livro.jpg"
path2 = "/home/kelvin/Desktop/livros.jpg"

img1  = cv2.imread(path1)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

img2  = cv2.imread(path2)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#surf = cv2.SURF(400) # threshold
surf = cv2.xfeatures2d.SURF_create()

kp1, des1 = surf.detectAndCompute(gray1, None)
kp2, des2 = surf.detectAndCompute(gray2, None)

bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)

matches = bf.match(des1, des2)

matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], img1, flags=2)

plt.imshow(img3),plt.show()

#for kp in keypoints:
#	print "Points(" + str(kp.pt[0]) + "," + str(kp.pt[1]) + "), "
	#print "X:" + str(kp.pt[0]) # X
	#print "Y:" + str(kp.pt[1]) # Y

#surf.hessianThreshold = 50000

#img = cv2.drawKeypoints(gray,keypoints,img)

#cv2.imwrite('surf_keypoints.jpg',img)
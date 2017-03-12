
###################################################################################################################
#
# This script is used to perform face recognition using the SIFT algorithm provided by the OpenCV library.
#
###################################################################################################################

# Import all libraries
import cv2, os
import numpy as np
import FaceRecognitionAssistant as assist

#######################################################################################
#
#   Parameters selection section
#
#######################################################################################

# Defines the path to the training folder
trainPath = "/home/kelvin/Desktop/FaceRecognition/TestesCurvaROC/BASE1/"

# Defines the path to the test folder
testPath  = "/home/kelvin/Desktop/FaceRecognition/TestesCurvaROC/TESTE1/"

# Defines a default size for all images (e.g. 100x100)
defaultSize = 100

#######################################################################################
#
#######################################################################################

# Creates the sift object passing a threshold variable by parameter
sift = cv2.xfeatures2d.SIFT_create()

# Creates the BFMatcher object - The crossCheck is not supported by BFMatcher
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)

# Vector used to store the training images
trainingImages = []

# Vector used to store the training images labels
labels = []

# Function that receives the path of each face image as parameter and include it in the training set (bf object)
def include_face(path):

    # Use these global variables
    global sift, bf, labels, trainingImages

    # Get the subject id (should be a number)
    subjectID = path.split("_")[1]

    # Load Image, Convert to Grayscale, Resize
    image = assist.PreprocessImage( path, defaultSize, defaultSize )
    
    # Detects and computes the keypoints and descriptors using the sift algorithm
    keypoints, descriptors = sift.detectAndCompute(image, None)
    
    # Creates an numpy array
    clusters = np.array([descriptors])

    # Add the array to the BFMatcher
    bf.add(clusters)

    # Store the image to generate an output image
    trainingImages.append( image )

    # Store the subjectID to check if the face recognition is correct
    labels.append( int(subjectID) )

# In the trainingPath folder search for all files in all directories
for dirname, dirnames, filenames in os.walk(trainPath):
    # For each file found
	for filename in filenames:
        # Creates the filePath joining the directory name with the file name
		filePath = os.path.join(dirname, filename)

        # Include the image file in the training set
		include_face(filePath)

# Train: Does nothing for BruteForceMatcher though.
bf.train()

# Count is used just to set a unique name for the output image file
facesRecognized  = 0
faceUnrecognized = 0
nonFaces = 0

# Function that tries to recognize each face (path passed by parameter)
def recognize_face(path):

    # Use these global variables
    global sift, bf, labels, trainingImages, facesRecognized, faceUnrecognized, nonFaces

    # Get the subject id (should be a number)
    subjectID = path.split("_")[1]
    subjectID = int(subjectID.split(".")[0])

    # Load Image, Convert to Grayscale, Resize
    image = assist.PreprocessImage( path, defaultSize, defaultSize )
    
    # Detects and computes the keypoints and descriptors using the sift algorithm
    keypoints, descriptors = sift.detectAndCompute(image, None)
    
    matches = bf.match(descriptors)
    matches = sorted(matches, key = lambda x:x.distance)
    
    # Creates a results vector to store the number of similar points for each image on the training set
    results = [0]*len(labels)

    # Based on the matches vector we create the results vector that represents how many points this test image are similar to each image in the training set
    for i in range(len(matches)):
        results[matches[i].imgIdx] += 1

    # Index receives the position of the maximum value in the results vector (it means that this is the most similar image)
    index = results.index(max(results))

    if subjectID >= 0:
        # Check if the labels[index] (most similar image) is equal to the expected subject (subjectID).
        # It means the face image was correctly recognized
        if labels[index] == subjectID:
            facesRecognized += 1
        else:
            faceUnrecognized += 1
    else:
        nonFaces += 1

# In the trainingPath folder search for all files in all directories
for dirname, dirnames, filenames in os.walk(testPath):
    # For each file found
    for filename in filenames:
        # Creates the filePath joining the directory name with the file name
        filePath = os.path.join(dirname, filename)

        # Try to recognize the current face image
        recognize_face(filePath)

print str(facesRecognized) + " faces recognized."
print str(faceUnrecognized) + " faces unrecognized."
print str(nonFaces) + " non faces."
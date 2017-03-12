
###################################################################################################################
#
# This script is used to perform face recognition using the Fisherface algorithm provided by the OpenCV library.
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

# Creates the fisherface object passing a threshold variable by parameter
fisherface = cv2.face.createFisherFaceRecognizer()

# Vector used to store the training images
trainingImages = []

# Vector used to store the training images labels
labels = []

# In the trainingPath folder search for all files in all directories
for dirname, dirnames, filenames in os.walk(trainPath):
    # For each file found
	for filename in filenames:
        # Creates the filePath joining the directory name with the file name
		filePath = os.path.join(dirname, filename)

        # Include the image file in the training set
		image, subjectID = assist.IncludeFace(filePath, defaultSize, defaultSize)
		
		# Store the image to generate an output image
		trainingImages.append( image )

	    # Store the subjectID to check if the face recognition is correct
		labels.append( subjectID )

# Train the face recognition algorithm
fisherface.train(trainingImages, np.array(labels))

# Count is used just to set a unique name for the output image file
facesRecognized  = 0
faceUnrecognized = 0
nonFaces = 0

# Function that tries to recognize each face (path passed by parameter)
def recognize_face(path):

	# Use these global variables
    global labels, trainingImages, fisherface
    global facesRecognized, faceUnrecognized, nonFaces

    # Get the subject id (should be a number)
    subjectID = path.split("_")[1]
    subjectID = int(subjectID.split(".")[0])

    # Load Image, Convert to Grayscale, Resize
    image = assist.PreprocessImage( path, defaultSize, defaultSize )

    # Perform the face recognition
    subject, confidence = fisherface.predict( image )

    if subjectID >= 0:
		# Check if the subject is equal to the expected subject (subjectID).
	    # It means the face image was correctly recognized
	    if subject == subjectID:
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

        # Include the image file in the training set
		recognize_face(filePath)

print str(facesRecognized) + " faces recognized."
print str(faceUnrecognized) + " faces unrecognized."
print str(nonFaces) + " non faces."

#
# This script is used to perform face recognition using the SURF algorithm provided by the OpenCV library.
#

# Import all libraries
import os
import cv2
import numpy as np

# Creates the SURF object passing a threshold variable by parameter
surf = cv2.xfeatures2d.SURF_create(400)

# Creates the BFMatcher object - The crossCheck is not supported by BFMatcher
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)

# Defines the path to the training folder
trainPath = "/home/kelvin/Desktop/FaceRecognition/TestesCurvaROC/Base1/"

# Defines the path to the test folder
testPath  = "/home/kelvin/Desktop/FaceRecognition/TestesCurvaROC/Teste1/"

# Defines the path to generate the output images
outputPath = "/home/kelvin/Desktop/RESULTS_SURF/"

# Defines a default size for all images (e.g. 100x100)
defaultSize = 200

# Vector used to store the training images
trainingImages = []

# Vector used to store the training images labels
labels = []

# Function that receives the path of each face image as parameter and include it in the training set (bf object)
def include_face(path):

    # Use these global variables
    global surf, bf, labels, trainingImages

    # Get the subject id (should be a number)
    subjectID = path.split("_")[1]
    #subjectID = path.split("_")[1]

    # Loads the image into the image variable
    image = cv2.imread(path)

    # Convert the image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image to a default size
    image = cv2.resize(image, (defaultSize, defaultSize), interpolation = cv2.INTER_CUBIC)
    
    # Detects and computes the keypoints and descriptors using the SURF algorithm
    keypoints, descriptors = surf.detectAndCompute(image, None)
    
    # Creates an numpy array
    clusters = np.array([descriptors])

    # Add the array to the BFMatcher
    bf.add(clusters)

    # Store the image to generate an output image
    trainingImages.append( image )

    # Store the subjectID to check if the face recognition is correct
    labels.append( subjectID )

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
count = 0

# Function that tries to recognize each face (path passed by parameter)
def recognize_face(path):

    # Use these global variables
    global surf, bf, labels, trainingImages, count

    # Get the subject id (should be a number)
    subjectID = path.split("_")[1]
    subjectID = subjectID.split(".")[0]

    # Loads the image into the image variable
    image = cv2.imread(path)

    # Convert the image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image to a default size
    image = cv2.resize(image, (defaultSize, defaultSize), interpolation = cv2.INTER_CUBIC)
    
    # Detects and computes the keypoints and descriptors using the SURF algorithm
    keypoints, descriptors = surf.detectAndCompute(image, None)
    
    matches = bf.match(descriptors)
    matches = sorted(matches, key = lambda x:x.distance)
    
    # Creates a results vector to store the number of similar points for each image on the training set
    results = [0]*len(labels)

    # Based on the matches vector we create the results vector that represents how many points this test image are similar to each image in the training set
    for i in range(len(matches)):
        results[matches[i].imgIdx] += 1

    # Index receives the position of the maximum value in the results vector (it means that this is the most similar image)
    index = results.index(max(results))

    # Check if the labels[index] (most similar image) is equal to the expected subject (subjectID).
    # It means the face image was correctly recognized
    if labels[index] == subjectID:
        # Concatenate the two images (from the index position in the training set and the current test image) to generate the output image
        tempImage = np.concatenate((trainingImages[index], image), axis=1)

        # Save the concatenated image to the output path
        cv2.imwrite(outputPath + str(count) + ".png", tempImage)
        print outputPath + str(count) + ".png"

        count += 1

# In the trainingPath folder search for all files in all directories
for dirname, dirnames, filenames in os.walk(testPath):
    # For each file found
    for filename in filenames:
        # Creates the filePath joining the directory name with the file name
        filePath = os.path.join(dirname, filename)

        # Try to recognize the current face image
        recognize_face(filePath)

print str(count) + " faces recognized."


import os

import FaceRecognition as fr
from Algorithms import Algorithms
from Interpolation import Interpolation

# Define the path to the training files/folder
trainPath = "/Users/kelvinsp/Desktop/Treinamento6/"

# Define the path to the test folder
testPath = "/Users/kelvinsp/Desktop/Teste6/"

faceRec = fr.FaceRecognition()

# Set the interpolation method (default: INTER_CUBIC)
faceRec.setInterpolation(Interpolation.INTER_CUBIC)

# Set a fixed size for the images (default 100x100)
faceRec.setDefaultSize(100, 100)

# Train the algorithms
faceRec.train(trainPath)

# In the testPath folder search for all files in all directories
for dirname, dirnames, filenames in os.walk(testPath):
    # For each file found
    for filename in filenames:
            # Creates the filePath joining the directory name with the file name
            filePath = os.path.join(dirname, filename)

            # Try to predict for the current file
            subjects = faceRec.predict(filePath)

            # If subjects is not None, print the results
            if subjects is not None:
                print filePath
                print subjects
            else:
                print "Subjects none"

# Save the result in a text file
faceRec.save()

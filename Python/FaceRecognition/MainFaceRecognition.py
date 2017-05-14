
import os
import cv2
import sys

sys.path.append('Auxiliary')
from Auxiliary import Auxiliary

sys.path.append('FaceRecognition')
from FaceRecognition import FaceRecognition

sys.path.append('Report')
from Report import Report

sys.path.append('Algorithms')
from Eigenfaces import Eigenfaces
from Fisherfaces import Fisherfaces
from LBPH import LBPH
from SIFT import SIFT
from SURF import SURF

def main():

    realPath = os.path.realpath(__file__)
    dirPath  = os.path.dirname(realPath)

    # Define the path to the training files/folder
    trainPath = dirPath + "/Dataset/Train/"

    # Define the path to the test folder
    testPath = dirPath + "/Dataset/Test/"

    # Define the path to the results folder
    resultsPath = dirPath + "/Results/"

    # Create the auxiliary object
    auxiliary = Auxiliary(sizeX=100, sizeY=100, interpolation=cv2.INTER_CUBIC)

    # Create the algorithm object
    algorithm = Eigenfaces() # numComponents=0, threshold=-1

	# Create the face recognition object
    faceRecog = FaceRecognition(algorithm, auxiliary)

    # Train the algorithm
    faceRecog.train(trainPath)

    # Try to recognize the faces
    faceRecog.recognizeFaces(testPath)

    # Create the report object
    report = Report(faceRecog)

    # Print the results
    report.printResults()

    # Save the report (text file)
    report.saveReport(resultsPath)

    # Save all results (summary, full report and images)
    #report.saveAllResults(resultsPath)

if __name__ == "__main__":
    main()

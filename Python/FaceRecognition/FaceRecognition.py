
# Import the libraries
import cv2
import os
import sys
import numpy as np

sys.path.append('Auxiliary')
from Auxiliary import Auxiliary

class FaceRecognition:
    """
    Class that provides an interface to the face recognition algorithms
    """

    def __init__(self, algorithm, auxiliary=Auxiliary()):
    	self.algorithm = algorithm
        self.auxiliary = auxiliary
        reset()

    def setDefaultSize(self, sizeX, sizeY):
        self.auxiliary.setDefaultSize(sizeX, sizeY)

    def setInterpolation(self, interpolation):
        self.auxiliary.setInterpolation(interpolation)

    def reset(self):
    	self.images = []
    	self.labels = []
        resetResults()

    def resetResults(self):
        self.nonFaces = 0
        self.recognized = 0
        self.unrecognized = 0

    def train(self, trainPath):
        """
        Function responsible for train the face recognition algorithm based on the image files from the trainPath.
        """
        reset()

        if trainPath == "":
            print "The train path is empty."
            sys.exit()

        # Load all imagens and labels
        self.images, self.labels = self.auxiliary.loadAllImagesForTrain(trainPath)

        # Train the algorithm
        self.algorithm.train(self.images, self.labels)

    def recognizeFaces(self, testPath):
        """
        Function that tries to recognize each face (path passed by parameter).
        """

        resetResults()

        if testPath == "":
            print "The test path is empty."
            sys.exit()

        # Load all imagens and labels
        tempImages, tempLabels = self.auxiliary.loadAllImagesForTest(testPath)

        # For each image
        for index in xrange(0,len(tempImages)):
            # Predict
            subjectID, confidence = self.algorithm.predict(tempImages[index])

            if tempLabels[index] >= 0:
                if subjectID == tempLabels[index]:
                    recognized += 1
                else:
                    unrecognized += 1
            else:
                nonFaces += 1
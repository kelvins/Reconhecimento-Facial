
# Import the libraries
import cv2
import os
import sys
import numpy as np

from auxiliary import Auxiliary

class FaceRecognition:
    """
    Class that provides an interface to the face recognition algorithms
    """

    def __init__(self, algorithm, auxiliary=Auxiliary(), threshold=-1):
    	self.algorithm = algorithm
        self.auxiliary = auxiliary
        self.threshold = threshold
        self.reset()

    def reset(self):
        # Reset all lists
        self.trainImages = []
        self.trainLabels = []
        # Reset the paths
        self.trainPath = ""
        self.testPath  = ""
        # Reset the results
        self.resetResults()

    def resetResults(self):
        # Reset all results
        self.recognized   = 0
        self.unrecognized = 0
        self.nonFaces     = 0

        # Reset the report
        self.predictSubjectIds   = []
        self.predictConfidence   = []

        # Reset test results
        self.testImages = []
        self.testLabels = []
        self.testFileNames = []

    def setAuxiliary(self, auxiliary):
        self.auxiliary = auxiliary

    def getAuxiliary(self):
        return self.auxiliary

    def setAlgorithm(self, algorithm):
        self.algorithm = algorithm

    def getAlgorithm(self):
        return self.algorithm

    def setThreshold(self, threshold):
        self.threshold = threshold

    def getThreshold(self):
        return self.threshold

    def getPredictedSubjectIds(self):
        return self.predictSubjectIds

    def getPredictedConfidence(self):
        return self.predictConfidence

    def getTestImages(self):
        return self.testImages

    def getTestLabels(self):
        return self.testLabels

    def getTestFileNames(self):
        return self.testFileNames

    def getTrainImages(self):
        return self.trainImages

    def getTrainLabels(self):
        return self.trainLabels

    def getResults(self):
        return self.recognized, self.unrecognized, self.nonFaces

    def getTrainPath(self):
        return self.trainPath

    def getTestPath(self):
        return self.testPath

    def train(self, trainPath):
        """
        Function responsible for train the face recognition algorithm based on the image files from the trainPath.
        """
        self.reset()

        # Store the train path
        self.trainPath = trainPath

        if trainPath == "":
            print "The train path is empty."
            sys.exit()

        # Load all imagens and labels
        self.trainImages, self.trainLabels, _ = self.auxiliary.loadAllImagesForTrain(trainPath)

        # Train the algorithm
        self.algorithm.train(self.trainImages, self.trainLabels)

    def recognizeFaces(self, testPath):
        """
        Function that tries to recognize each face (path passed by parameter).
        """
        self.resetResults()

        # Store the test path
        self.testPath = testPath

        if testPath == "":
            print "The test path is empty."
            sys.exit()

        # Load all imagens and labels
        self.testImages, self.testLabels, self.testFileNames = self.auxiliary.loadAllImagesForTest(testPath)

        # For each image
        for index in xrange(0, len(self.testImages)):
            # Predict
            subjectID, confidence = self.algorithm.predict(self.testImages[index])

            # Store the predicted results to be used in the report
            self.predictSubjectIds.append( subjectID )
            self.predictConfidence.append( confidence )

            # Approach not using threshold (face images manually classified)
            if self.threshold == -1:
                if self.testLabels[index] >= 0:
                    if subjectID == self.testLabels[index]:
                        self.recognized += 1
                    else:
                        self.unrecognized += 1
                else:
                    self.nonFaces += 1
            # Approach using threshold (don't know what is nonface)
            else:
                if confidence <= self.threshold:
                    self.recognized += 1
                else:
                    self.unrecognized += 1


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

        # Reset all lists
        self.trainImages = []
        self.trainLabels = []
        # Reset the paths
        self.trainPath = ""
        self.testPath = ""

        # Reset all results
        self.recognized = 0
        self.unrecognized = 0
        self.nonFaces = 0

        # Reset all results (using threshold)
        self.recognizedBelowThreshold = 0
        self.unrecognizedBelowThreshold = 0
        self.nonFacesBelowThreshold = 0
        self.recognizedAboveThreshold = 0
        self.unrecognizedAboveThreshold = 0
        self.nonFacesAboveThreshold = 0

        # Reset the report
        self.predictSubjectIds = []
        self.predictConfidence = []

        # Reset test results
        self.testImages = []
        self.testLabels = []
        self.testFileNames = []

    def reset(self):
        """
        Reset all values, including train and test paths
        """
        # Reset all lists
        self.trainImages = []
        self.trainLabels = []
        # Reset the paths
        self.trainPath = ""
        self.testPath = ""
        # Reset the results
        self.resetResults()

    def resetResults(self):
        """
        Reset all results
        """
        # Reset all results
        self.recognized = 0
        self.unrecognized = 0
        self.nonFaces = 0

        # Reset all results (using threshold)
        self.recognizedBelowThreshold = 0
        self.unrecognizedBelowThreshold = 0
        self.nonFacesBelowThreshold = 0
        self.recognizedAboveThreshold = 0
        self.unrecognizedAboveThreshold = 0
        self.nonFacesAboveThreshold = 0

        # Reset the report
        self.predictSubjectIds = []
        self.predictConfidence = []

        # Reset test results
        self.testImages = []
        self.testLabels = []
        self.testFileNames = []

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
        self.trainImages, self.trainLabels, _ = self.auxiliary.loadAllImagesForTrain(
            trainPath)

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
        self.testImages, self.testLabels, self.testFileNames = self.auxiliary.loadAllImagesForTest(
            testPath)

        # For each image
        for index in xrange(0, len(self.testImages)):
            # Predict
            subjectID, confidence = self.algorithm.predict(
                self.testImages[index])

            # Store the predicted results to be used in the report
            self.predictSubjectIds.append(subjectID)
            self.predictConfidence.append(confidence)

            # Approach not using threshold (face images manually classified)
            if self.threshold == -1:
                if self.testLabels[index] >= 0:
                    if subjectID == self.testLabels[index]:
                        self.recognized += 1
                    else:
                        self.unrecognized += 1
                else:
                    self.nonFaces += 1
            # Approach using threshold
            else:
                # Compute results below threshold
                if confidence <= self.threshold:
                    if self.testLabels[index] >= 0:
                        if subjectID == self.testLabels[index]:
                            self.recognizedBelowThreshold += 1
                        else:
                            self.unrecognizedBelowThreshold += 1
                    else:
                        self.nonFacesBelowThreshold += 1
                # Compute results above threshold
                else:
                    if self.testLabels[index] >= 0:
                        if subjectID == self.testLabels[index]:
                            self.recognizedAboveThreshold += 1
                        else:
                            self.unrecognizedAboveThreshold += 1
                    else:
                        self.nonFacesAboveThreshold += 1

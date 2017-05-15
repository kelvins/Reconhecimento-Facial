
# Import the libraries
import cv2
import os
import sys
import numpy as np

sys.path.append('../Voting')
from Voting import Voting

sys.path.append('../Auxiliary')
from Auxiliary import Auxiliary

class MachineryCommittee:
    """
    Class that provides an interface for the MachineryCommittee
    """

    def __init__(self, fralgorithms=[], auxiliary=Auxiliary(), voting=Voting()):
        self.fralgorithms = fralgorithms
        self.auxiliary = auxiliary
        self.voting = voting
        self.reset()

    def reset(self):
        """
        Reset all lists and results.
        It is used to reset all values to re-train the algorithm
        """
        self.trainImages = []
        self.trainLabels = []
        # Reset the paths
        self.trainPath = ""
        self.testPath  = ""
        # Reset the results
        self.resetResults()

    def resetResults(self):
        """
        Reset results (including the test lists and the predictions)
        It is used to reset only the results of the tests
        """
        # Reset all results
        self.recognized   = 0
        self.unrecognized = 0
        self.nonFaces     = 0

        # Reset the predicted results
        self.predictSubjectIds = []
        self.predictConfidence = []

        # Reset test results
        self.testImages = []
        self.testLabels = []
        self.testFileNames = []

    def setFRAlgorithms(self, fralgorithms):
        self.fralgorithms = fralgorithms

    def getFRAlgorithms(self):
        return self.fralgorithms

    def setAuxiliary(self, auxiliary):
        self.auxiliary = auxiliary

    def getAuxiliary(self):
        return self.auxiliary

    def setVoting(self, voting):
        self.voting = voting

    def getVoting(self):
        return self.voting

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
        return self.nonFaces, self.recognized, self.unrecognized

    def getTrainPath(self):
        return self.trainPath

    def getTestPath(self):
        return self.testPath

    def train(self, trainPath):
        """
        Function responsible for train the face recognition algorithm based on the image files from the trainPath.
        """
        # Reset all lists and results
        self.reset()

        # Store the train path
        self.trainPath = trainPath

        if trainPath == "":
            print "The train path is empty."
            sys.exit()

        # Load all imagens and labels
        self.trainImages, self.trainLabels, _ = self.auxiliary.loadAllImagesForTrain(trainPath)

        # Train all the algorithms
        for index in xrange(0, len(self.fralgorithms)):
            self.fralgorithms[index].train(self.trainImages, self.trainLabels)

    def recognizeFaces(self, testPath):
        """
        Function that tries to recognize each face (path passed by parameter).
        """
        # Reset the results
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
            subjectID  = []
            confidence = []

            # Predict
            for index in xrange(0, len(self.fralgorithms)):
                subID, conf = self.fralgorithms[index].predict(self.testImages[index])
                subjectID.append(subID)
                confidence.append(conf)

            # If using weighted voting the subjectID length should be equal to the weights length
            result = self.voting.vote(subjectID)

            # Store the predicted results to be used in the report
            self.predictSubjectIds.append( result )
            # We cannot use the confidence because we are using multiple algorithms
            #self.predictConfidence.append( confidence )

            # Approach not using threshold (face images manually classified)
            if self.testLabels[index] >= 0:
                if result == self.testLabels[index]:
                    self.recognized += 1
                else:
                    self.unrecognized += 1
            else:
                self.nonFaces += 1

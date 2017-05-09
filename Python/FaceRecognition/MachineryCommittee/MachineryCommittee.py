
# Import the libraries
import cv2
import os
import sys
import numpy as np

sys.path.append('../Voting')
from Voting import Voting

sys.path.append('../Auxiliary')
from Auxiliary import Auxiliary

sys.path.append('../FaceRecognition')
from FaceRecognition import FaceRecognition

class MachineryCommittee:
    """
    Class that provides an interface for the MachineryCommittee
    """

    def __init__(self, faceRecognitionAlgorithms=[], auxiliary=Auxiliary(), voting=Voting()):
        self.auxiliary = auxiliary
        self.faceRecognitionAlgorithms = faceRecognitionAlgorithms
        self.voting = voting
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

    def getResults(self):
        return self.nonFaces, self.recognized, self.unrecognized

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

        # Train the algorithms
        for index in xrange(0, len(self.faceRecognitionAlgorithms)):
            self.faceRecognitionAlgorithms[index].train(self.images, self.labels)

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
            subjectID  = []
            confidence = []

            # Predict
            for index in xrange(0, len(self.faceRecognitionAlgorithms)):
                subID, conf = self.faceRecognitionAlgorithms[index].predict(tempImages[index])
                subjectID.append(subID)
                confidence.append(conf)

            # If using weighted voting the subjectID length should be equal to the weights length
            result = voting.vote(subjectID)

            # Approach not using threshold (face images manually classified)
            if tempLabels[index] >= 0:
                if result == tempLabels[index]:
                    recognized += 1
                else:
                    unrecognized += 1
            else:
                nonFaces += 1

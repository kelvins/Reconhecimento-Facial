
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

sys.path.append('../Algorithms')
from Eigenfaces import Eigenfaces
from Fisherfaces import Fisherfaces
from LBPH import LBPH
from SIFT import SIFT
from SURF import SURF

class MachineryCommittee:
    """
    Class that provides an interface for the MachineryCommittee
    """

    def __init__(self, eigenfaces=Eigenfaces(), fisherfaces=Fisherfaces(), LBPH=LBPH(), SIFT=SIFT(), SURF=SURF(), auxiliary=Auxiliary(), voting=Voting()):
        self.auxiliary = auxiliary

        self.eigenfaces = eigenfaces
        self.fisherfaces = fisherfaces
        self.LBPH = LBPH
        self.SIFT = SIFT
        self.SURF = SURF

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

        # Train the algorithm
        self.eigenfaces.train(self.images, self.labels)
        self.fisherfaces.train(self.images, self.labels)
        self.LBPH.train(self.images, self.labels)
        self.SIFT.train(self.images, self.labels)
        self.SURF.train(self.images, self.labels)

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
            subjectID1, confidence1 = self.eigenfaces.predict(tempImages[index])
            subjectID2, confidence2 = self.fisherfaces.predict(tempImages[index])
            subjectID3, confidence3 = self.LBPH.predict(tempImages[index])
            subjectID4, confidence4 = self.SIFT.predict(tempImages[index])
            subjectID5, confidence5 = self.SURF.predict(tempImages[index])

            results = [subjectID1, subjectID2, subjectID3, subjectID4, subjectID5]
            result = voting.vote(results)

            if tempLabels[index] >= 0:
                if result == tempLabels[index]:
                    recognized += 1
                else:
                    unrecognized += 1
            else:
                nonFaces += 1
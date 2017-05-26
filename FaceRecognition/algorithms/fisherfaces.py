
# Import the libraries
import cv2
import os
import numpy as np

class Fisherfaces:
    """
    Class that provides easy access to the Fisherfaces algorithm
    """

    def __init__(self, numComponents=0, threshold=-1):
        """
        Set the default values
        """
        if numComponents < 0:
            numComponents = 0

        # Creates the fisherfaces object passing a threshold variable by parameter
        if threshold >= 0:
            self.faceRec = cv2.face.createFisherFaceRecognizer(num_components=numComponents, threshold=threshold)
        else:
            self.faceRec = cv2.face.createFisherFaceRecognizer(num_components=numComponents) # threshold=DBL_MAX
            
        self.algorithmTrained = False

    def getAlgorithmName(self):
        return "Fisherfaces"

    def train(self, images, labels):
        """
        Train the face recognition algorithm
        """
        self.faceRec.train(images, np.array(labels))
        self.algorithmTrained = True

    def predict(self, image):
        """
        Predict the image
        """
        if self.algorithmTrained is False:
            print "The face recognition algorithm was not trained."
            sys.exit()

        # Return the subject ID (label) and the confidence
        return self.faceRec.predict( image ) 


# Import the libraries
import cv2
import os
import numpy as np

class Eigenfaces:
    """
    Class that provides easy access to the Eigenfaces algorithm
    """

    def __init__(self, numComponents=-1, threshold=-1):
        """
        Set the default values
        """
        # Creates the eigenface object passing a threshold variable by parameter
        if numComponents >= 0  and threshold >= 0:
            self.faceRec = cv2.face.createEigenFaceRecognizer(num_components=numComponents, threshold=threshold)
        elif numComponents >= 0:
            self.faceRec = cv2.face.createEigenFaceRecognizer(num_components=numComponents)
        elif numComponents >= 0:
            self.faceRec = cv2.face.createEigenFaceRecognizer(threshold=threshold)
        else:
            self.faceRec = cv2.face.createEigenFaceRecognizer() # num_components=0, threshold=DBL_MAX

    def train(self, images, labels):
        """
        Train the face recognition algorithm
        """
        self.faceRec.train(images, np.array(labels))

    def predict(self, image):
        """
        Predict the image
        """
        # Return the subject ID (label) and the confidence
        return self.faceRec.predict( image ) 

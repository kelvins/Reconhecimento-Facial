
# Import the libraries
import cv2
import os
import numpy as np

class LBPH:
    """
    Class that provides easy access to the LBPH algorithm
    """

    def __init__(self, radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=-1):
        """
        Set the default values
        """
        if radius < 1:
            radius = 1
        if neighbors < 1:
            neighbors = 1
        if grid_x < 1:
            grid_x = 1
        if grid_y < 1:
            grid_y = 1

        # Creates the LBPH object passing a threshold variable by parameter
        if threshold >= 0:
            self.faceRec = cv2.face.createLBPHFaceRecognizer(radius=radius, neighbors=neighbors, grid_x=grid_x, grid_y=grid_y, threshold=threshold)
        else:
            self.faceRec = cv2.face.createLBPHFaceRecognizer(radius=radius, neighbors=neighbors, grid_x=grid_x, grid_y=grid_y) # threshold=DBL_MAX

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

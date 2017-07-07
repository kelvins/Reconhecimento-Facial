
# Import the libraries
import cv2
import sys
import numpy as np


class Fisherfaces:
    """
    Class that provides easy access to the Fisherfaces algorithm.
    """

    def __init__(self, numComponents=0, threshold=-1):
        """
        Set the default values.
        :param numComponents: The number of components (default 0).
        :param threshold: The threshold (default 0).
        """
        if numComponents < 0:
            numComponents = 0

        # Creates the fisherfaces object passing a threshold variable by
        # parameter
        if threshold >= 0:
            self.faceRec = cv2.face.createFisherFaceRecognizer(
                num_components=numComponents, threshold=threshold)
        else:
            self.faceRec = cv2.face.createFisherFaceRecognizer(
                num_components=numComponents)  # threshold=DBL_MAX

        self.algorithmTrained = False

    def getAlgorithmName(self):
        """
        Get the algorithm name.
        :return: The algorithm name.
        """
        return "Fisherfaces"

    def train(self, images, labels):
        """
        Train the face recognition algorithm
        :param images: A slice with all images for training.
        :param labels: A slice with all labels corresponding to the images.
        """
        self.faceRec.train(images, np.array(labels))
        self.algorithmTrained = True

    def predict(self, image):
        """
        Predict the image. Given a new image this function will make the prediction.
        :param image: The image we want to predict.
        :return: The subject ID (label) and the confidence.
        """
        # Check if the algorithm was trained
        if self.algorithmTrained is False:
            print "The face recognition algorithm was not trained."
            sys.exit()

        # Return the subject ID (label) and the confidence
        return self.faceRec.predict(image)

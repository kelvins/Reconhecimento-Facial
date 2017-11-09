
# Import the libraries
import cv2
import sys
import numpy as np


class LBPH(object):
    """
    Class that provides easy access to the LBPH algorithm.
    """

    __algorithm_name = "Local Binary Patterns Histogram (LBPH)"

    def __init__(self, radius=1, neighbors=8,
                 grid_x=8, grid_y=8, threshold=-1):
        """
        Set the default values.
        :param radius: The radius (default 1).
        :param neighbors: The number of neighbors (default 8).
        :param grid_x: The grid X (default 8).
        :param grid_y: The grid Y (default 8).
        :param threshold: The threshold (default -1).
        """

        # If the parameter is invalid get its default value
        if radius < 1:
            radius = 1
        if neighbors < 1:
            neighbors = 8
        if grid_x < 1:
            grid_x = 8
        if grid_y < 1:
            grid_y = 8

        # Creates the LBPH object passing a threshold variable by parameter
        if threshold >= 0:
            self.faceRec = cv2.face.createLBPHFaceRecognizer(
                radius=radius,
                neighbors=neighbors,
                grid_x=grid_x,
                grid_y=grid_y,
                threshold=threshold)
        else:
            self.faceRec = cv2.face.createLBPHFaceRecognizer(
                radius=radius,
                neighbors=neighbors,
                grid_x=grid_x,
                grid_y=grid_y)  # threshold=DBL_MAX

        self.trained = False

    @property
    def algorithm_name(self):
        """
        :return: __algorithm_name
        """
        return self.__algorithm_name

    def train(self, images, labels):
        """
        Train the face recognition algorithm
        :param images: A slice with all images for training.
        :param labels: A slice with all labels corresponding to the images.
        """
        self.faceRec.train(images, np.array(labels))
        self.trained = True

    def predict(self, image):
        """
        Predict the image. Given a new image this function will make the prediction.
        :param image: The image we want to predict.
        :return: The subject ID (label) and the confidence.
        """
        # Check if the algorithm was trained
        if self.trained is False:
            print("The {} algorithm was not trained.".format(self.__algorithm_name))
            sys.exit()

        # Return the subject ID (label) and the confidence
        return self.faceRec.predict(image)

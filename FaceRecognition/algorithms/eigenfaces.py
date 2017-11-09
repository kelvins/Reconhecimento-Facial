
# Import the libraries
import cv2
import sys
import numpy as np


class Eigenfaces(object):
    """
    Class that provides easy access to the Eigenfaces algorithm.
    """

    __algorithm_name = "Eigenfaces"

    def __init__(self, num_components=0, threshold=-1):
        """
        Set the default values.
        :param num_components: The number of components (default 0).
        :param threshold: The threshold (default -1).
        """

        if num_components < 0:
            num_components = 0

        # Creates the eigenface object passing a threshold variable by
        # parameter
        if threshold >= 0:
            self.faceRec = cv2.face.createEigenFaceRecognizer(
                num_components=num_components, threshold=threshold)
        else:
            self.faceRec = cv2.face.createEigenFaceRecognizer(
                num_components=num_components)  # threshold=DBL_MAX

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

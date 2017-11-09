
# Import the libraries
import cv2
import sys
import numpy as np


class SURF(object):
    """
    Class that provides easy access to the SURF algorithm.
    """

    __algorithm_name = "Speeded Up Robust Features (SURF)"

    def __init__(self, hessian_threshold=100, n_octaves=4, n_octave_layers=3,
                 extended=False, upright=False, distance=cv2.NORM_L2, cross_check=False):
        """
        Set the default values.
        :param hessian_threshold: The hessian threshold (default 100).
        :param n_octaves: The number of octaves (default 4).
        :param n_octave_layers: The number of octave layers (default 3).
        :param extended: The extended parameter (default false).
        :param upright: The upright parameter (default false).
        :param distance: The distance parameter (default cv2.NORM_L2 - euclidean distance).
        :param cross_check: The cross check parameter (default false).
        """

        # If the parameter is invalid get its default value
        if hessian_threshold < 0:
            hessian_threshold = 100
        if n_octaves < 0:
            n_octaves = 4
        if n_octave_layers < 0:
            n_octave_layers = 3

        # Creates the SURF object
        self.faceRec = cv2.xfeatures2d.SURF_create(
            hessianThreshold=hessian_threshold,
            nOctaves=n_octaves,
            nOctaveLayers=n_octave_layers,
            extended=extended,
            upright=upright)

        # Creates the matcher object
        self.matcher = cv2.BFMatcher(distance, crossCheck=cross_check)

        self.labels = []
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
        self.labels = labels

        for image in images:

            # Detects and computes the keypoints and descriptors using the SURF
            # algorithm
            keypoints, descriptors = self.faceRec.detectAndCompute(image, None)

            # Creates an numpy array
            clusters = np.array([descriptors])

            # Add the array to the BFMatcher
            self.matcher.add(clusters)

        # Train: Does nothing for BruteForceMatcher though
        self.matcher.train()
        self.trained = True

    def predict(self, image):
        """
        Predict the image. Given a new image this function will make the prediction.
        :param image: The image we want to predict.
        :return: The subject ID (label) and the confidence.
        """
        if self.trained is False:
            print("The {} algorithm was not trained.".format(self.__algorithm_name))
            sys.exit()

        # Detects and computes the keypoints and descriptors using the SURF
        # algorithm
        keypoints, descriptors = self.faceRec.detectAndCompute(image, None)

        # Get all matches based on the descriptors
        matches = self.matcher.match(descriptors)

        # Order by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Creates a results vector to store the number of similar points for
        # each image on the training set
        results = [0] * len(self.labels)

        # Based on the matches vector we create the results vector that
        # represents how many points this test image are similar to each image
        # in the training set
        for match in matches:
            if match.imgIdx >= 0 and match.imgIdx < len(results):
                results[match.imgIdx] += 1
            else:
                print("Error invalid index")
                sys.exit()

        # Index receives the position of the maximum value in the results
        # vector (it means that this is the most similar image)
        index = results.index(max(results))

        # Calculate the confidence based on the number of matches and the max result
        # The confidence range is: 0 - 100
        # The closer to zero higher is the confidence
        if len(matches) > 0:
            confidence = 100.0 - \
                ((float(max(results)) * 100.0) / float(len(matches)))
        else:
            confidence = 0.0

        return self.labels[index], confidence

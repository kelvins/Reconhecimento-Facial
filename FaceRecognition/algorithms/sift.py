
# Import the libraries
import cv2
import sys
import numpy as np


class SIFT(object):
    """
    Class that provides easy access to the SIFT algorithm.
    """

    __algorithm_name = "Scale-Invariant Feature Transform (SIFT)"

    def __init__(self, n_features=0, n_octave_layers=3, contrast_threshold=0.04,
                 edge_threshold=10, sigma=1.6, distance=cv2.NORM_L2, cross_check=False):
        """
        Set the default values.
        :param n_features: The number of features (default 0).
        :param n_octave_layers: The number of octave layers (default 3).
        :param contrast_threshold: The contrast threshold (default 0.04).
        :param edge_threshold: The edge threshold (default 10).
        :param sigma: The sigma value (default 1.6).
        :param distance: The distance (default cv2.NORM_L2 - euclidean distance).
        :param cross_check: The cross check parameter (default false).
        """

        # If the parameter is invalid get its default value
        if n_features < 0:
            n_features = 0
        if n_octave_layers < 0:
            n_octave_layers = 3
        if contrast_threshold < 0.0:
            contrast_threshold = 0.04
        if edge_threshold < 0:
            edge_threshold = 10
        if sigma < 0.0:
            sigma = 1.6

        # Creates the SIFT object
        self.faceRec = cv2.xfeatures2d.SIFT_create(
            nfeatures=n_features,
            nOctaveLayers=n_octave_layers,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
            sigma=sigma)

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

            # Detects and computes the keypoints and descriptors using the sift
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

        # Detects and computes the keypoints and descriptors using the sift
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

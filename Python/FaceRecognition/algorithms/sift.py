
# Import the libraries
import cv2
import os
import numpy as np

class SIFT:
    """
    Class that provides easy access to the SIFT algorithm
    """

    def __init__(self, nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6, distance=cv2.NORM_L2, crossCheck=False):
        """
        Set the default values
        """
        
        # If the parameter is invalid get its default value
        if nfeatures < 0:
            nfeatures = 0
        if nOctaveLayers < 0:
            nOctaveLayers = 3
        if contrastThreshold < 0.0:
            contrastThreshold = 0.04
        if edgeThreshold < 0:
            edgeThreshold = 10
        if sigma < 0.0:
            sigma = 1.6

        # Creates the SIFT object
        self.faceRec = cv2.xfeatures2d.SIFT_create(nfeatures=nfeatures, nOctaveLayers=nOctaveLayers, contrastThreshold=contrastThreshold, edgeThreshold=edgeThreshold, sigma=sigma)

        # Creates the matcher object
        self.matcher = cv2.BFMatcher(distance, crossCheck=crossCheck)

        self.labels = []
        self.algorithmTrained = False

    def getAlgorithmName(self):
        return "Scale-Invariant Feature Transform (SIFT)"

    def train(self, images, labels):
        """
        Train the face recognition algorithm
        """
        self.labels = labels

        for image in images:

            # Detects and computes the keypoints and descriptors using the sift algorithm
            keypoints, descriptors = self.faceRec.detectAndCompute(image, None)
    
            # Creates an numpy array
            clusters = np.array([descriptors])

            # Add the array to the BFMatcher
            self.matcher.add(clusters)

        # Train: Does nothing for BruteForceMatcher though
        self.matcher.train()
        self.algorithmTrained = True

    def predict(self, image):
        """
        Predict the image
        """
        if self.algorithmTrained is False:
            print "The face recognition algorithm was not trained."
            sys.exit()

        # Detects and computes the keypoints and descriptors using the sift algorithm
        keypoints, descriptors = self.faceRec.detectAndCompute(image, None)
    
        # Get all matches based on the descriptors
        matches = self.matcher.match(descriptors)

        # Order by distance
        matches = sorted(matches, key = lambda x:x.distance)
    
        # Creates a results vector to store the number of similar points for each image on the training set
        results = [0]*len(self.labels)

        # Based on the matches vector we create the results vector that represents how many points this test image are similar to each image in the training set
        for match in matches:
            if match.imgIdx >= 0 and match.imgIdx < len(results):
                results[match.imgIdx] += 1
            else:
                print "Error invalid index"

        # Index receives the position of the maximum value in the results vector (it means that this is the most similar image)
        index = results.index(max(results))

        # Calculate the confidence based on the number of matches and the max result
        # The confidence range is: 0 - 100
        # The closer to zero higher is the confidence
        if len(matches) > 0:
            confidence = 100.0 - ((float(max(results)) * 100.0) / float(len(matches)))
        else:
            confidence = 0.0

        return self.labels[index], confidence

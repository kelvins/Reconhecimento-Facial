
# Import the libraries
import cv2
import os
import sys
import numpy as np

from .voting import Voting
from .auxiliary import Auxiliary


class Ensemble(object):
    """
    Class that provides an interface for the Ensemble
    """

    def __init__(self, fr_algorithms=list(),
                 auxiliary=Auxiliary(), voting=Voting()):
        self.fr_algorithms = fr_algorithms
        self.auxiliary = auxiliary
        self.voting = voting

        self.train_images = list()
        self.train_labels = list()

        # Reset the paths
        self.train_path = ""
        self.test_path = ""

        # Reset all results
        self.recognized = 0
        self.unrecognized = 0
        self.non_faces = 0

        # Reset the predicted results
        self.predict_subject_ids = list()
        self.predict_confidence = list()

        # Reset test results
        self.test_images = list()
        self.test_labels = list()
        self.test_file_names = list()

    def reset(self):
        """
        Reset all lists and results.
        It is used to reset all values to re-train the algorithm
        """
        self.train_images = list()
        self.train_labels = list()
        # Reset the paths
        self.train_path = ""
        self.test_path = ""
        # Reset the results
        self.reset_results()

    def reset_results(self):
        """
        Reset results (including the test lists and the predictions)
        It is used to reset only the results of the tests
        """
        # Reset all results
        self.recognized = 0
        self.unrecognized = 0
        self.non_faces = 0

        # Reset the predicted results
        self.predict_subject_ids = list()
        self.predict_confidence = list()

        # Reset test results
        self.test_images = list()
        self.test_labels = list()
        self.test_file_names = list()

    def train(self, train_path):
        """
        Function responsible for train the face recognition algorithm based on the image files from the trainPath.
        """
        # Reset all lists and results
        self.reset()

        # Store the train path
        self.train_path = train_path

        if train_path == "":
            print("The train path is empty.")
            sys.exit()

        # Load all images and labels
        self.train_images, self.train_labels, _ = self.auxiliary.load_all_images_for_train(
            train_path)

        # Train all the algorithms
        for index in range(0, len(self.fr_algorithms)):
            self.fr_algorithms[index].train(
                self.train_images, self.train_labels)

    def recognize_faces(self, test_path):
        """
        Function that tries to recognize each face (path passed by parameter).
        """
        # Reset the results
        self.reset_results()

        # Store the test path
        self.test_path = test_path

        if test_path == "":
            print("The test path is empty.")
            sys.exit()

        # Load all images and labels
        self.test_images, self.test_labels, self.test_file_names = self.auxiliary.load_all_images_for_test(
            test_path)

        # For each image
        for index in range(0, len(self.test_images)):
            subject_id = list()
            confidence = list()

            # Predict
            for i in range(0, len(self.fr_algorithms)):
                sub_id, conf = self.fr_algorithms[i].predict(
                    self.test_images[index])
                subject_id.append(sub_id)
                confidence.append(conf)

            # If using weighted voting the subjectID length should be equal to
            # the weights length
            result = self.voting.vote(subject_id)

            # Store the predicted results to be used in the report
            self.predict_subject_ids.append(result)

            # As we don't work with confidences in ensemble
            # We can store the subjectID list to check which algorithm has
            # predicted which subject
            self.predict_confidence.append(subject_id)

            # Approach not using threshold (face images manually classified)
            if self.test_labels[index] >= 0:
                if result == self.test_labels[index]:
                    self.recognized += 1
                else:
                    self.unrecognized += 1
            else:
                self.non_faces += 1

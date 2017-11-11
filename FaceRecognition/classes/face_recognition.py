
# Import the libraries
import sys

from .auxiliary import Auxiliary


class FaceRecognition(object):
    """
    Class that provides an interface to the face recognition algorithms
    """

    def __init__(self, algorithm, auxiliary=Auxiliary(), threshold=-1):
        self.algorithm = algorithm
        self.auxiliary = auxiliary
        self.threshold = threshold

        # Reset all lists
        self.train_images = list()
        self.train_labels = list()
        # Reset the paths
        self.train_path = ""
        self.test_path = ""

        # Reset all results
        self.recognized = 0
        self.unrecognized = 0
        self.non_faces = 0

        # Reset all results (using threshold)
        self.recognized_below_threshold = 0
        self.unrecognized_below_threshold = 0
        self.non_faces_below_threshold = 0
        self.recognized_above_threshold = 0
        self.unrecognized_above_threshold = 0
        self.non_faces_above_threshold = 0

        # Reset the report
        self.predict_subject_ids = list()
        self.predict_confidence = list()

        # Reset test results
        self.test_images = list()
        self.test_labels = list()
        self.test_file_names = list()

    def reset(self):
        """
        Reset all values, including train and test paths
        """
        # Reset all lists
        self.train_images = list()
        self.train_labels = list()
        # Reset the paths
        self.train_path = ""
        self.test_path = ""
        # Reset the results
        self.reset_results()

    def reset_results(self):
        """
        Reset all results
        """
        # Reset all results
        self.recognized = 0
        self.unrecognized = 0
        self.non_faces = 0

        # Reset all results (using threshold)
        self.recognized_below_threshold = 0
        self.unrecognized_below_threshold = 0
        self.non_faces_below_threshold = 0
        self.recognized_above_threshold = 0
        self.unrecognized_above_threshold = 0
        self.non_faces_above_threshold = 0

        # Reset the report
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
        self.reset()

        # Store the train path
        self.train_path = train_path

        if train_path == "":
            print("The train path is empty.")
            sys.exit()

        # Load all images and labels
        self.train_images, self.train_labels, _ = self.auxiliary.load_all_images_for_train(
            train_path)

        # Train the algorithm
        self.algorithm.train(self.train_images, self.train_labels)

    def recognize_faces(self, test_path):
        """
        Function that tries to recognize each face (path passed by parameter).
        """
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
            # Predict
            subject_id, confidence = self.algorithm.predict(
                self.test_images[index])

            # Store the predicted results to be used in the report
            self.predict_subject_ids.append(subject_id)
            self.predict_confidence.append(confidence)

            # Approach not using threshold (face images manually classified)
            if self.threshold == -1:
                if self.test_labels[index] >= 0:
                    if subject_id == self.test_labels[index]:
                        self.recognized += 1
                    else:
                        self.unrecognized += 1
                else:
                    self.non_faces += 1
            # Approach using threshold
            else:
                # Compute results below threshold
                if confidence <= self.threshold:
                    if self.test_labels[index] >= 0:
                        if subject_id == self.test_labels[index]:
                            self.recognized_below_threshold += 1
                        else:
                            self.unrecognized_below_threshold += 1
                    else:
                        self.non_faces_below_threshold += 1
                # Compute results above threshold
                else:
                    if self.test_labels[index] >= 0:
                        if subject_id == self.test_labels[index]:
                            self.recognized_above_threshold += 1
                        else:
                            self.unrecognized_above_threshold += 1
                    else:
                        self.non_faces_above_threshold += 1

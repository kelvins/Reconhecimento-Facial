
# Import the libraries
import os
import time

from .voting import Voting
from .face_recognition import FaceRecognition
from .ensemble import Ensemble
from .auxiliary import Auxiliary


class Report(object):
    """
    Class that provides an interface to generate reports
    """

    def __init__(self, class_object):
        """
        Get the object (FaceRecognition or Ensemble)
        """
        self.class_object = class_object

    def generate_report_summary(self):
        """
        Generate a report summary with information about the test.
        Return the content as a string.
        """
        if isinstance(self.class_object, FaceRecognition):
            content = "Face Recognition (single algorithm)"
        elif isinstance(self.class_object, Ensemble):
            content = "Ensemble (multiple algorithms)"
        else:
            # No class object defined
            return ""

        # No predictions found
        if not self.class_object.predict_subject_ids:
            return ""

        content += "\n\nDate/Time: " + time.strftime("%d/%m/%Y %H:%M:%S")
        content += "\nTrain Path: " + self.class_object.train_path
        content += "\nTest Path: " + self.class_object.test_path + "\n"

        # For the face recognition class get only the name of the algorithm
        if isinstance(self.class_object, FaceRecognition):
            content += "\nAlgorithm: " + self.class_object.algorithm.algorithm_name
            if self.class_object.threshold >= 0:
                content += "\nThreshold Used: " + \
                    str(self.class_object.threshold)
            else:
                content += "\nThreshold Not Used."

        # For the Ensemble class get the name of all algorithms
        elif isinstance(self.class_object, Ensemble):
            content += "\nVoting Scheme: " + self.class_object.voting.get_voting_scheme_name()
            weights = self.class_object.voting.weights

            for index in range(0, len(self.class_object.fr_algorithms)):
                content += "\nAlgorithm: " + \
                    self.class_object.fr_algorithms[index].algorithm_name
                # If it is using the WEIGHTED voting scheme
                if self.class_object.voting.voting_scheme == Voting().weighted:
                    # If the index is valid for the weights list
                    if index < len(weights):
                        content += " - Weight: " + str(weights[index])

        content += "\n\nTotal Images Analyzed: " + \
            str(len(self.class_object.test_file_names))

        accuracy2 = 0.0

        if isinstance(self.class_object, FaceRecognition):
            if self.class_object.threshold >= 0:
                total_face_images = self.class_object.recognized_below_threshold + \
                    self.class_object.unrecognized_below_threshold
                # Calculate the accuracy using only the results below the
                # threshold
                accuracy2 = Auxiliary.calc_accuracy(
                    self.class_object.recognized_below_threshold, total_face_images)

                total_face_images += self.class_object.recognized_above_threshold + \
                    self.class_object.unrecognized_above_threshold
                # Calculate the accuracy using the total number of face images
                accuracy = Auxiliary.calc_accuracy(
                    self.class_object.recognized_below_threshold, total_face_images)

                content += "\nRecognized Faces Below Threshold: " + \
                    str(self.class_object.recognized_below_threshold)
                content += "\nUnrecognized Faces Below Threshold: " + \
                    str(self.class_object.unrecognized_below_threshold)
                content += "\nNon Faces Below Threshold: " + \
                    str(self.class_object.non_faces_below_threshold)
                content += "\nRecognized Faces Above Threshold: " + \
                    str(self.class_object.recognized_above_threshold)
                content += "\nUnrecognized Faces Above Threshold: " + \
                    str(self.class_object.unrecognized_above_threshold)
                content += "\nNon Faces Above Threshold: " + \
                    str(self.class_object.non_faces_above_threshold)
            else:
                total_face_images = float(
                    self.class_object.recognized + self.class_object.unrecognized)
                accuracy = Auxiliary.calc_accuracy(
                    self.class_object.recognized, total_face_images)
                content += "\nRecognized Faces: " + \
                    str(self.class_object.recognized)
                content += "\nUnrecognized Faces: " + \
                    str(self.class_object.unrecognized)
                content += "\nNon Faces: " + str(self.class_object.non_faces)
        else:
            total_face_images = float(
                self.class_object.recognized + self.class_object.unrecognized)
            accuracy = Auxiliary.calc_accuracy(
                self.class_object.recognized, total_face_images)
            content += "\nRecognized Faces: " + \
                str(self.class_object.recognized)
            content += "\nUnrecognized Faces: " + \
                str(self.class_object.unrecognized)
            content += "\nNon Faces: " + str(self.class_object.non_faces)

        content += "\nRecognition Rate - Recognized / Total Face Images"
        content += "\nAccuracy: " + str(accuracy) + " %"

        if isinstance(self.class_object, FaceRecognition):
            if self.class_object.threshold >= 0:
                content += "\nAccuracy Only Below Threshold: " + \
                    str(accuracy2) + " %"

        size_x, size_y = self.class_object.auxiliary.get_default_size()
        content += "\n\nDefault Size Images: " + \
            str(size_x) + "x" + str(size_y)
        content += "\nInterpolation Method: " + \
            self.class_object.auxiliary.get_interpolation_method_name()
        content += "\nSupported Files: " + \
            ', '.join(self.class_object.auxiliary.supported_files)
        return content

    def generate_full_report(self):
        """
        Generate the full report.
        Return the content containing the information about each predicted image.
        """
        # Get the predicted results
        predict_subject_ids = self.class_object.predict_subject_ids
        predict_confidence = self.class_object.predict_confidence
        # Get the test information (labels and filenames)
        test_labels = self.class_object.test_labels
        test_file_names = self.class_object.test_file_names

        content = ""

        # Create each line based on the predicted subject IDs
        for index in range(0, len(predict_subject_ids)):
            # Format: 1: Expected subject: 3: Classified as subject: 2: With
            # confidence: 4123.123123: File name: 1_3
            content += str(index + 1)
            content += ": Expected subject: " + str(test_labels[index])
            content += ": Classified as subject: " + \
                str(predict_subject_ids[index])

            if isinstance(self.class_object, FaceRecognition):
                content += ": With confidence: " + \
                    str(predict_confidence[index])
            elif isinstance(self.class_object, Ensemble):
                content += ": Predicted Subjects: " + \
                    ', '.join(map(str, predict_confidence[index]))

            content += ": File name: " + test_file_names[index]
            content += "\n"

        return content

    def print_results(self):
        """
        Function used to show the results
        """
        print("========================= Results =========================")
        print(self.generate_report_summary())
        print("===========================================================")

    def save_report(self, path=""):
        """
        Function used to automatically save the report in a defined folder.
        Save only the text report not the images.
        """

        # Generate the report content
        content = self.generate_report_summary()
        content += "\n===========================================================\n"
        content += self.generate_full_report()

        # Make sure that none folder will have the same name
        time.sleep(1)

        # If the parameters were set include it in the folder name
        file_name = time.strftime("%Y_%m_%d_%H_%M_%S") + ".txt"

        # If the path is not empty use it in the filename
        if path != "":
            # If the path does not exist, create it
            if not os.path.exists(path):
                os.makedirs(path)

            if path.endswith(".txt"):
                file_name = path
            elif path.endswith("/"):
                file_name = path + file_name
            else:
                file_name = path + "/" + file_name

        # Save the text file
        Auxiliary.write_text_file(content, file_name)

    def save_all_results(self, path=""):
        """
        Function used to automatically save the report in a defined folder.
        Save the entire results, including the summary report, full report and all images.
        """

        # If the path is not empty use it in the filename
        if path != "":
            if path.endswith("/") is not True:
                path += "/"

        # Defined the name of the new folder
        path += time.strftime("%Y_%m_%d_%H_%M_%S") + "/"

        # If the path does not exist, create it
        if not os.path.exists(path):
            os.makedirs(path)

        # Save the report
        self.save_report(path)

        # Create 3 new folders
        recognized_folder = path + "Recognized/"
        unrecognized_folder = path + "Unrecognized/"
        non_faces_folder = path + "NonFaces/"

        os.makedirs(recognized_folder)
        os.makedirs(unrecognized_folder)
        os.makedirs(non_faces_folder)

        # The predicted results
        predict_subject_ids = self.class_object.predict_subject_ids
        predict_confidence = self.class_object.predict_confidence
        # The tests information
        test_images = self.class_object.test_images
        test_labels = self.class_object.test_labels
        # test_file_names = self.class_object.testFileNames
        # The training information
        train_images = self.class_object.train_images
        train_labels = self.class_object.train_labels

        delimiter = "_"

        for index in range(0, len(predict_subject_ids)):
            # Patter: 1_Expected_2_Classified_2_Confidence_40192.12938291.png
            label = str(index) + delimiter + "Expected" + \
                delimiter + str(test_labels[index]) + delimiter
            label += "Classified" + delimiter + \
                str(predict_subject_ids[index]) + delimiter

            if isinstance(self.class_object, FaceRecognition):
                label += "Confidence" + delimiter + \
                    str(predict_confidence[index])
            elif isinstance(self.class_object, Ensemble):
                label += "Voting" + delimiter + self.class_object.voting.get_voting_scheme_name()

            label += ".png"

            # Find the image that matches based on the trainLabel and
            # predictedSubjectIDs
            image1 = test_images[index]
            image2 = None
            for i in range(0, len(train_labels)):
                if str(train_labels[i]) == str(predict_subject_ids[index]):
                    image2 = train_images[i]

            # Concatenate the images
            image = Auxiliary.concatenate_images(image1, image2)

            # Get the correct fileName
            if str(test_labels[index]) == "-1":
                file_name = non_faces_folder
            elif str(test_labels[index]) == str(predict_subject_ids[index]):
                file_name = recognized_folder
            else:
                file_name = unrecognized_folder

            file_name += label

            # Save the concatenated image in the correct folder
            Auxiliary.save_image(file_name, image)

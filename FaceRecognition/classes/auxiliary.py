
# Import the libraries
import cv2
import os
import numpy as np


class Auxiliary(object):
    """
    Class that provides some auxiliary functions.
    """

    def __init__(self, size_x=100, size_y=100, interpolation=cv2.INTER_CUBIC):
        """
        Set the default values for the image size and the interpolation method.
        Available interpolation methods provided by OpenCV: INTER_CUBIC, INTER_AREA, INTER_LANCZOS4, INTER_LINEAR, INTER_NEAREST
        :param size_x: Set the default image width (default = 100).
        :param size_y: Set the default image height (default = 100).
        :param interpolation: Set the default interpolation method (default cv2.INTER_CUBIC).
        """
        self.size_x = size_x
        self.size_y = size_y
        self.interpolation = interpolation

        # Declare all supported files
        self.supported_files = ["png", "jpg", "jpeg"]

    def set_default_size(self, size_x, size_y):
        """
        Set the default size.
        :param size_x: Image width.
        :param size_y: Image height.
        """
        if size_x > 0:
            self.size_x = size_x
        if size_y > 0:
            self.size_y = size_y

    def get_default_size(self):
        """
        Get the default image size defined (default is 100x100).
        """
        return self.size_x, self.size_y

    def get_interpolation_method_name(self):
        """
        Get the selected interpolation method name.
        :return: A string containing the interpolation method name.
        """
        if self.interpolation == cv2.INTER_CUBIC:
            return "cv2.INTER_CUBIC"
        if self.interpolation == cv2.INTER_AREA:
            return "cv2.INTER_AREA"
        if self.interpolation == cv2.INTER_LANCZOS4:
            return "cv2.INTER_LANCZOS4"
        if self.interpolation == cv2.INTER_LINEAR:
            return "cv2.INTER_LINEAR"
        if self.interpolation == cv2.INTER_NEAREST:
            return "cv2.INTER_NEAREST"

        raise NameError("Invalid interpolation method name")
        return ""

    @staticmethod
    def calc_accuracy(recognized_images, total_face_images):
        """
        Calculates the accuracy (percentage) using the formula:
        acc = (recognized_images / total_face_images) * 100
        :param recognized_images: The number of recognized face images.
        :param total_face_images: The number of total face images.
        :return: The accuracy.
        """
        try:
            return (float(recognized_images) /
                    float(total_face_images)) * 100.0
        except ZeroDivisionError:
            return 0.0

    @staticmethod
    def write_text_file(content, file_name):
        """
        Write the content to a text file based on the file name.
        :param content: The content as a string.
        :param file_name: The file name (e.g. home/user/test.txt)
        """
        # Save the text file
        text_file = open(file_name, "w")
        text_file.write(content)
        text_file.close()

    @staticmethod
    def is_grayscale(image):
        """
        Check if an image is in grayscale.
        :param image: The image.
        :return: True if the image is in grayscale.
        """
        if len(image.shape) <= 2:
            return True

        h, w = image.shape[:2]  # rows, cols, channels
        for i in range(w):
            for j in range(h):
                p = image[i, j]
                if p[0] != p[1] != p[2]:
                    return False
        return True

    @staticmethod
    def to_grayscale(image):
        """
        Convert an image to grayscale
        :param image: The image.
        :return: The image in grayscale.
        """
        if image is None:
            print("Invalid Image: Could not convert to grayscale")
            return None
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def load_image(path):
        """
        Load an image based on the path passed by parameter.
        :param path: The path to the image file.
        :return: The image object.
        """
        return cv2.imread(path)

    @staticmethod
    def save_image(file_name, image):
        """
        Save an image based on the fileName passed by parameter.
        :param file_name: The file name.
        :param image: The image.
        """
        cv2.imwrite(file_name, image)

    @staticmethod
    def resize_image(image, size_x, size_y, interpolation_method):
        """
        Resize an image.
        :param image: The image object.
        :param size_x: The image width.
        :param size_y: The image height.
        :param interpolation_method: The interpolation method.
        :return: The resized image.
        """
        if image is None:
            print("Invalid Image: Could not be resized")
            return -1

        rows, cols = image.shape
        if rows <= 0 or cols <= 0:
            print("Invalid Image Sizes: Could not be resized")
            return -1

        return cv2.resize(image, (size_x, size_y),
                          interpolation=interpolation_method)

    def preprocess_image(self, path):
        """
        Preprocess an image. Load an image, convert to grayscale and resize it.
        :param path: The image path.
        :return: The preprocessed image.
        """
        # Load the image
        image = self.load_image(path)

        if image is None:
            print("Could not load the image:", path)
            return None

        # Convert to grayscale
        image = self.to_grayscale(image)
        # Resize the image
        image = self.resize_image(
            image, self.size_x, self.size_y, self.interpolation)
        # Return the processed image
        return image

    @staticmethod
    def concatenate_images(left_image, right_image):
        """
        Concatenate two images side by side (horizontally) and returns a new one.
        :param left_image: The image that should be put to the left.
        :param right_image: The image that should be put to the right.
        :return: The new concatenated image.
        """
        try:
            return np.concatenate((left_image, right_image), axis=1)
        except ValueError:
            return None

    def extract_images_paths(self, path):
        """
        Extract all paths for each image in a directory.
        :param path: The directory path.
        :return: A list with all file paths.
        """
        paths = []

        # In the path folder search for all files in all directories
        for dir_name, dir_names, file_names in os.walk(path):
            # For each file found
            for file_name in file_names:
                # Check if it is a valid image file
                if file_name.split(".")[1] in self.supported_files:
                    # Creates the filePath joining the directory name and the
                    # file name
                    paths.append(os.path.join(dir_name, file_name))

        return paths

    @staticmethod
    def extract_files_paths(path):
        """
        Extract all paths for all files type.
        :param path: The directory path.
        :return: A list with all paths for all files.
        """
        paths = []

        # In the path folder search for all files in all directories
        for dir_name, dir_names, file_names in os.walk(path):
            # For each file found
            for file_name in file_names:
                # Creates the filePath joining the directory name and the file
                # name
                paths.append(os.path.join(dir_name, file_name))

        return paths

    def load_all_images_for_train(self, train_path):
        """
        Load all images for training.
        :param train_path: The train path.
        :return: Three lists with the images, labels and file names.
        """
        images = []
        labels = []
        file_name = []

        paths = self.extract_images_paths(train_path)

        # For each file path
        for file_path in paths:
            # Check if it is a valid image file
            if file_path.split(".")[1] in self.supported_files:

                # Get the subject id (label) based on the format:
                # subjectID_imageNumber.png
                path_split = file_path.split("/")
                temp_name = path_split[len(path_split) - 1]
                subject_id = int(temp_name.split("_")[0])

                images.append(self.preprocess_image(file_path))
                labels.append(subject_id)
                file_name.append(temp_name.split(".")[0])

        return images, labels, file_name

    def load_all_images_for_test(self, test_path):
        """
        Load all images for test.
        :param test_path: The test path.
        :return: Three lists with the images, labels and file names.
        """
        images = []
        labels = []
        file_name = []

        paths = self.extract_images_paths(test_path)

        # For each file path
        for file_path in paths:

            # Check if it is a valid image file
            if file_path.split(".")[1] in self.supported_files:

                # Get the subject id (label)
                # IMPORTANT: it follows the pattern: imageNumber_subjectID.png
                # It is different from the pattern on the training set
                path_split = file_path.split("/")
                temp_name = path_split[len(path_split) - 1]
                subject_id = temp_name.split("_")[1]
                subject_id = int(subject_id.split(".")[0])

                image = self.preprocess_image(file_path)

                if image is None:
                    return None, None, None

                images.append(image)
                labels.append(subject_id)
                file_name.append(temp_name.split(".")[0])

        return images, labels, file_name

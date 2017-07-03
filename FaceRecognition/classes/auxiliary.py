
# Import the libraries
import cv2
import os
import time
from PIL import Image
import numpy as np


class Auxiliary:
    """
    Class that provides some auxiliary functions.
    """

    def __init__(self, sizeX=100, sizeY=100, interpolation=cv2.INTER_CUBIC):
        """
        Set the default values for the image size and the interpolation method.
        Available interpolation methods provided by OpenCV: INTER_CUBIC, INTER_AREA, INTER_LANCZOS4, INTER_LINEAR, INTER_NEAREST
        :param sizeX: Set the default image width (default = 100).
        :param sizeY: Set the default image height (default = 100).
        :param interpolation: Set the default interpolation method (default cv2.INTER_CUBIC).
        """
        self.sizeX = sizeX
        self.sizeY = sizeY
        self.interpolation = interpolation

        # Declare all supported files
        self.supportedFiles = ["png", "jpg", "jpeg"]

    def setDefaultSize(self, sizeX, sizeY):
        """
        Set the default size.
        :param sizeX: Image width.
        :param sizeY: Image height.
        """
        if sizeX > 0:
            self.sizeX = sizeX
        if sizeY > 0:
            self.sizeY = sizeY

    def getDefaultSize(self):
        """
        Get the default image size defined (default is 100x100).
        """
        return self.sizeX, self.sizeY

    def getInterpolationMethodName(self):
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
        return ""

    def calcAccuracy(self, recognizedImages, totalFaceImages):
        """
        Calculates the accuracy (percentage) using the formula:
        acc = recognizedImages / totalFaceImages * 100
        :param recognizedImages: The number of recognized face images.
        :param totalFaceImages: The number of total face images.
        :return: The accuracy.
        """
        # Avoid division by zero
        if totalFaceImages > 0:
            return (float(recognizedImages) / float(totalFaceImages)) * 100.0
        else:
            return 0.0

    def writeTextFile(self, content, fileName):
        """
        Write the content to a text file based on the file name.
        :param content: The content as a string.
        :param fileName: The file name (e.g. home/user/test.txt)
        """
        # Save the text file
        textFile = open(fileName, "w")
        textFile.write(content)
        textFile.close()

    def isGrayscale(self, image):
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

    def toGrayscale(self, image):
        """
        Convert an image to grayscale
        :param image: The image.
        :return: The image in grayscale.
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def loadImage(self, path):
        """
        Load an image based on the path passed by parameter.
        :param path: The path to the image file.
        :return: The image object.
        """
        return cv2.imread(path)

    def saveImage(self, fileName, image):
        """
        Save an image based on the fileName passed by parameter.
        :param fileName: The file name.
        :param image: The image.
        """
        cv2.imwrite(fileName, image)

    def resizeImage(self, image, sizeX, sizeY, interpolationMethod):
        """
        Resize an image.
        :param image: The image object.
        :param sizeX: The image width.
        :param sizeY: The image height.
        :param interpolationMethod: The interpolation method.
        :return: The resized image.
        """
        return cv2.resize(image, (sizeX, sizeY),
                          interpolation=interpolationMethod)

    def preprocessImage(self, path):
        """
        Preprocess an image. Load an image, convert to grayscale and resize it.
        :param path: The image path.
        :return: The preprocessed image.
        """
        # Load the image
        image = self.loadImage(path)
        # Convert to grayscale
        image = self.toGrayscale(image)
        # Resize the image
        image = self.resizeImage(
            image, self.sizeX, self.sizeY, self.interpolation)
        # Return the processed image
        return image

    def concatenateImages(self, leftImage, rightImage):
        """
        Concatenate two images side by side (horizontally) and returns a new one.
        :param leftImage: The image that should be put to the left.
        :param rightImage: The image that should be put to the right.
        :return: The new concatenated image.
        """
        return np.concatenate((leftImage, rightImage), axis=1)

    def extractImagesPaths(self, path):
        """
        Extract all paths for each image in a directory.
        :param path: The directory path.
        :return: A list with all file paths.
        """
        paths = []

        # In the path folder search for all files in all directories
        for dirname, dirnames, filenames in os.walk(path):
            # For each file found
            for filename in filenames:
                # Check if it is a valid image file
                if filename.split(".")[1] in self.supportedFiles:
                    # Creates the filePath joining the directory name and the
                    # file name
                    paths.append(os.path.join(dirname, filename))

        return paths

    def extractFilesPaths(self, path):
        """
        Extract all paths for all files type.
        :param path: The directory path.
        :return: A list with all paths for all files.
        """
        paths = []

        # In the path folder search for all files in all directories
        for dirname, dirnames, filenames in os.walk(path):
            # For each file found
            for filename in filenames:
                # Creates the filePath joining the directory name and the file
                # name
                paths.append(os.path.join(dirname, filename))

        return paths

    def loadAllImagesForTrain(self, trainPath):
        """
        Load all images for training.
        :param trainPath: The train path.
        :return: Three lists with the images, labels and file names.
        """
        images = []
        labels = []
        fileName = []

        paths = self.extractImagesPaths(trainPath)

        # For each file path
        for filePath in paths:
            # Check if it is a valid image file
            if filePath.split(".")[1] in self.supportedFiles:

                # Get the subject id (label) based on the format:
                # subjectID_imageNumber.png
                pathSplit = filePath.split("/")
                tempName = pathSplit[len(pathSplit) - 1]
                subjectID = int(tempName.split("_")[0])

                images.append(self.preprocessImage(filePath))
                labels.append(subjectID)
                fileName.append(tempName.split(".")[0])

        return images, labels, fileName

    def loadAllImagesForTest(self, testPath):
        """
        Load all images for test.
        :param testPath: The test path.
        :return: Three lists with the images, labels and file names.
        """
        images = []
        labels = []
        fileName = []

        paths = self.extractImagesPaths(testPath)

        # For each file path
        for filePath in paths:

            # Check if it is a valid image file
            if filePath.split(".")[1] in self.supportedFiles:

                # Get the subject id (label)
                # IMPORTANT: it follows the patter: imageNumber_subjectID.png
                # It is different from the pattern on the training set
                pathSplit = filePath.split("/")
                tempName = pathSplit[len(pathSplit) - 1]
                subjectID = tempName.split("_")[1]
                subjectID = int(subjectID.split(".")[0])

                images.append(self.preprocessImage(filePath))
                labels.append(subjectID)
                fileName.append(tempName.split(".")[0])

        return images, labels, fileName

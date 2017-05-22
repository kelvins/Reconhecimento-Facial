
# Import the libraries
import cv2
import os
import time
from PIL import Image
import numpy as np

class Auxiliary:
    """
    Class that provides some auxiliary functions
    """

    def __init__(self, sizeX=100, sizeY=100, interpolation=cv2.INTER_CUBIC):
        """
        Set the default values
        """
        self.sizeX = sizeX
        self.sizeY = sizeY
        self.interpolation = interpolation
        # INTER_CUBIC, INTER_AREA, INTER_LANCZOS4, INTER_LINEAR, INTER_NEAREST

        # Declare all supported files
        self.supportedFiles = ["png", "jpg", "jpeg"]

    def setDefaultSize(self, sizeX, sizeY):
        """
        Set the default size for the imagens (default is 100x100)
        """
        if sizeX > 0:
            self.sizeX = sizeX
        if sizeY > 0:
            self.sizeY = sizeY

    def getDefaultSize(self):
        """
        Get the default size defined (default is 100x100)
        """
        return self.sizeX, self.sizeY

    def setSupportedFiles(self, supportedFiles):
        """
        Set the default supportedFiles list (default is ["png", "jpg", "jpeg"])
        """
        self.supportedFiles = supportedFiles

    def getSupportedFiles(self):
        """
        Set the supportedFiles list (default is ["png", "jpg", "jpeg"])
        """
        return self.supportedFiles

    def setInterpolation(self, interpolation):
        """
        Set the default interpolation method (default is cv2.INTER_CUBIC)
        """
        self.interpolation = interpolation

    def getInterpolationMethodName(self):
        """
        Get the selected interpolation method
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

    def writeTextFile(self, content, fileName):
        """
        Write the content to a text file based on the file name
        """
        # Save the text file
        textFile = open(fileName, "w")
        textFile.write(content)
        textFile.close()

    def isGrayscale(self, image):
        """
        Check if an image is in grayscale
        """
        if len(image.shape) <= 2:
            return True

        h, w = image.shape[:2] # rows, cols, channels
        for i in range(w):
            for j in range(h):
                p = image[i, j]
                if p[0] != p[1] != p[2]:
                    return False
        return True

    def toGrayscale(self, image):
        """
        Convert an image to grayscale
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def loadImage(self, path):
        """
        Load an image based on the path passed by parameter
        """
        return cv2.imread(path)

    def saveImage(self, fileName, image):
        """
        Save an image based on the fileName passed by parameter
        """
        cv2.imwrite(fileName, image)

    def resizeImage(self, image, sizeX, sizeY, interpolationMethod):
        """
        Convert an image to grayscale
        """
        return cv2.resize(image, (sizeX, sizeY), interpolation=interpolationMethod)

    def preprocessImage(self, path):
        """
        Preprocess an image
        """
        # Load the image
        image = self.loadImage(path)
        # Convert to grayscale
        image = self.toGrayscale(image)
        # Resize the image
        image = self.resizeImage(image, self.sizeX, self.sizeY, self.interpolation)
        # Return the processed image
        return image

    def concatenateImages(self, leftImage, rightImage):
        """
        Concatenate two images side by side (horizontally) and returns a new one
        """
        return np.concatenate((leftImage, rightImage), axis=1)

    def extractImagesPaths(self, path):
        """
        Extract all paths for each image
        """
        paths = []

         # In the path folder search for all files in all directories
        for dirname, dirnames, filenames in os.walk(path):
            # For each file found
            for filename in filenames:
                # Check if it is a valid image file
                if filename.split(".")[1] in self.supportedFiles:
                    # Creates the filePath joining the directory name and the file name
                    paths.append( os.path.join(dirname, filename) )

        return paths

    def extractFilesPaths(self, path):
        """
        Extract all paths for all files type
        """
        paths = []

         # In the path folder search for all files in all directories
        for dirname, dirnames, filenames in os.walk(path):
            # For each file found
            for filename in filenames:
                # Creates the filePath joining the directory name and the file name
                paths.append( os.path.join(dirname, filename) )

        return paths

    def loadAllImagesForTrain(self, trainPath):
        """
        Load all images for train
        """
        images   = []
        labels   = []
        fileName = []

        paths = self.extractImagesPaths(trainPath)

         # For each file path
        for filePath in paths:
            # Check if it is a valid image file
            if filePath.split(".")[1] in self.supportedFiles:

                # Get the subject id (label) based on the format: subjectID_imageNumber.png
                pathSplit = filePath.split("/")
                tempName  = pathSplit[len(pathSplit)-1]
                subjectID = int(tempName.split("_")[0])

                images.append( self.preprocessImage(filePath) )
                labels.append( subjectID )
                fileName.append( tempName.split(".")[0] )

        return images, labels, fileName

    def loadAllImagesForTest(self, testPath):
        """
        Load all images for test
        """
        images   = []
        labels   = []
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
                tempName  = pathSplit[len(pathSplit)-1]
                subjectID = tempName.split("_")[1]
                subjectID = int(subjectID.split(".")[0])

                images.append( self.preprocessImage(filePath) )
                labels.append( subjectID )
                fileName.append( tempName.split(".")[0] )

        return images, labels, fileName

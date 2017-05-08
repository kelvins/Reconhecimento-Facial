
# Import the libraries
import cv2
import os

class Auxiliary:
    """
    Class that provides some auxiliary functions
    """

    def __init__(self):
        """
        Set the default values
    	"""
        self.sizeX = 100
        self.sizeY = 100
        self.interpolation = cv2.INTER_CUBIC
        # INTER_CUBIC, INTER_AREA, INTER_LANCZOS4, INTER_LINEAR, INTER_NEAREST

        # Declare all supported files
        self.supportedFiles = ["png", "jpg", "jpeg"]

    def setDefaultSize(self, sizeX, sizeY):
        """
        Set the default size for the imagens (default is 100x100)
        """
        self.sizeX = sizeX
        self.sizeY = sizeY

    def setInterpolation(self, interpolation):
        """
        Set the default interpolation method (default is cv2.INTER_CUBIC)
        """
        self.interpolation = interpolation

    def setSupportedFiles(self, supportedFiles):
        """
        Set the default supportedFiles list (default is ["png", "jpg", "jpeg"])
        """
        self.supportedFiles = supportedFiles

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
        image = loadImage(path)
        # Convert to grayscale
        image = toGrayscale(image)
        # Resize the image
        image = resizeImage(image, self.sizeX, self.sizeY, self.interpolation)
        # Return the processed image
        return image

    def extractImagesPaths(self, path):
        """
        Extract all paths for each image
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
        images = []
        labels = []

        paths = extractImagesPaths(trainPath)

         # For each file path
        for filePath in paths:
            # Check if it is a valid image file
            if filePath.split(".")[1] in self.supportedFiles:

                # Get the subject id (label) based on the format: subjectID_imageNumber.png
                pathSplit = filePath.split("/")
                tempName  = pathSplit[len(pathSplit)-1]
                subjectID = int(tempName.split("_")[0])

                images.append( preprocessImage(filePath) )
                labels.append( subjectID )

        return images, labels

    def loadAllImagesForTest(self, testPath):
        """
        Load all images for test
        """
        images = []
        labels = []

        paths = extractImagesPaths(testPath)

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

                images.append( preprocessImage(filePath) )
                labels.append( subjectID )
                    
        return images, labels

    def printResults(self, content):
        """
        Function used to show the results
        """
        print "========================= Results =========================\n"
        print content
        print "==========================================================="

    def saveResults(self, content, path=""):
        """
        Function used to automatically save the results in a defined folder
        """

        # Make sure that none folder will have the same name
        time.sleep(1)

        # If the parameters were set include it in the folder name
        fileName = time.strftime("%Y_%m_%d_%H_%M_%S") + ".txt"

        # If the path is not empty use it in the filename
        if path != "":
            fileName = path + "/" + fileName

        # Save the text file
        textFile = open(fileName, "w")
        textFile.write(content)
        textFile.close()

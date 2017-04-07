
# Import the libraries
import cv2
import os
import sys
import time
import numpy as np

class Algorithms:
    """
    Algorithms is used to define the face recognition algorithm.
    """
    EIGENFACES, FISHERFACES, LBPH, SIFT, SURF = range(5)


class Interpolation:
    """
    Interpolation is used to define the interpolation method used to resize the images.
    """
    INTER_CUBIC, INTER_NEAREST, INTER_LINEAR, INTER_AREA, INTER_LANCZOS4 = range(5)

supported_files = ["png", "jpg", "jpeg"]

# Class that encapsulates all 5 face recognition algorithms
class FaceRecognition:

    def __init__(self):
    	"""
    	Set the default values
    	Size: 100x100
    	Algorithm: Eigenfaces
    	Interpolation: INTER_CUBIC (bicubic interpolation)
    	"""
        self.sizeX = 100
        self.sizeY = 100
        self.algorithm = Algorithms.EIGENFACES
        self.interpolation = Interpolation.INTER_CUBIC
        self.interpolationTitle = "INTER_CUBIC"

        # Eigenfaces paramters
        self.eigenfacesParameter = 0

        # Fisherfaces parameters
        self.fisherfacesParameter = 0

        # LBPH Parameters
        self.radius = 1
        self.neighbors = 8
        self.gridX = 8
        self.gridY = 8

    def newFaceRecAlgorithm(self):
    	"""
    	Creates the face recognition object based on the selected algorithm.
    	Set some values to the report.
    	Reset the results.
    	"""

        self.parameters = ""

        # Vector used to store the training images
        self.trainingImages = []

        # Vector used to store the training images labels
        self.labels = []

        self.content = "Date/Time : " + \
            time.strftime("%d/%m/%Y %H:%M:%S") + "\n"

        # Creates the face recognition object based on the selected algorithm
        if self.algorithm == Algorithms.EIGENFACES:

            if self.eigenfacesParameter == 0:
                self.faceRec = cv2.face.createEigenFaceRecognizer()
            else:
                self.faceRec = cv2.face.createEigenFaceRecognizer(
                    self.eigenfacesParameter)

            self.algorithmTitle = "EIGENFACES"
            self.parameters = str(self.eigenfacesParameter)

        elif self.algorithm == Algorithms.FISHERFACES:

            if self.fisherfacesParameter == 0:
                self.faceRec = cv2.face.createFisherFaceRecognizer()
            else:
                self.faceRec = cv2.face.createFisherFaceRecognizer(
                    self.fisherfacesParameter)

            self.algorithmTitle = "FISHERFACES"
            self.parameters = str(self.fisherfacesParameter)

        elif self.algorithm == Algorithms.LBPH:

            self.faceRec = cv2.face.createLBPHFaceRecognizer(
                radius=self.radius, neighbors=self.neighbors, grid_x=self.gridX, grid_y=self.gridY)
            self.algorithmTitle = "LBPH"

        elif self.algorithm == Algorithms.SIFT:

            self.faceRec = cv2.xfeatures2d.SIFT_create()
            self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            self.algorithmTitle = "SIFT"

        elif self.algorithm == Algorithms.SURF:

            self.faceRec = cv2.xfeatures2d.SURF_create()  # 400
            self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            self.algorithmTitle = "SURF"

        else:
            print "Invalid algorithm selected."
            sys.exit()

        self.content += "Algorithm : " + self.algorithmTitle + "\n"
        self.content += "Parameters : " + self.parameters + "\n"

        self.recognizedFaces = 0
        self.unrecognizedFaces = 0
        self.nonFaces = 0

        self.recognizedFacesImages = []
        self.unrecognizedFacesImages = []
        self.nonFacesImages = []

    def setAlgorithm(self, algorithm):
        """
        Set the selected algorithm.

        :param algorithm: The selected algorithm (e.g. Algorithms.EIGENFACES)
        """
        self.algorithm = algorithm

    def setEigenfacesParameter(self, parameter):
        """
        Set the eigenfaces parameters.

        :param algorithm: The parameter for the eigenfaces method.
        """
        self.eigenfacesParameter = parameter

    def getEigenfacesParameter(self):
        """
        Get the eigenfaces parameters.

        :return: the selected parameter for the eigenfaces method.
        """
        return self.eigenfacesParameter

    def setFisherfacesParameter(self, parameter):
        self.fisherfacesParameter = parameter

    def getFisherfacesParameter(self):
        return self.fisherfacesParameter

    def setLBPHParameters(self, radius, neighbors, gridX, gridY):
        self.radius = radius
        self.neighbors = neighbors
        self.gridX = gridX
        self.gridY = gridY

    def getLBPHParameters(self):
        return self.radius, self.neighbors, self.gridX, self.gridY

    def getRecognizedFacesImages(self):
        return self.recognizedFacesImages

    def getUnrecognizedFacesImages(self):
        return self.unrecognizedFacesImages

    def getNonFacesImages(self):
        return self.nonFacesImages

    def getContent(self):
        return self.content

    def getAlgorithmTitle(self):
        return self.algorithmTitle

    def getInterpolationTitle(self):
        return self.interpolationTitle

    def getRecognizedFaces(self):
        return self.recognizedFaces

    def getUnrecognizedFaces(self):
        return self.unrecognizedFaces

    def getNonFaces(self):
        return self.nonFaces

    def getInterpolation(self):
        return self.interpolation

    def setInterpolation(self, interpolation):
        self.interpolation = interpolation

    def setDefaultSize(self, sizeX, sizeY):
        self.sizeX = sizeX
        self.sizeY = sizeY

    def setSizeX(self, sizeX):
        self.sizeX = sizeX

    def setSizeY(self, sizeY):
        self.sizeY = sizeY

    def preprocessImage(self, path):
        """
        Function responsible for load the image, convert to grayscale and resize to a default size.
        """
        # Loads the image into the image variable
        image = cv2.imread(path)
        # Convert the image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize the image to a default size
        if self.interpolation == Interpolation.INTER_CUBIC:
            image = cv2.resize(image, (self.sizeX, self.sizeY),
                               interpolation=cv2.INTER_CUBIC)
            self.interpolationTitle = "INTER_CUBIC"
        elif self.interpolation == Interpolation.INTER_AREA:
            image = cv2.resize(image, (self.sizeX, self.sizeY),
                               interpolation=cv2.INTER_AREA)
            self.interpolationTitle = "INTER_AREA"
        elif self.interpolation == Interpolation.INTER_LANCZOS4:
            image = cv2.resize(image, (self.sizeX, self.sizeY),
                               interpolation=cv2.INTER_LANCZOS4)
            self.interpolationTitle = "INTER_LANCZOS4"
        elif self.interpolation == Interpolation.INTER_LINEAR:
            image = cv2.resize(image, (self.sizeX, self.sizeY),
                               interpolation=cv2.INTER_LINEAR)
            self.interpolationTitle = "INTER_LINEAR"
        elif self.interpolation == Interpolation.INTER_NEAREST:
            image = cv2.resize(image, (self.sizeX, self.sizeY),
                               interpolation=cv2.INTER_NEAREST)
            self.interpolationTitle = "INTER_NEAREST"
        else:
            print "Error trying to resize the image."
            sys.exit()

        return image

    def includeFace(self, path):
        """
        Function that receives the path of each face image as parameter and include it in the training set (bf object).
        """
        if path.split(".")[1] in supported_files:
            # Get the subject id (should be a number)
            subjectID = int(path.split("_")[1].split(".")[0])

            # Load Image, Convert to Grayscale, Resize
            image = self.preprocessImage(path)

            if self.algorithm == Algorithms.SIFT or self.algorithm == Algorithms.SURF:
                # Detects and computes the keypoints and descriptors using the SURF
                # algorithm
                keypoints, descriptors = self.faceRec.detectAndCompute(image, None)

                # Creates an numpy array
                clusters = np.array([descriptors])

                # Add the array to the BFMatcher
                self.bf.add(clusters)

            return image, subjectID
        return None, 0

    def train(self, trainPath):
        """
        Function responsible for train the face recognition algorithm based on the image files from the trainPath.
        """

        # Creates a new face recognition object and clear the variables
        self.newFaceRecAlgorithm()

        if trainPath == "":
            print "Empty train path."
            sys.exit()

        self.content += "TrainPath : " + trainPath + "\n"

        # In the trainingPath folder search for all files in all directories
        for dirname, dirnames, filenames in os.walk(trainPath):
            # For each file found
            for filename in filenames:
                # Creates the filePath joining the directory name and the file name
                filePath = os.path.join(dirname, filename)

                # Include the image file in the training set
                image, subjectID = self.includeFace(filePath)

                if image is not None:
                    # Store the image to generate an output image
                    self.trainingImages.append(image)

                    # Store the subjectID to check if the face recognition is
                    # correct
                    self.labels.append(subjectID)

        if self.algorithm == Algorithms.SIFT or self.algorithm == Algorithms.SURF:
            self.bf.train()
        else:
            # Train the face recognition algorithm
            self.faceRec.train(self.trainingImages, np.array(self.labels))

    def recognizeFace(self, path):
        """
        Function that tries to recognize each face (path passed by parameter).
        """

        if path.split(".")[1] in supported_files:
            # Get the subject id (should be a number)
            subjectID = path.split("_")[1]
            subjectID = int(subjectID.split(".")[0])

            # Load Image, Convert to Grayscale, Resize
            image = self.preprocessImage(path)

            if self.algorithm == Algorithms.SIFT or self.algorithm == Algorithms.SURF:
                # Detects and computes the keypoints and descriptors using the SURF
                # algorithm
                keypoints, descriptors = self.faceRec.detectAndCompute(image, None)

                matches = self.bf.match(descriptors)
                matches = sorted(matches, key=lambda x: x.distance)

                # Creates a results vector to store the number of similar points
                # for each image on the training set
                results = [0] * len(self.labels)

                # Based on the matches vector we create the results vector that
                # represents how many points this test image are similar to each
                # image in the training set
                for i in range(len(matches)):
                    results[matches[i].imgIdx] += 1

                # Index receives the position of the maximum value in the results
                # vector (it means that this is the most similar image)
                index = results.index(max(results))

                subject = self.labels[index]
            else:
                # Perform the face recognition
                subject, confidence = self.faceRec.predict(image)

                index = self.labels.index(subject)

        # Concatenate the two images (from the index position in the training
        # set and the current test image) to generate the output image
        tempImage = np.concatenate((self.trainingImages[index], image), axis=1)

        if subjectID >= 0:
            # Check if the subject is equal to the expected subject (subjectID).
            # It means the face image was correctly recognized
            if subject == subjectID:
                # Save the concatenated image to the vector
                self.recognizedFacesImages.append(tempImage)

                self.recognizedFaces += 1
            else:
                # Save the concatenated image to the vector
                self.unrecognizedFacesImages.append(tempImage)

                self.unrecognizedFaces += 1
        else:
            # Save the concatenated image to the vector
            self.nonFacesImages.append(tempImage)

            self.nonFaces += 1

    def predict(self, testPath):
        """
        Function that tries to recognize each face (path passed by parameter)
        """

        if testPath == "":
            print "Empty test path."
            sys.exit()

        self.content += "TestPath  : " + testPath + "\n"
        self.content += "SizeX : " + str(self.sizeX) + "\n"
        self.content += "SizeY : " + str(self.sizeY) + "\n"
        self.content += "Interpolation : " + self.interpolationTitle + "\n"

        # In the trainingPath folder search for all files in all directories
        for dirname, dirnames, filenames in os.walk(testPath):
            # For each file found
            for filename in filenames:
                # Ignore the text file
                if filename.split(".")[1] in supported_files:
                    # Creates the filePath joining the directory name with the
                    # file name
                    filePath = os.path.join(dirname, filename)

                    # Include the image file in the training set
                    self.recognizeFace(filePath)

        self.content += "RecognizedFaces   : " + \
            str(self.recognizedFaces) + "\n"
        self.content += "UnrecognizedFaces : " + \
            str(self.unrecognizedFaces) + "\n"
        self.content += "NonFaces          : " + str(self.nonFaces) + "\n"

    def showResults(self):
        """
        Function used to show the results in the screen.
        """

        print "==========================================================\n"
        print self.content
        print "=========================================================="

    def save(self, path=""):
        """
        Function used to automatically save the results in a defined folder.
        """

        # Make sure that none folder will have the same name
        time.sleep(1)

        # Format the parameters using underline
        tempParameters = self.parameters.replace(", ", "_")
        tempParameters = self.parameters.replace(",", "_")
        tempParameters = self.parameters.replace(" ", "_")

        # If the parameters were set include it in the folder name
        if tempParameters != "":
            folderName = time.strftime(
                "%Y_%m_%d_%H_%M_%S") + "_" + self.algorithmTitle + "_" + tempParameters
        else:
            folderName = time.strftime(
                "%Y_%m_%d_%H_%M_%S") + "_" + self.algorithmTitle

        if path != "":
            folderName = path + folderName

        os.makedirs(folderName)

        # Salva o arquivo de texto
        textFile = open(folderName + "/Report.txt", "w")
        textFile.write(self.content)
        textFile.close()

        count = 1
        os.makedirs(folderName + "/RecognizedFaces")
        for image in self.recognizedFacesImages:
            # Save the concatenated image to the output path
            cv2.imwrite(folderName + "/RecognizedFaces/" +
                        str(count) + ".png", image)
            count += 1

        count = 1
        os.makedirs(folderName + "/UnrecognizedFaces")
        for image in self.unrecognizedFacesImages:
            # Save the concatenated image to the output path
            cv2.imwrite(folderName + "/UnrecognizedFaces/" +
                        str(count) + ".png", image)
            count += 1

        count = 1
        os.makedirs(folderName + "/NonFaces")
        for image in self.nonFacesImages:
            # Save the concatenated image to the output path
            cv2.imwrite(folderName + "/NonFaces/" + str(count) + ".png", image)
            count += 1


# Import the libraries
import cv2
import os
import sys
import time
import numpy as np

from Algorithms import Algorithms
from Interpolation import Interpolation

# Declare all supported files
supported_files = ["png", "jpg", "jpeg"]

class FaceRecognition:
    """
    Class that encapsulates all 5 face recognition algorithms
    """

    def __init__(self):
    	"""
    	Set the default values
    	Size: 100x100
    	Algorithm: Eigenfaces
    	Interpolation: INTER_CUBIC (bicubic interpolation)
    	"""
        self.sizeX = 100
        self.sizeY = 100
        self.interpolation = Interpolation.INTER_CUBIC
        self.interpolationTitle = "INTER_CUBIC"

        self.nonFaces = 0

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
    	Set some values to the report. Reset the results.
    	"""

        self.parameters = ""

        # Vector used to store the training images
        self.trainingImages = []

        # Vector used to store the training images labels
        self.labels = []

        self.content = "Date/Time : " + \
            time.strftime("%d/%m/%Y %H:%M:%S") + "\n"

        self.faceRecMethods = []
        self.bf = []

        self.faceRecMethods.append(cv2.face.createEigenFaceRecognizer(self.eigenfacesParameter))
        self.faceRecMethods.append(cv2.face.createFisherFaceRecognizer(self.fisherfacesParameter))
        self.faceRecMethods.append(cv2.face.createLBPHFaceRecognizer(
            radius=self.radius, neighbors=self.neighbors, grid_x=self.gridX, grid_y=self.gridY))

        self.faceRecMethods.append(cv2.xfeatures2d.SIFT_create())
        self.bf.append(cv2.BFMatcher(cv2.NORM_L2, crossCheck=False))

        self.faceRecMethods.append(cv2.xfeatures2d.SURF_create())  # 400
        self.bf.append(cv2.BFMatcher(cv2.NORM_L2, crossCheck=False))

        self.algorithmTitle = "EIGENFACES-FISHERFACES-LBPH-SIFT-SURF"
        self.parameters = "Eigenfaces Parameters:" + str(self.eigenfacesParameter) + "\n"
        self.parameters += "Fisherfaces Parameters:" + str(self.fisherfacesParameter)

        self.content += "Algorithms : " + self.algorithmTitle + "\n"
        self.content += "Parameters : " + "\n" + self.parameters + "\n"

    def setEigenfacesParameter(self, parameter):
        """
        Set the eigenfaces parameters.
        :param parameter: The parameter for the eigenfaces method.
        """
        self.eigenfacesParameter = parameter

    def getEigenfacesParameter(self):
        """
        Get the eigenfaces parameters.
        :return: the selected parameter for the eigenfaces method.
        """
        return self.eigenfacesParameter

    def setFisherfacesParameter(self, parameter):
        """
        Set the fisherfaces parameters.
        :param parameter: The parameter for the fisherfaces method.
        """
        self.fisherfacesParameter = parameter

    def getFisherfacesParameter(self):
        """
        Get the fisherfaces parameters.
        :return: the selected parameter for the fisherfaces method.
        """
        return self.fisherfacesParameter

    def setLBPHParameters(self, radius, neighbors, gridX, gridY):
        """
        Set the fisherfaces parameters.
        :param radius: Radius
        :param neighbors: Neighbors.
        :param gridX: Grid X.
        :param gridY: Grid Y.
        """
        self.radius = radius
        self.neighbors = neighbors
        self.gridX = gridX
        self.gridY = gridY

    def getLBPHParameters(self):
        """
        Get the LBPH parameters.
        :return: the selected parameter for the LBPH method.
        """
        return self.radius, self.neighbors, self.gridX, self.gridY

    def getContent(self):
        """
        Get the report content.
        :return: the report content.
        """
        return self.content

    def getInterpolationTitle(self):
        """
        Get the selected interpolation method title.
        :return: the selected interpolation method title.
        """
        return self.interpolationTitle

    def getNonFaces(self):
        """
        Get the number of non faces.
        """
        return self.nonFaces

    def getInterpolation(self):
        """
        Get the selected interpolation method.
        :return: the selected interpolation method.
        """
        return self.interpolation

    def setInterpolation(self, interpolation):
        """
        Set the interpolation method based on the Interpolation class.
        :param interpolation: The selected interpolation method.
        """
        self.interpolation = interpolation

        if self.interpolation == Interpolation.INTER_CUBIC:
            self.interpolationTitle = "INTER_CUBIC"
        elif self.interpolation == Interpolation.INTER_AREA:
            self.interpolationTitle = "INTER_AREA"
        elif self.interpolation == Interpolation.INTER_LANCZOS4:
            self.interpolationTitle = "INTER_LANCZOS4"
        elif self.interpolation == Interpolation.INTER_LINEAR:
            self.interpolationTitle = "INTER_LINEAR"
        elif self.interpolation == Interpolation.INTER_NEAREST:
            self.interpolationTitle = "INTER_NEAREST"

    def setDefaultSize(self, sizeX, sizeY):
        """
        Set the default size for the imagens (default is 100x100).
        :param sizeX: The selected X size.
        :param sizeY: The selected Y size.
        """
        self.sizeX = sizeX
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
        elif self.interpolation == Interpolation.INTER_AREA:
            image = cv2.resize(image, (self.sizeX, self.sizeY),
                               interpolation=cv2.INTER_AREA)
        elif self.interpolation == Interpolation.INTER_LANCZOS4:
            image = cv2.resize(image, (self.sizeX, self.sizeY),
                               interpolation=cv2.INTER_LANCZOS4)
        elif self.interpolation == Interpolation.INTER_LINEAR:
            image = cv2.resize(image, (self.sizeX, self.sizeY),
                               interpolation=cv2.INTER_LINEAR)
        elif self.interpolation == Interpolation.INTER_NEAREST:
            image = cv2.resize(image, (self.sizeX, self.sizeY),
                               interpolation=cv2.INTER_NEAREST)
        else:
            print "Error trying to resize the image."
            sys.exit()

        return image

    def includeFace(self, path):
        """
        Function that receives the path of each face image as parameter and include it in the training set (bf object).
        """

        # Check if it is a valid image file
        if path.split(".")[1] in supported_files:

            # Get the subject id (should be a number) based on the format: subjectID_imageNumber.png
            pathSplit = path.split("/")
            fileName  = pathSplit[len(pathSplit)-1]
            subjectID = int(fileName.split("_")[0])

            # Load Image, Convert to Grayscale, Resize
            image = self.preprocessImage(path)

            # Detects and computes the keypoints and descriptors using the SIFT and SURF algorithms
            keypoints1, descriptors1 = self.faceRecMethods[3].detectAndCompute(image, None)
            keypoints2, descriptors2 = self.faceRecMethods[4].detectAndCompute(image, None)

            # Creates an numpy array
            clusters1 = np.array([descriptors1])
            clusters2 = np.array([descriptors2])

            # Add the array to the BFMatcher (for SIFT and SURF)
            self.bf[0].add(clusters1)
            self.bf[1].add(clusters2)

            return image, subjectID
        else:
            # If it is not a valid image file
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

                # If is is a valid image and it was correct included
                if image is not None:
                    # Store the image to generate an output image
                    self.trainingImages.append(image)

                    # Store the subjectID to check if the face recognition is correct
                    self.labels.append(subjectID)

        # Train the all face recognition algorithms
        self.faceRecMethods[0].train(self.trainingImages, np.array(self.labels)) # Eigenfaces
        self.faceRecMethods[1].train(self.trainingImages, np.array(self.labels)) # Fisherfaces
        self.faceRecMethods[2].train(self.trainingImages, np.array(self.labels)) # LBPH
        self.bf[0].train() # SIFT
        self.bf[1].train() # SURF

    def recognizeFace(self, path):
        """
        Function that tries to recognize each face (path passed by parameter).
        """

        # Check if it is an image file
        if path.split(".")[1] in supported_files:

            # Get the subject id (should be a number)
            # IMPORTANT: it follows the patter: imageNumber_subjectID.png
            # It is different from the pattern on the training set
            subjectID = path.split("_")[1]
            subjectID = int(subjectID.split(".")[0])

            # If it is not a facial image
            if subjectID < 0:
                self.nonFaces += 1

            # Load Image, Convert to Grayscale, Resize
            image = self.preprocessImage(path)

            # Stores the subject recognized for each algorithm
            subjects = []

            # 0 - Eigenfaces
            # 1 - Fisherfaces
            # 2 - LBPH
            for index in range(0, 3):
                # Perform the face recognition
                subject, confidence = self.faceRecMethods[index].predict(image)

                # Append the subject recognized by the algorithm
                subjects.append(subject)

            # 3 - faceRecMethods: SIFT
            # 4 - faceRecMethods: SURF
            # 0 - bf: SIFT
            # 1 - bf: SURF
            for index in range(0, 2):

                # Detects and computes the keypoints and descriptors using the SIFT and SURF algorithm
                keypoints, descriptors = self.faceRecMethods[index+3].detectAndCompute(image, None)

                matches = self.bf[index].match(descriptors)
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

                # Append the subject recognized by the algorithm
                subjects.append(self.labels[index])

            # Return the subjects recognized for each algorithm
            return subjects
        else:
            return None

    def predict(self, filePath):
        """
        Function that tries to recognize the face (path passed by parameter)
        """

        subjects = None

        if filePath == "":
            print "Empty file path."
        else:
            self.content += "FilePath  : " + filePath + "\n"
            self.content += "SizeX : " + str(self.sizeX) + "\n"
            self.content += "SizeY : " + str(self.sizeY) + "\n"
            self.content += "Interpolation : " + self.interpolationTitle + "\n"

            # Check if it is a valid image file
            if filePath.split(".")[1] in supported_files:
                # Include the image file in the training set
                subjects = self.recognizeFace(filePath)

        # Return the subjects predict for each algorithm:
        # subjects[0] = Eigenfaces
        # subjects[1] = Fisherfaces
        # subjects[2] = LBPH
        # subjects[3] = SIFT
        # subjects[4] = SURF
        return subjects

    def showResults(self):
        """
        Function used to show the results in the screen.
        """

        self.content += "NonFaces : " + str(self.nonFaces) + "\n"

        print "==========================================================\n"
        print self.content
        print "=========================================================="

    def save(self, path=""):
        """
        Function used to automatically save the results in a defined folder.
        """

        self.content += "NonFaces : " + str(self.nonFaces) + "\n"

        # Make sure that none folder will have the same name
        time.sleep(1)

        # Format the parameters using underline
        tempParameters = self.parameters.replace(", ", "_")
        tempParameters = self.parameters.replace(",", "_")
        tempParameters = self.parameters.replace(" ", "_")

        # If the parameters were set include it in the folder name
        fileName = time.strftime("%Y_%m_%d_%H_%M_%S") + ".txt"

        if path != "":
            fileName = path + "/" + fileName

        # Salva o arquivo de texto
        textFile = open(fileName, "w")
        textFile.write(self.content)
        textFile.close()

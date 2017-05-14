
# Import the libraries
import os
import sys
import time

sys.path.append('../FaceRecognition')
from FaceRecognition import FaceRecognition

class Report:
    """
    Class that provides an interface to generate reports
    """

    def __init__(self, faceRecognition):
        """
        Get the faceRecognition object
        """
    	self.faceRecognition = faceRecognition

    def generateReportSummary(self):
        """
        Generate a report summary with informations about the test.
        Return the content as a string.
        """
        content = time.strftime("%d/%m/%Y %H:%M:%S")
        content += "\nAlgorithm: " + self.faceRecognition.getAlgorithm().getAlgorithmName()
        recognized, unrecognized, nonFaces = self.faceRecognition.getResults()
        content += "\n\nTotal Images Analyzed: " + str(recognized + unrecognized + nonFaces)
        content += "\nRecognized Faces: "   + str(recognized)
        content += "\nUnrecognized Faces: " + str(unrecognized)
        content += "\nNon Faces: "          + str(nonFaces)
        if self.faceRecognition.getThreshold() >= 0:
            content += "\nThreshold Used: "     + str(self.faceRecognition.getThreshold())
        else:
            content += "\nThreshold Not Used."
        sizeX, sizeY = self.faceRecognition.getAuxiliary().getDefaultSize()
        content += "\n\nDefault Size Images: " + str(sizeX) + "x" + str(sizeY)
        content += "\nInterpolation Method: "  + self.faceRecognition.getAuxiliary().getInterpolationMethodName()
        content += "\nSupported Files: " + ', '.join(self.faceRecognition.getAuxiliary().getSupportedFiles())
        return content

    def generateFullReport(self):
        """
        Generate the full report.
        Return the content containing the information about each predicted image.
        """
        # Get the predicted results
        predictSubjectIds = self.faceRecognition.getPredictedSubjectIds()
        predictConfidence = self.faceRecognition.getPredictedConfidence()
        # Get the test information (labels and filenames)
        testLabels    = self.faceRecognition.getTestLabels()
        testFileNames = self.faceRecognition.getTestFileNames()

        content = ""

        # Create each line based on the predicted subject IDs
        for index in xrange(0, len(predictSubjectIds)):
            # Format: 1: Expected subject: 3: Classified as subject: 2: With confidence: 4123.123123: File name: 1_3
            content += str(index)
            content += ": Expected subject: " + str(testLabels[index])
            content += ": Classified as subject: " + str(predictSubjectIds[index])
            content += ": With confidence: " + str(predictConfidence[index])
            content += ": File name: " + testFileNames[index]
            content += "\n"

        return content

    def printResults(self):
        """
        Function used to show the results
        """
        print "========================= Results ========================="
        print self.generateReportSummary()
        print "==========================================================="

    def saveReport(self, path=""):
        """
        Function used to automatically save the report in a defined folder.
        Save only the text report not the images.
        """

        # Generate the report content
        content = self.generateReportSummary()
        content += "\n================================\n"
        content += self.generateFullReport()

        # Make sure that none folder will have the same name
        time.sleep(1)

        # If the parameters were set include it in the folder name
        fileName = time.strftime("%Y_%m_%d_%H_%M_%S") + ".txt"

        # If the path is not empty use it in the filename
        if path != "":
            if path.endswith("/"):
                fileName = path + fileName
            else:
                fileName = path + "/" + fileName

        # Save the text file
        self.faceRecognition.getAuxiliary().writeTextFile(content, fileName)

    def saveAllResults(self, path=""):
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
        self.saveReport(path)

        # Create 3 new folders
        recognizedFolder   = path + "Recognized/"
        unrecognizedFolder = path + "Unrecognized/"
        nonfacesFolder     = path + "NonFaces/"

        os.makedirs(recognizedFolder)
        os.makedirs(unrecognizedFolder)
        os.makedirs(nonfacesFolder)

        # The predicted results
        predictSubjectIds = self.faceRecognition.getPredictedSubjectIds()
        predictConfidence = self.faceRecognition.getPredictedConfidence()
        # The tests information
        testImages    = self.faceRecognition.getTestImages()
        testLabels    = self.faceRecognition.getTestLabels()
        testFileNames = self.faceRecognition.getTestFileNames()
        # The training information
        trainImages   = self.faceRecognition.getTrainImages()
        trainLabels   = self.faceRecognition.getTrainLabels()

        delimiter = "_"

        for index in xrange(0, len(predictSubjectIds)):
            # Patter: 1_Expected_2_Classified_2_Confidence_40192.12938291.png
            label = str(index) + delimiter + "Expected" + delimiter + str(testLabels[index]) + delimiter
            label += "Classified" + delimiter + str(predictSubjectIds[index]) + delimiter
            label += "Confidence" + delimiter + str(predictConfidence[index]) + ".png"

            # Find the image that matches based on the trainLabel and predictedSubjectIDs
            image1 = testImages[index]
            image2 = None
            for i in xrange(0, len(trainLabels)):
                if str(trainLabels[i]) == str(predictSubjectIds[index]):
                    image2 = trainImages[i]

            # Concatenate the images
            image = self.faceRecognition.getAuxiliary().concatenateImages(image1, image2)

            # Get the correct fileName
            fileName = ""
            if str(testLabels[index]) == "-1":
                fileName = nonfacesFolder
            elif str(testLabels[index]) == str(predictSubjectIds[index]):
                fileName = recognizedFolder
            else:
                fileName = unrecognizedFolder

            fileName += label

            # Save the concatenated image in the correct folder
            self.faceRecognition.getAuxiliary().saveImage(fileName, image)

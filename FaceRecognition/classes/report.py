
# Import the libraries
import os
import sys
import time

from voting import Voting
from face_recognition import FaceRecognition
from ensemble import Ensemble
from auxiliary import Auxiliary


class Report:
    """
    Class that provides an interface to generate reports
    """

    def __init__(self, object):
        """
        Get the object (FaceRecognition or Ensemble)
        """
        self.object = object
        self.auxiliary = Auxiliary()

    def generateReportSummary(self):
        """
        Generate a report summary with informations about the test.
        Return the content as a string.
        """
        if isinstance(self.object, FaceRecognition):
            content = "Face Recognition (single algorithm)"
        elif isinstance(self.object, Ensemble):
            content = "Ensemble (multiple algorithms)"

        content += "\n\nDate/Time: " + time.strftime("%d/%m/%Y %H:%M:%S")
        content += "\nTrain Path: " + self.object.trainPath
        content += "\nTest Path: " + self.object.testPath + "\n"

        # For the face recognition class get only the name of the algorithm
        if isinstance(self.object, FaceRecognition):
            content += "\nAlgorithm: " + self.object.algorithm.getAlgorithmName()
            if self.object.threshold >= 0:
                content += "\nThreshold Used: " + \
                    str(self.object.threshold)
            else:
                content += "\nThreshold Not Used."

        # For the Ensemble class get the name of all algorithms
        elif isinstance(self.object, Ensemble):
            content += "\nVoting Scheme: " + self.object.voting.getVotingSchemeName()
            weights = self.object.voting.weights

            for index in xrange(0, len(self.object.fralgorithms)):
                content += "\nAlgorithm: " + \
                    self.object.fralgorithms[index].getAlgorithmName()
                # If it is using the WEIGHTED voting scheme
                if self.object.voting.getVotingScheme() == Voting.WEIGHTED:
                    # If the index is valid for the weights list
                    if index < len(weights):
                        content += " - Weight: " + str(weights[index])

        content += "\n\nTotal Images Analyzed: " + \
            str(len(self.object.testFileNames))

        totalFaceImages = 0.0
        accuracy2 = 0.0

        if isinstance(self.object, FaceRecognition):
            if self.object.threshold >= 0:
                totalFaceImages = self.object.recognizedBelowThreshold + self.object.unrecognizedBelowThreshold
                # Calculate the accuracy using only the results below the
                # threshold
                accuracy2 = self.auxiliary.calcAccuracy(
                    self.object.recognizedBelowThreshold, totalFaceImages)

                totalFaceImages += self.object.recognizedAboveThreshold + self.object.unrecognizedAboveThreshold
                # Calculate the accuracy using the total number of face images
                accuracy = self.auxiliary.calcAccuracy(
                    self.object.recognizedBelowThreshold, totalFaceImages)

                content += "\nRecognized Faces Below Threshold: " + \
                    str(self.object.recognizedBelowThreshold)
                content += "\nUnrecognized Faces Below Threshold: " + \
                    str(self.object.unrecognizedBelowThreshold)
                content += "\nNon Faces Below Threshold: " + \
                    str(self.object.nonFacesBelowThreshold)
                content += "\nRecognized Faces Above Threshold: " + \
                    str(self.object.recognizedAboveThreshold)
                content += "\nUnrecognized Faces Above Threshold: " + \
                    str(self.object.unrecognizedAboveThreshold)
                content += "\nNon Faces Above Threshold: " + \
                    str(self.object.nonFacesAboveThreshold)
            else:
                totalFaceImages = float(
                    self.object.recognized + self.object.unrecognized)
                accuracy = self.auxiliary.calcAccuracy(
                    self.object.recognized, totalFaceImages)
                content += "\nRecognized Faces: " + \
                    str(self.object.recognized)
                content += "\nUnrecognized Faces: " + \
                    str(self.object.unrecognized)
                content += "\nNon Faces: " + str(self.object.nonFaces)
        else:
            totalFaceImages = float(
                self.object.recognized + self.object.unrecognized)
            accuracy = self.auxiliary.calcAccuracy(
                self.object.recognized, totalFaceImages)
            content += "\nRecognized Faces: " + \
                str(self.object.recognized)
            content += "\nUnrecognized Faces: " + \
                str(self.object.unrecognized)
            content += "\nNon Faces: " + str(self.object.nonFaces)

        content += "\nRecognition Rate - Recognized / Total Face Images"
        content += "\nAccuracy: " + str(accuracy) + " %"

        if isinstance(self.object, FaceRecognition):
            if self.object.threshold >= 0:
                content += "\nAccuracy Only Below Threshold: " + \
                    str(accuracy2) + " %"

        sizeX, sizeY = self.object.auxiliary.getDefaultSize()
        content += "\n\nDefault Size Images: " + str(sizeX) + "x" + str(sizeY)
        content += "\nInterpolation Method: " + \
            self.object.auxiliary.getInterpolationMethodName()
        content += "\nSupported Files: " + \
            ', '.join(self.object.auxiliary.supportedFiles)
        return content

    def generateFullReport(self):
        """
        Generate the full report.
        Return the content containing the information about each predicted image.
        """
        # Get the predicted results
        predictSubjectIds = self.object.predictedSubjectIds
        predictConfidence = self.object.predictedConfidence
        # Get the test information (labels and filenames)
        testLabels = self.object.testLabels
        testFileNames = self.object.testFileNames

        content = ""

        # Create each line based on the predicted subject IDs
        for index in xrange(0, len(predictSubjectIds)):
            # Format: 1: Expected subject: 3: Classified as subject: 2: With
            # confidence: 4123.123123: File name: 1_3
            content += str(index + 1)
            content += ": Expected subject: " + str(testLabels[index])
            content += ": Classified as subject: " + \
                str(predictSubjectIds[index])

            if isinstance(self.object, FaceRecognition):
                content += ": With confidence: " + \
                    str(predictConfidence[index])
            elif isinstance(self.object, Ensemble):
                content += ": Predicted Subjects: " + \
                    ', '.join(map(str, predictConfidence[index]))

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
        content += "\n===========================================================\n"
        content += self.generateFullReport()

        # Make sure that none folder will have the same name
        time.sleep(1)

        # If the parameters were set include it in the folder name
        fileName = time.strftime("%Y_%m_%d_%H_%M_%S") + ".txt"

        # If the path is not empty use it in the filename
        if path != "":
            # If the path does not exist, create it
            if not os.path.exists(path):
                os.makedirs(path)

            if path.endswith(".txt"):
                fileName = path
            elif path.endswith("/"):
                fileName = path + fileName
            else:
                fileName = path + "/" + fileName

        # Save the text file
        self.auxiliary.writeTextFile(content, fileName)

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
        recognizedFolder = path + "Recognized/"
        unrecognizedFolder = path + "Unrecognized/"
        nonfacesFolder = path + "NonFaces/"

        os.makedirs(recognizedFolder)
        os.makedirs(unrecognizedFolder)
        os.makedirs(nonfacesFolder)

        # The predicted results
        predictSubjectIds = self.object.predictedSubjectIds
        predictConfidence = self.object.predictedConfidence
        # The tests information
        testImages = self.object.testImages
        testLabels = self.object.testLabels
        testFileNames = self.object.testFileNames
        # The training information
        trainImages = self.object.trainImages
        trainLabels = self.object.trainLabels

        delimiter = "_"

        for index in xrange(0, len(predictSubjectIds)):
            # Patter: 1_Expected_2_Classified_2_Confidence_40192.12938291.png
            label = str(index) + delimiter + "Expected" + \
                delimiter + str(testLabels[index]) + delimiter
            label += "Classified" + delimiter + \
                str(predictSubjectIds[index]) + delimiter

            if isinstance(self.object, FaceRecognition):
                label += "Confidence" + delimiter + \
                    str(predictConfidence[index])
            elif isinstance(self.object, Ensemble):
                label += "Voting" + delimiter + self.object.voting.getVotingSchemeName()

            label += ".png"

            # Find the image that matches based on the trainLabel and
            # predictedSubjectIDs
            image1 = testImages[index]
            image2 = None
            for i in xrange(0, len(trainLabels)):
                if str(trainLabels[i]) == str(predictSubjectIds[index]):
                    image2 = trainImages[i]

            # Concatenate the images
            image = self.auxiliary.concatenateImages(image1, image2)

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
            self.auxiliary.saveImage(fileName, image)

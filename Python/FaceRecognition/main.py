
import os
import cv2
import sys

from classes.auxiliary import Auxiliary
from classes.face_recognition import FaceRecognition
from classes.voting import Voting
from classes.machinery_committee import MachineryCommittee
from classes.report import Report

from algorithms.eigenfaces import Eigenfaces
from algorithms.fisherfaces import Fisherfaces
from algorithms.lbph import LBPH
from algorithms.sift import SIFT
from algorithms.surf import SURF

"""
# Set the paths to train, test and store the results
realPath = os.path.realpath(__file__)
dirPath  = os.path.dirname(realPath)

# Define the path to the training files/folder
trainPath = dirPath + "/dataset/train/1/"

# Define the path to the test folder
testPath = dirPath + "/dataset/test/1/"

# Define the path to the results folder
resultsPath = dirPath + "/results/"
"""

algs = [Eigenfaces(), Fisherfaces(), LBPH(), SIFT(), SURF()]
algNames = ["Eigenfaces", "Fisherfaces", "LBPH", "SIFT", "SURF"]

trainPath = "/home/kelvin/Desktop/FaceRecognition/Train/"
testPath  = "/home/kelvin/Desktop/FaceRecognition/Test/"
resultsPath = "/home/kelvin/Desktop/FaceRecognition/Results/"

def faceFecognition():

    global trainPath, testPath, resultsPath

    # Algorithms loop
    for algsIndex in xrange(0, len(algs)):
        # Train folder loop
        for trainIndex in xrange(1, 7):
            # Test folder loop
            for testIndex in xrange(1, 13):

                # Create the auxiliary object
                auxiliary = Auxiliary(sizeX=100, sizeY=100, interpolation=cv2.INTER_CUBIC)

                # Create the algorithm object
                algorithm = algs[algsIndex]
                #algorithm = Eigenfaces()
                #algorithm = Fisherfaces()
                #algorithm = LBPH()
                #algorithm = SIFT()
                #algorithm = SURF()

                # Create the face recognition object
                faceRecog = FaceRecognition(algorithm, auxiliary)

                # Train the algorithm
                faceRecog.train(trainPath + str(trainIndex) + "/")

                # Try to recognize the faces
                faceRecog.recognizeFaces(testPath + str(testIndex) + "/")

                # Create the report object
                report = Report(faceRecog)

                # Print the results
                report.printResults()

                # Save the report (text file)
                report.saveReport(resultsPath + algNames[algsIndex] + "/")

                # Save all results (summary, full report and images)
                #report.saveAllResults(resultsPath)

def machineryCommittee():

    global trainPath, testPath, resultsPath

    # Train folder loop
    for trainIndex in xrange(1, 7):
        # Test folder loop
        for testIndex in xrange(1, 13):

            # Create the auxiliary object
            auxiliary = Auxiliary(sizeX=100, sizeY=100, interpolation=cv2.INTER_CUBIC)

            # Create the algorithm object
            algorithms = algs
            """
            algorithms = []
            algorithms.append( Eigenfaces() )
            algorithms.append( Fisherfaces() )
            algorithms.append( LBPH() )
            algorithms.append( SIFT() )
            algorithms.append( SURF() )
            """

            # Set the weights based on the algorithms list order
            #weights = [10, 10, 10, 10, 10]

            # Create the voting object setting the WEIGHTED as the voting scheme
            #voting = Voting(Voting.WEIGHTED, weights)
            voting = Voting()

            # Create the face recognition object
            machineryCommittee = MachineryCommittee(algorithms, auxiliary, voting)

            # Train the algorithm
            machineryCommittee.train(trainPath + str(trainIndex) + "/")

            # Try to recognize the faces
            machineryCommittee.recognizeFaces(testPath + str(testIndex) + "/")

            # Create the report object
            report = Report(machineryCommittee)

            # Print the results
            report.printResults()

            # Save the report (text file)
            report.saveReport(resultsPath)

            # Save all results (summary, full report and images)
            #report.saveAllResults(resultsPath)

if __name__ == "__main__":
    faceFecognition()
    #machineryCommittee()

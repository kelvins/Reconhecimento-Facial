
import os
import cv2
import sys
import gc

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

algNames = ["Eigenfaces", "Fisherfaces", "LBPH", "SIFT", "SURF"]

trainPath = "/home/kelvin/Desktop/FaceRecognition/Tests/TREINAMENTO/BASE"
testPath  = "/home/kelvin/Desktop/FaceRecognition/Tests/TESTE/TESTEVIDEO"
resultsPath = "/home/kelvin/Desktop/FaceRecognition/Tests/RESULTS/"

def faceFecognition():

    global trainPath, testPath, resultsPath

    # Algorithms loop
    for algsIndex in xrange(4, 5):
        # Train folder loop
        for trainIndex in xrange(1, 7):
            # Test folder loop
            for testIndex in xrange(1, 13):

                # Create the auxiliary object
                auxiliary = Auxiliary(sizeX=100, sizeY=100, interpolation=cv2.INTER_CUBIC)

                print algNames[algsIndex] + ": Train: " + str(trainIndex) + ": Test: " + str(testIndex)

                # Create the algorithm object
                if algsIndex == 0:
                    algorithm = Eigenfaces()
                elif algsIndex == 1:
                    algorithm = Fisherfaces()
                elif algsIndex == 2:
                    algorithm = LBPH()
                elif algsIndex == 3:
                    algorithm = SIFT()
                elif algsIndex == 4:
                    algorithm = SURF()
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
                #report.printResults()

                # Save the report (text file)
                report.saveReport(resultsPath + algNames[algsIndex] + "/" + str(trainIndex) + "/")

                # Save all results (summary, full report and images)
                #report.saveAllResults(resultsPath)

                del auxiliary
                del algorithm
                del faceRecog
                del report
                
                gc.collect()

def machineryCommittee():

    global trainPath, testPath, resultsPath

    # Train folder loop
    for trainIndex in xrange(1, 7):
        # Test folder loop
        for testIndex in xrange(1, 13):

            # Create the auxiliary object
            auxiliary = Auxiliary(sizeX=100, sizeY=100, interpolation=cv2.INTER_CUBIC)

            print "Train: " + str(trainIndex) + ": Test: " + str(testIndex)

            # Create the algorithm object
            algorithms = []
            algorithms.append( Eigenfaces() )
            algorithms.append( Fisherfaces() )
            algorithms.append( LBPH() )
            algorithms.append( SIFT() )
            algorithms.append( SURF() )

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
            #report.printResults()

            # Save the report (text file)
            report.saveReport(resultsPath + "COMITE/" + str(trainIndex) + "/")

            # Save all results (summary, full report and images)
            #report.saveAllResults(resultsPath)

            del auxiliary
            del algorithms
            del voting
            del machineryCommittee
            del report
            
            gc.collect()


if __name__ == "__main__":
    #faceFecognition()
    machineryCommittee()

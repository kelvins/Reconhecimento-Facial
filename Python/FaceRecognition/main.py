
import os
import cv2
import sys
import gc

from classes.auxiliary import Auxiliary
from classes.face_recognition import FaceRecognition
from classes.voting import Voting
from classes.ensemble import Ensemble
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
"""
initialThreshold = [1200,  400,  20,  2,  2]
finalThreshold   = [1800, 1000, 140, 80, 80]
stepThreshold    = [   5,    5,   2,  2,  2]
"""
trainPath = "/home/kelvins/Desktop/Reconhecimento-Facial/Python/Dataset/Train/BASE"
testPath  = "/home/kelvins/Desktop/Reconhecimento-Facial/Python/Dataset/Test/VIDEO"
resultsPath = "/home/kelvins/Desktop/Reconhecimento-Facial/Python/Dataset/Results/20170522_Ensembles/Majority"

#LBPH 20 - 140 - 1
#EIGENFACES 1200 - 1800 - 5
#FISHERFACES 400 - 1000 - 5

def faceFecognition():

    global trainPath, testPath, resultsPath

    # Algorithms loop
    for algsIndex in xrange(0, 5):
        # Train folder loop
        for trainIndex in xrange(1, 7):
            # Test folder loop
            for testIndex in xrange(1, 13):
                # For each threshold
                for threshold in xrange(initialThreshold[algsIndex], finalThreshold[algsIndex]+1, stepThreshold[algsIndex]):

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
                    faceRecog = FaceRecognition(algorithm, auxiliary, threshold)

                    # Train the algorithm
                    faceRecog.train(trainPath + str(trainIndex) + "/")

                    # Try to recognize the faces
                    faceRecog.recognizeFaces(testPath + str(testIndex) + "/")

                    # Create the report object
                    report = Report(faceRecog)

                    # Print the results
                    #report.printResults()

                    # Save the report (text file)
                    report.saveReport(resultsPath + algNames[algsIndex] + "/" + str(trainIndex) + "/" + str(testIndex) + "/")

                    # Save all results (summary, full report and images)
                    #report.saveAllResults(resultsPath)

                    del auxiliary
                    del algorithm
                    del faceRecog
                    del report

                    gc.collect()

def ensemble():

    global trainPath, testPath, resultsPath

    weights = []
    weights.append([60, 10, 10, 10, 10])
    weights.append([10, 60, 10, 10, 10])
    weights.append([10, 10, 60, 10, 10])
    weights.append([10, 10, 10, 60, 10])
    weights.append([10, 10, 10, 10, 60])

    for weight in weights:
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
                voting = Voting(Voting.WEIGHTED, weight)
                #voting = Voting()

                # Create the ensemble object
                ensemble = Ensemble(algorithms, auxiliary, voting)

                # Train the algorithm
                ensemble.train(trainPath + str(trainIndex) + "/")

                # Try to recognize the faces
                ensemble.recognizeFaces(testPath + str(testIndex) + "/")

                # Create the report object
                report = Report(ensemble)

                # Print the results
                #report.printResults()

                # Save the report (text file)
                report.saveReport(resultsPath + "/" + str(trainIndex) + "/")

                # Save all results (summary, full report and images)
                #report.saveAllResults(resultsPath)

                del auxiliary
                del algorithms
                del voting
                del ensemble
                del report

                gc.collect()


if __name__ == "__main__":
    #faceFecognition()
    ensemble()

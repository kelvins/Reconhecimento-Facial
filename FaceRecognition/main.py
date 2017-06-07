import cv2
import sys

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

dirPath = "/home/kelvins/Desktop/Reconhecimento-Facial/Dataset"
trainPath = dirPath + "/Train/BASE1/"
testPath = dirPath + "/Test/VIDEO1/"
resultsPath = dirPath + "/Results/"


def faceFecognition():
    global trainPath, testPath, resultsPath

    # Create the auxiliary object
    auxiliary = Auxiliary(sizeX=100, sizeY=100, interpolation=cv2.INTER_CUBIC)

    # Create the algorithm object
    algorithm = Eigenfaces()
    #algorithm = Fisherfaces()
    #algorithm = LBPH()
    #algorithm = SIFT()
    #algorithm = SURF()

    # Create the face recognition object
    faceRecog = FaceRecognition(algorithm, auxiliary)

    # Train the algorithm
    faceRecog.train(trainPath)

    # Try to recognize the faces
    faceRecog.recognizeFaces(testPath)

    # Create the report object
    report = Report(faceRecog)

    # Print the results
    report.printResults()

    # Save the report (text file)
    report.saveReport(resultsPath)

    # Save all results (summary, full report and images)
    # report.saveAllResults(resultsPath)


def ensemble():
    global trainPath, testPath, resultsPath

    # Create the auxiliary object
    auxiliary = Auxiliary(sizeX=100, sizeY=100, interpolation=cv2.INTER_CUBIC)

    # Create and add all algorithms for the ensemble
    algorithms = []
    algorithms.append(Eigenfaces())
    algorithms.append(LBPH())
    algorithms.append(Fisherfaces())
    algorithms.append(SIFT())
    algorithms.append(SURF())

    # Create the voting object (Majority or Weighted)
    #voting = Voting()
    voting = Voting(Voting.WEIGHTED, [10, 20, 30, 30, 40])

    # Create the ensemble object
    ensemble = Ensemble(algorithms, auxiliary, voting)

    # Train the algorithm
    ensemble.train(trainPath)

    # Try to recognize the faces
    ensemble.recognizeFaces(testPath)

    # Create the report object
    report = Report(ensemble)

    # Print the results
    report.printResults()

    # Save the report (text file)
    report.saveReport(resultsPath)

    # Save all results (summary, full report and images)
    # report.saveAllResults(resultsPath)


if __name__ == "__main__":
    faceFecognition()
    ensemble()


import os
import cv2
import sys

#sys.path.append('Auxiliary')
#from Auxiliary import Auxiliary
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

def faceFecognition():

    realPath = os.path.realpath(__file__)
    dirPath  = os.path.dirname(realPath)

    # Define the path to the training files/folder
    trainPath = dirPath + "/dataset/train/"

    # Define the path to the test folder
    testPath = dirPath + "/dataset/test/"

    # Define the path to the results folder
    resultsPath = dirPath + "/results/"

    # Create the auxiliary object
    auxiliary = Auxiliary(sizeX=100, sizeY=100, interpolation=cv2.INTER_CUBIC)

    # Create the algorithm object
    algorithm = Eigenfaces() # numComponents=0, threshold=-1

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
    #report.saveReport(resultsPath)

    # Save all results (summary, full report and images)
    #report.saveAllResults(resultsPath)

def machineryCommittee():

    realPath = os.path.realpath(__file__)
    dirPath  = os.path.dirname(realPath)

    # Define the path to the training files/folder
    trainPath = dirPath + "/Dataset/Train/"

    # Define the path to the test folder
    testPath = dirPath + "/Dataset/Test/"

    # Define the path to the results folder
    resultsPath = dirPath + "/Results/"

    # Create the auxiliary object
    auxiliary = Auxiliary(sizeX=100, sizeY=100, interpolation=cv2.INTER_CUBIC)

    # Create the algorithm object
    algorithms = []
    algorithms.append( Eigenfaces() )
    algorithms.append( Fisherfaces() )
    algorithms.append( LBPH() )
    algorithms.append( SIFT() )
    algorithms.append( SURF() )

    # Set the weights based on the algorithms list order
    weights = [20, 20, 20, 20, 20]

    # Create the voting object setting the WEIGHTED as the voting scheme
    voting = Voting(Voting.WEIGHTED, weights)

    # Create the face recognition object
    machineryCommittee = MachineryCommittee(algorithms, auxiliary, voting)

    # Train the algorithm
    machineryCommittee.train(trainPath)

    # Try to recognize the faces
    machineryCommittee.recognizeFaces(testPath)

    # Create the report object
    report = Report(machineryCommittee)

    # Print the results
    report.printResults()

    # Save the report (text file)
    #report.saveReport(resultsPath)

    # Save all results (summary, full report and images)
    #report.saveAllResults(resultsPath)

if __name__ == "__main__":
    faceFecognition()
    machineryCommittee()

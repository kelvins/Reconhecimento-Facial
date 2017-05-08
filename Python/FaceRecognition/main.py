
import os
import cv2
import sys

sys.path.append('Auxiliary')
from Auxiliary import Auxiliary

sys.path.append('FaceRecognition')
from FaceRecognition import FaceRecognition

sys.path.append('Algorithms')
from Eigenfaces import Eigenfaces
from Fisherfaces import Fisherfaces
from LBPH import LBPH
from SIFT import SIFT
from SURF import SURF

def main():
    # Define the path to the training files/folder
    trainPath = "/Users/kelvinsp/Desktop/Treinamento6/"

    # Define the path to the test folder
    testPath = "/Users/kelvinsp/Desktop/Teste6/"

    auxiliary = Auxiliary(sizeX=100, sizeY=100, interpolation=cv2.INTER_CUBIC)
    algorithm = Eigenfaces() # numComponents=0, threshold=-1

    faceRecog = FaceRecognition(algorithm, auxiliary)
    faceRecog.train(trainPath)
    faceRecog.recognizeFaces(testPath)
    nonFaces, recognized, unrecognized = faceRecog.getResults()

    print "nonFaces" + str(nonFaces)
    print "recognized" + str(recognized)
    print "unrecognized" + str(unrecognized)

if __name__ == "__main__":
    main()

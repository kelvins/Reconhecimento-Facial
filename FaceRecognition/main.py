import cv2

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

dirPath = "/home/ubuntu/Desktop/Reconhecimento-Facial/Dataset"
train_path = dirPath + "/Train/BASE1/"
test_path = dirPath + "/Test/VIDEO1/"
results_path = dirPath + "/Results/"


def face_recognition():
    global train_path, test_path, results_path

    # Create the auxiliary object
    auxiliary = Auxiliary(
        size_x=100,
        size_y=100,
        interpolation=cv2.INTER_CUBIC)

    # Create the algorithm object
    algorithm = Eigenfaces()
    # algorithm = Fisherfaces()
    # algorithm = LBPH()
    # algorithm = SIFT()
    # algorithm = SURF()

    # Create the face recognition object
    face_recog = FaceRecognition(algorithm, auxiliary)

    # Train the algorithm
    face_recog.train(train_path)

    # Try to recognize the faces
    face_recog.recognize_faces(test_path)

    # Create the report object
    report = Report(face_recog)

    # Print the results
    report.print_results()

    # Save the report (text file)
    #report.save_report(results_path)

    # Save all results (summary, full report and images)
    # report.save_all_results(results_path)


def ensemble():
    global train_path, test_path, results_path

    # Create the auxiliary object
    auxiliary = Auxiliary(
        size_x=100,
        size_y=100,
        interpolation=cv2.INTER_CUBIC)

    # Create and add all algorithms for the ensemble
    algorithms = list()
    algorithms.append(Eigenfaces())
    algorithms.append(LBPH())
    algorithms.append(Fisherfaces())
    algorithms.append(SIFT())
    algorithms.append(SURF())

    # Create the voting object (Majority or Weighted)
    voting = Voting(Voting().weighted, [10, 10, 10, 10, 10])

    # Create the ensemble object
    ensemble = Ensemble(algorithms, auxiliary, voting)

    # Train the algorithm
    ensemble.train(train_path)

    # Try to recognize the faces
    ensemble.recognize_faces(test_path)

    # Create the report object
    report = Report(ensemble)

    # Print the results
    report.print_results()

    # Save the report (text file)
    #report.save_report(results_path)

    # Save all results (summary, full report and images)
    # report.save_all_results(results_path)


if __name__ == "__main__":
    face_recognition()
    ensemble()

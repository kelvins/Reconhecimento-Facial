import sys
import unittest

sys.path.append('../classes')
sys.path.append('../algorithms')
from report import Report
from auxiliary import Auxiliary
from face_recognition import FaceRecognition
from eigenfaces import Eigenfaces

auxiliary = Auxiliary()
algorithm = Eigenfaces()
faceRecog = FaceRecognition(algorithm, auxiliary)
report = Report(faceRecog)


class ReportSummaryTest(unittest.TestCase):
    def test1(self):
        wrong_report = Report(auxiliary)
        self.assertEqual(wrong_report.generate_report_summary(), "")

    def test2(self):
        self.assertEqual(report.generate_report_summary(), "")


if __name__ == '__main__':
    unittest.main()

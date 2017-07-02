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
        self.assertEqual(report.generateReportSummary(), "")


if __name__ == '__main__':
    unittest.main()

import sys
import unittest

sys.path.append('../Auxiliary')
from Auxiliary import Auxiliary

auxiliary = Auxiliary()

class GetInterpolationMethodNameTest(unittest.TestCase):
    def test1(self):
        self.assertEqual(auxiliary.getInterpolationMethodName(), "cv2.INTER_CUBIC")

if __name__ == '__main__':
    unittest.main()

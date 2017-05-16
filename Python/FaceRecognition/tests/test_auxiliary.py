import cv2
import sys
import unittest

sys.path.append('../classes')
from auxiliary import Auxiliary

auxiliary = Auxiliary()

class GetInterpolationMethodNameTest(unittest.TestCase):
    def test1(self):
        self.assertEqual(auxiliary.getInterpolationMethodName(), "cv2.INTER_CUBIC")

    def test2(self):
    	auxiliary.setInterpolation(cv2.INTER_LANCZOS4)
        self.assertEqual(auxiliary.getInterpolationMethodName(), "cv2.INTER_LANCZOS4")

    def test2(self):
    	auxiliary.setInterpolation(123)
        self.assertEqual(auxiliary.getInterpolationMethodName(), "")

class GetSupportedFilesTest(unittest.TestCase):
    def test1(self):
        self.assertEqual(auxiliary.getSupportedFiles(), ["png", "jpg", "jpeg"])

    def test2(self):
        auxiliary.setSupportedFiles(["png", "jpg", "jpeg", "gif"])
        self.assertEqual(auxiliary.getSupportedFiles(), ["png", "jpg", "jpeg", "gif"])

class GetDefaultSizeTest(unittest.TestCase):
    def test1(self):
        self.assertEqual(auxiliary.getDefaultSize(), (100, 100))

    def test2(self):
        auxiliary.setDefaultSize(200, 200)
        self.assertEqual(auxiliary.getDefaultSize(), (200, 200))

    def test3(self):
        auxiliary.setDefaultSize(-5, -5)
        self.assertEqual(auxiliary.getDefaultSize(), (200, 200))

    def test4(self):
        auxiliary.setDefaultSize(-5, 100)
        self.assertEqual(auxiliary.getDefaultSize(), (200, 100))

    def test5(self):
        auxiliary.setDefaultSize(50, -5)
        self.assertEqual(auxiliary.getDefaultSize(), (50, 100))

class GrayscaleTest(unittest.TestCase):
    def test1(self):
        img = auxiliary.loadImage("images/python.png")
        self.assertEqual(auxiliary.isGrayscale(img), False)

    def test2(self):
        img = auxiliary.loadImage("images/python.png")
        img = auxiliary.toGrayscale(img)
        auxiliary.saveImage("images/python_gray.png", img)
        self.assertEqual(auxiliary.isGrayscale(img), True)

    def test3(self):
        img = auxiliary.loadImage("images/python_gray.png")
        self.assertEqual(auxiliary.isGrayscale(img), True)

class PreprocessImageTest(unittest.TestCase):
    def test1(self):
        img = auxiliary.loadImage("images/python.png")
        img = auxiliary.resizeImage(img, 400, 400, cv2.INTER_CUBIC)
        self.assertEqual(img.shape[:2], (400, 400))

    def test2(self):
    	auxiliary.setDefaultSize(100, 100)
    	auxiliary.setInterpolation(cv2.INTER_CUBIC)
        img = auxiliary.preprocessImage("images/python.png")
        self.assertEqual(img.shape[:2], (100, 100))
        self.assertEqual(auxiliary.isGrayscale(img), True)

class ConcatenateImagesTest(unittest.TestCase):
    def test1(self):
        img1 = auxiliary.loadImage("images/python.png")
        img1 = auxiliary.resizeImage(img1, 100, 100, cv2.INTER_CUBIC)

        img2 = auxiliary.loadImage("images/python_gray.png")
        img2 = auxiliary.resizeImage(img2, 100, 100, cv2.INTER_CUBIC)

        img = auxiliary.concatenateImages(img1, img2)
        auxiliary.saveImage("images/python_concatenated.png", img)

        img = auxiliary.loadImage("images/python_concatenated.png")
        self.assertEqual(img.shape[:2], (100, 200))

class ExtractPathsTest(unittest.TestCase):
    def test1(self):
        paths = auxiliary.extractImagesPaths("images/")
        self.assertEqual(len(paths), 3)

class WriteTextFileTest(unittest.TestCase):
    def test1(self):
        content  = "WriteTextFileTest"
        fileName = "images/WriteTextFileTest.txt"
        auxiliary.writeTextFile(content, fileName)

        paths = auxiliary.extractFilesPaths("images/")
        self.assertEqual(len(paths), 4)

if __name__ == '__main__':
    unittest.main()

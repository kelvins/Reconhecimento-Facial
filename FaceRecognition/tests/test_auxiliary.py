import cv2
import sys
import unittest

sys.path.append('../classes')
from auxiliary import Auxiliary

auxiliary = Auxiliary()

class GetInterpolationMethodNameTest(unittest.TestCase):
    def test1(self):
        self.assertEqual(
            auxiliary.get_interpolation_method_name(),
            "cv2.INTER_CUBIC")

    def test2(self):
        auxiliary.interpolation = cv2.INTER_LANCZOS4
        self.assertEqual(
            auxiliary.get_interpolation_method_name(),
            "cv2.INTER_LANCZOS4")

    def test2(self):
        auxiliary.interpolation = 123
        self.assertEqual(auxiliary.get_interpolation_method_name(), "")


class GetSupportedFilesTest(unittest.TestCase):
    def test1(self):
        self.assertEqual(auxiliary.supported_files, ["png", "jpg", "jpeg"])

    def test2(self):
        auxiliary.supported_files = ["png", "jpg", "jpeg", "gif"]
        self.assertEqual(
            auxiliary.supported_files, ["png", "jpg", "jpeg", "gif"])


class GetDefaultSizeTest(unittest.TestCase):
    def test1(self):
        self.assertEqual(auxiliary.get_default_size(), (100, 100))

    def test2(self):
        auxiliary.set_default_size(200, 200)
        self.assertEqual(auxiliary.get_default_size(), (200, 200))

    def test3(self):
        auxiliary.set_default_size(-5, -5)
        self.assertEqual(auxiliary.get_default_size(), (200, 200))

    def test4(self):
        auxiliary.set_default_size(-5, 100)
        self.assertEqual(auxiliary.get_default_size(), (200, 100))

    def test5(self):
        auxiliary.set_default_size(50, -5)
        self.assertEqual(auxiliary.get_default_size(), (50, 100))


class GrayscaleTest(unittest.TestCase):
    def test1(self):
        img = auxiliary.load_image("images/python.png")
        self.assertEqual(auxiliary.is_grayscale(img), False)

    def test2(self):
        img = auxiliary.load_image("images/python.png")
        img = auxiliary.to_grayscale(img)
        auxiliary.save_image("images/python_gray.png", img)
        self.assertEqual(auxiliary.is_grayscale(img), True)

    def test3(self):
        img = auxiliary.load_image("images/python_gray.png")
        self.assertEqual(auxiliary.is_grayscale(img), True)


class PreprocessImageTest(unittest.TestCase):
    def test1(self):
        img = auxiliary.load_image("images/python.png")
        img = auxiliary.resize_image(img, 400, 400, cv2.INTER_CUBIC)
        self.assertEqual(img.shape[:2], (400, 400))

    def test2(self):
        auxiliary.set_default_size(100, 100)
        auxiliary.interpolation = cv2.INTER_CUBIC
        img = auxiliary.preprocess_image("images/python.png")
        self.assertEqual(img.shape[:2], (100, 100))
        self.assertEqual(auxiliary.is_grayscale(img), True)


class ConcatenateImagesTest(unittest.TestCase):
    def test1(self):
        img1 = auxiliary.load_image("images/python.png")
        img1 = auxiliary.resize_image(img1, 100, 100, cv2.INTER_CUBIC)

        img2 = auxiliary.load_image("images/python_gray.png")
        img2 = auxiliary.resize_image(img2, 100, 100, cv2.INTER_CUBIC)

        img = auxiliary.concatenate_images(img1, img2)
        auxiliary.save_image("images/python_concatenated.png", img)

        img = auxiliary.load_image("images/python_concatenated.png")
        self.assertEqual(img.shape[:2], (100, 200))


class ExtractPathsTest(unittest.TestCase):
    def test1(self):
        paths = auxiliary.extract_images_paths("images/")
        self.assertEqual(len(paths), 3)


class WriteTextFileTest(unittest.TestCase):
    def test1(self):
        content = "WriteTextFileTest"
        fileName = "images/WriteTextFileTest.txt"
        auxiliary.write_text_file(content, fileName)

        paths = auxiliary.extract_files_paths("images/")
        self.assertEqual(len(paths), 4)


if __name__ == '__main__':
    unittest.main()

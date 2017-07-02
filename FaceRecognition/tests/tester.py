import glob
import unittest

# Get all files finished with Test.py
testFiles = glob.glob('test_*.py')

# Get a list with file name (without the extension .py)
fileNames = [testFile[0:len(testFile) - 3] for testFile in testFiles]

# Create the suites
suites = [unittest.defaultTestLoader.loadTestsFromName(
    fileName) for fileName in fileNames]

# Create and run the suite
suite = unittest.TestSuite(suites)
unittest.TextTestRunner().run(suite)

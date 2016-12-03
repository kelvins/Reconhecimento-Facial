
#!/usr/bin/python
# Filename: m10andl5.py

import cv2

# Function responsible for load the image, convert to grayscale and resize to a default size
def PreprocessImage(path, sizeX, sizeY):

    # Loads the image into the image variable
    image = cv2.imread(path)

    # Convert the image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image to a default size
    image = cv2.resize(image, (sizeX, sizeY), interpolation = cv2.INTER_CUBIC)

    return image

# Function that receives the path of each face image as parameter and include it in the training set (bf object)
def IncludeFace(path, sizeX, sizeY):

    # Get the subject id (should be a number)
    subjectID = path.split("_")[1]

    # Load Image, Convert to Grayscale, Resize
    image = PreprocessImage( path, sizeX, sizeY )

    return image, int(subjectID)
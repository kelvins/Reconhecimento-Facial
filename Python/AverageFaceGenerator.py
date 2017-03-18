
# Generates the average face

import os
import time
from PIL import Image
import numpy as np
import cv2

# Set the file path where we will search for the face images
filePath = 'C:\\Users\\x\\Desktop\\base1\\'

# Set the default size
width = 100
height = 100

# Store all np arrays
faceSpace = np.empty([width, height])

# Count how many files were used to generate the average face
count = 0

supported_formats = ["png", "jpg", "bmp"]

# Go through all files in the file path
for dirname, dirnames, filenames in os.walk(filePath):
    for filename in filenames:

        # Allowed file types: PNG, JPG and BMP
        if filename.split(".")[1] in supported_formats:

            # Create a temporary file name
            tempFileName = os.path.join(dirname, filename)

            # Open the image and convert to grayscale
            gray_image = Image.open(tempFileName).convert('L')

            # Resize the image to a default size
            gray_image = gray_image.resize((width, height), Image.ANTIALIAS)

            # Store the image as a numpy array
            imgArray = np.array(gray_image)

            # Sum the array to the faceSpace
            faceSpace = faceSpace + imgArray

            # Count how many face images we have
            count += 1

# Calculates an average numpy array
faceSpace = faceSpace / count

# Create the new image name following the pattern:
# year_month_day_hour_minute_second
fileName = time.strftime("%Y_%m_%d_%H_%M_%S")

# Save the image in PNG format
cv2.imwrite(filePath + fileName + '.png', faceSpace)

print 'It were used ' + str(count) + ' to generate the average face'

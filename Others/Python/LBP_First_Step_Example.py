
# First step of local binary patterns

import os, sys
from PIL import Image

path = '/Users/kelvinsp/Desktop/'
image = 'eu.jpg'

# Load and convert the image to gray scale
gray_image = Image.open(image).convert('L')
# Get the image as a matrix of pixels
pixels = gray_image.load()

tempGray_image = Image.open(image).convert('L')
tempPixels = tempGray_image.load()

# For each pixel
for row in range(1, gray_image.size[0]-1):
    for col in range(1, gray_image.size[1]-1):

        binaryResult = ""
        threshold = pixels[row, col]

        # 3 x 3
        for tempRow in range(row-1, row+2):
            for tempCol in range(col-1, col+2):

                # If is not the threshold point
                if tempRow != row or tempCol != col:

                    if pixels[tempRow, tempCol] >= threshold:
                        binaryResult += "1"
                    else:
                        binaryResult += "0"

        tempPixels[row, col] = int(binaryResult, 2)

#gray_image.show()
#tempGray_image.show()

tempGray_image.save(path + 'newLBPimage.jpg')

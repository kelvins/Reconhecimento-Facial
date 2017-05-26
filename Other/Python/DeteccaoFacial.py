# Importa as todas bibliotecas que serao utilizadas
import cv2, sys, os, time
import numpy as np
from PIL import Image

scaleFactor = 1.2
min_neighbors = 5
min_size = 10

videosPath = "/home/kelvin/Desktop/FaceRecognition/Videos/"
video = "grupos_MAOS_1.MPG"

classifiersPath = "/home/kelvin/Desktop/FaceRecognition/Cascade_Classifiers/haarcascades_cuda/"
cascadeClassifier = "haarcascade_frontalface_default.xml"
#cascadeClassifier = "lbpcascade_frontalface.xml"
#cascadeClassifier = "haarcascade_frontalface_alt2.xml"
#cascadeClassifier = "haarcascade_frontalface_alt_tree.xml"

print "Started at " + time.strftime("%d-%m-%Y %H-%M-%S")
content = "Started at " + time.strftime("%d-%m-%Y %H-%M-%S") + "\n\n"
content += "Face Cascade Parameters : \n - Scale Factor : " + str(scaleFactor) + " \n - Min. Neighbors : " + str(min_neighbors) + " \n - Min. Size : " + str(min_size) + ", " + str(min_size) + " \n - Flags : cv2.CASCADE_SCALE_IMAGE\n\n"
content += "Cascade Classifier : " + cascadeClassifier + "\n"
content += "Video : " + video + "\n"

# Caminho do arquivo XML do classificador
cascPath = classifiersPath + cascadeClassifier

# Cria um objeto para o classificador
faceCascade = cv2.CascadeClassifier(cascPath)

# Caminho do video que sera analisado
video_capture = cv2.VideoCapture(videosPath + video)

# Define os frames per second
#print video_capture.get(cv2.CAP_PROP_FPS)
#video_capture.set(cv2.CAP_PROP_FPS, 60)
#print video_capture.get(cv2.CAP_PROP_FPS)

# Variavel global que controla o numero de imagens faciais encontradas
totalFacesDetected = 0
countFrames = 0

# Folder name based on the date/time
folder = "Deteccao_Facial_" + time.strftime("%Y-%m-%d_%H-%M-%S")
# Create a new folder
os.makedirs(folder)

content += "FPS : " + str(video_capture.get(cv2.CAP_PROP_FPS)) + "\n\n"
appendContent = ""

while True:

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if frame is None:
        break

    countFrames += 1

    # Gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = scaleFactor,
        minNeighbors = min_neighbors,
        minSize = (min_size, min_size),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    # Para cada face detectada
    for (x, y, w, h) in faces:
        
        totalFacesDetected += 1

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        crop_face = frame[y: y + h, x: x + w]

        appendContent += "Frame: " + str(countFrames) + " Image size w: " + str(w) + " h: " + str(h) + "\n"; 
        print "Frame: " + str(countFrames) + " Image size w: " + str(w) + " h: " + str(h); 

        cv2.imwrite(folder + "/" + str(totalFacesDetected) + ".png", crop_face)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Wait until the key Q or C was pressed and then quit/close
    if( cv2.waitKey(1) & 0xFF == ord('q') ) :
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

content += "\nTotal Frames : " + str(countFrames) + "\n"
content += "\nTotal Faces Detected : " + str(totalFacesDetected) + "\n"

print "Finished at " + time.strftime("%d-%m-%Y %H-%M-%S")
content += "Finished at " + time.strftime("%d-%m-%Y %H-%M-%S") + "\n\n"

content += appendContent

txtFile = open(folder + "/report.txt", "w")
txtFile.write(content)
txtFile.close()

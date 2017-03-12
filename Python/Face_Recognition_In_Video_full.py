# Importa as todas bibliotecas que serao utilizadas
import cv2, sys, os, time
import numpy as np
from PIL import Image

for p in xrange(1,4):

	if p == 1:
		# Parameters
		scaleFactor = 1.05
		min_neighbors = 5
		min_size = 30
	elif p == 2:
		# Parameters
		scaleFactor = 1.1
		min_neighbors = 5
		min_size = 30
	elif p == 3:
		# Parameters
		scaleFactor = 1.2
		min_neighbors = 5
		min_size = 10


	for x in xrange(1,13):
		
		# Content that will be saved in the text file
		content = ""
		general_content = ""
		confidence_threshold = 0

		imagesPath = "/home/kelvin/Desktop/Mestrado/Photos/faces_png2/"
		videosPath = "/home/kelvin/Desktop/Mestrado/Videos/"

		if x == 1 or x == 2 or x == 3:
			# Paths
			classifiersPath = "/home/kelvin/Desktop/Mestrado/Cascade_Classifiers/haarcascades_cuda/"
			cascadeClassifier = "haarcascade_frontalface_default.xml"
		if x == 4 or x == 5 or x == 6:
			# Paths
			classifiersPath = "/home/kelvin/Desktop/Mestrado/Cascade_Classifiers/lbpcascades/"
			cascadeClassifier = "lbpcascade_frontalface.xml"
		if x == 7 or x == 8 or x == 9:
			# Paths
			classifiersPath = "/home/kelvin/Desktop/Mestrado/Cascade_Classifiers/haarcascades_cuda/"
			cascadeClassifier = "haarcascade_frontalface_alt2.xml"
		if x == 10 or x == 11 or x == 12:
			# Paths
			classifiersPath = "/home/kelvin/Desktop/Mestrado/Cascade_Classifiers/haarcascades_cuda/"
			cascadeClassifier = "haarcascade_frontalface_alt_tree.xml"

		faceRecognizer = ""
		if x == 1 or x == 4 or x == 7 or x == 10:
			recognizer = cv2.face.createLBPHFaceRecognizer()
			faceRecognizer = "LBPH"
			confidence_threshold = 80
		if x == 2 or x == 5 or x == 8 or x == 11:
			recognizer = cv2.face.createEigenFaceRecognizer()
			faceRecognizer = "Eigen Faces"
			confidence_threshold = 1500
		if x == 3 or x == 6 or x == 9 or x == 12:
			recognizer = cv2.face.createFisherFaceRecognizer()
			faceRecognizer = "Fisher Faces"
			confidence_threshold = 800

		content += "Face Recognizer : " + faceRecognizer + "\n"

		video = "grupos_MAOS_1.MPG"

		print "Started at " + time.strftime("%d-%m-%Y %H-%M-%S")
		content += "Started at " + time.strftime("%d-%m-%Y %H-%M-%S") + "\n\n"

		general_content = time.strftime("%d-%m-%Y %H-%M-%S") + ";" + cascadeClassifier + ";" + video + ";" + faceRecognizer + ";" + str(scaleFactor) + ";" + str(min_neighbors) + ";" + str(min_size) + ";"

		# Caminho do arquivo XML do classificador
		cascPath = classifiersPath + cascadeClassifier
		content += "Cascade Classifier : " + cascadeClassifier + "\n"
		# Cria um objeto para o classificador
		faceCascade = cv2.CascadeClassifier(cascPath)

		# Caminho do video que sera analisado
		video_capture = cv2.VideoCapture(videosPath + video)
		content += "Video : " + video + "\n"

		# Define os frames per second
		#print video_capture.get(cv2.CAP_PROP_FPS)
		#video_capture.set(cv2.CAP_PROP_FPS, 60)
		#print video_capture.get(cv2.CAP_PROP_FPS)

		# Variavel global que controla o numero de imagens faciais encontradas para ser utilizado como label
		images_count = 1

		# Vetores que irao armazenar as imagens e o titulo das imagens para utilizar no treinamento
		images = []
		labels = []

		# Vetor que guarda uma imagem de cada sujeito apenas para gerar a imagem de saida com o sujeito correto
		subjects = [None]*19
		subject_count = 1

		# Folder name based on the date/time
		folder = time.strftime("%Y-%m-%d_%H-%M-%S")
		# Create a new folder
		os.makedirs(folder)
		# Create a new folder
		os.makedirs(str(folder)+"/CORRECT")
		os.makedirs(str(folder)+"/INCORRECT")

		# Funcao que recebe o caminho da imagem e busca as faces presentes na imagem, inserindo as 
		# faces encontradas no vetor images e o nome/titulo no vetor labels
		def find_face(path):
		    # Utiliza a variavel global
		    global images_count
		    # Utilizara os vetores globais
		    global images
		    global labels
		    global subjects
		    global subject_count

		    subject = path.split("_")[2]

		    # Le a imagem e carrega ela na variavel image
		    image = cv2.imread(path)
		    # Transforma a imagem para escala de cinza e cria a nova imagem na variavel gray_image
		    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		    # Chama a funcao que detecta as faces passando como parametro a imagem e alguns outros parametros necessarios
		    face = faceCascade.detectMultiScale(
		        gray_image,
		        scaleFactor = scaleFactor,
		        minNeighbors = min_neighbors,
		        minSize = (min_size, min_size),
		        flags = cv2.CASCADE_SCALE_IMAGE
		    )

		    # Laco que percorre todas as faces encontradas na imagem e encontra a posicao (X e Y) e tamanho da imagem facial (H e W)
		    for (x, y, w, h) in face:
		        # Recorta apenas a imagem da face
		        crop_face = gray_image[y:y+h, x:x+w]
		        crop_face = cv2.resize(crop_face, (100,100), interpolation = cv2.INTER_CUBIC)
		        # Adiciona a imagem recortada ao vetor images
		        images.append( crop_face )
		        # Adiciona um label/titulo para a imagem
		        labels.append( int(subject) )

		        subjects[int(subject)-1] = crop_face

		        # Incrementa o numero de imagens contradas
		        images_count += 1
		        # Mostra a imagem na tela
		        #cv2.imshow("Adding faces to traning set...", cv2.rectangle(image, (x, y), (x+w, y+h), (250, 250, 250), 1))
		        # Sleep/Tempo que a imagem ira aparecer na tela para o usuario
		        #cv2.waitKey(50)q


		image_paths = os.listdir(imagesPath)
		# Laco responsavel por carregar todas as imagens para treinamento
		# Neste caso esta pegando todas as imagens frontais (das fotos do cadastro)
		#for x in xrange(1,20):
		    # Chama a funcao que procura as faces na imagem
		#    find_face(imagesPath + str(x) + ".jpg")
		for x in xrange(0,len(image_paths)):
		    # Chama a funcao que procura as faces na imagem
		    find_face(imagesPath + image_paths[x])

		# Chama a funcao de treinamento passando como parametro as imagens e os titulos
		recognizer.train(images, np.array(labels))

		countFrames = 0

		content += "FPS : " + str(video_capture.get(cv2.CAP_PROP_FPS)) + "\n"
		content += "Face Cascade Parameters : \n - Scale Factor : " + str(scaleFactor) + " \n - Min. Neighbors : " + str(min_neighbors) + " \n - Min. Size : " + str(min_size) + ", " + str(min_size) + " \n - Flags : cv2.CASCADE_SCALE_IMAGE\n\n"

		totalFacesDetected = 0
		totalFacesRecognized = 0
		totalFacesIncorrectRecognized = 0

		min_confidence = 9999999999999
		max_confidence = -1
		avg_confidence = 0

		content_results = ""

		while True:

		    # Capture frame-by-frame
		    ret, frame = video_capture.read()

		    if frame is None:
		        break

		    countFrames += 1

		    # Gray scale
		    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		    #faces = faceCascade.detectMultiScale(
		    faces = faceCascade.detectMultiScale(
		        gray,
		        scaleFactor = scaleFactor,
		        minNeighbors = min_neighbors,
		        minSize = (min_size, min_size),
		        flags= cv2.CASCADE_SCALE_IMAGE
		    )

		    # Draw a rectangle around the faces
		    for (x, y, w, h) in faces:
		    	totalFacesDetected += 1
		        #cv2.rectangle(frame, (x, y), (x+w, y+h), (250, 250, 250), 1)

		        #croppedImage = frame[y:y+h+1, x:x+w+1]
		        #capturedFaces.append( croppedImage )
		        #n_faces += 1

		        # Chama a funcao que faz a precicao passando como parametro apenas a imagem facial ja recortada
		        # a funcao retorna o numero da imagem que encontrou e a conficanca
		        crop_face = frame[y: y + h, x: x + w]
		        gray_image = cv2.cvtColor(crop_face, cv2.COLOR_BGR2GRAY)
		        gray_image = cv2.resize(gray_image, (100,100), interpolation = cv2.INTER_CUBIC)
		        nbr_predicted, conf = recognizer.predict( gray_image )
		        #nbr_predicted, conf = recognizer.predict( predict_image[y: y + h, x: x + w] )
		        #nbr_predicted, conf = recognizer.predict( predict_image )

		        # Recebe o numero da imagem atual para comparar com o numero da imagem nbr_predicted
		        nbr_actual = countFrames #int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))

		        if conf < min_confidence:
		        	min_confidence = conf
		        elif conf > max_confidence:
		        	max_confidence = conf

		        # Se o numero da imagem predita for igual ao da imagem que esta sendo comparada entao foi reconhecido com sucesso
		        # aqui tem um problema pois o numero pode variar entao acredito que seja melhor se basear pela confianca
		        #if nbr_actual == nbr_predicted:
		        avg_confidence += conf

		        if conf < confidence_threshold:
		            print "Subject from frame {} is Correctly Recognized as subject {} with confidence {}".format(nbr_actual, nbr_predicted, conf)
		            filename = "Subject_from_Frame_" + str(nbr_actual) + "_Predicted_as_Subject_" + str(nbr_predicted) + "_with_Confidence_" + str(round(conf, 6)) + ".png"
		            
		            totalFacesRecognized += 1

		            content_results += filename + "\n"
		            filename = folder + "/CORRECT/" + filename
		        else:
		            print "Subject from frame {} is INCORRECT Recognized as subject {} with confidence {}".format(nbr_actual, nbr_predicted, conf)
		            filename = "Subject_from_Frame_" + str(nbr_actual) + "_INCORRECT_Predicted_as_Subject_" + str(nbr_predicted) + "_with" + "_Confidence_" + str(round(conf, 6)) + ".png"
		            
		            totalFacesIncorrectRecognized += 1

		            content_results += filename + "\n"
		            filename = folder + "/INCORRECT/" + filename

		        # Coloca as duas imagens uma ao lado da outra para salvar
		        """h1, w1 = gray_image.shape[:2]
		        h2, w2 = images[nbr_predicted-1].shape[:2]
		        vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
		        vis[:h1, :w1] = gray_image
		        vis[:h2, w1:w1+w2] = images[nbr_predicted-1]
		        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)"""
		        if nbr_predicted-1 <= len(subjects)-1:
		        	vis = np.concatenate((gray_image, subjects[nbr_predicted-1]), axis=1)
		        	# Salva a imagem
		        	cv2.imwrite(filename, vis)

		    # Display the resulting frame
		    cv2.imshow('Video', frame)

		    # Wait until the key Q or C was pressed and then quit/close
		    if( cv2.waitKey(1) & 0xFF == ord('q') ) :
		        break

		content += "\n"

		if totalFacesDetected > 0:
			avg_confidence = avg_confidence/totalFacesDetected

		# When everything is done, release the capture
		video_capture.release()
		cv2.destroyAllWindows()

		content += "Total Faces Detected : " + str(totalFacesDetected) + "\n"
		content += "Total Faces Recognized : " + str(totalFacesRecognized) + "\n"
		content += "Total Faces Incorrectly Recognized : " + str(totalFacesIncorrectRecognized) + "\n\n"

		content += "Confidence Threshold : " + str(confidence_threshold) + "\n\n"

		content += "Min. Confidence : " + str(min_confidence) + "\n"
		content += "Max. Confidence : " + str(max_confidence) + "\n"
		content += "Avg. Confidence : " + str(avg_confidence) + "\n\n"

		print "Finished at " + time.strftime("%d-%m-%Y %H-%M-%S")
		content += "Finished at " + time.strftime("%d-%m-%Y %H-%M-%S") + "\n\n"

		content += content_results

		general_content += str(totalFacesDetected) + ";" + str(totalFacesRecognized) + ";" + str(totalFacesIncorrectRecognized) + ";" + str(confidence_threshold) + ";" + str(min_confidence) + ";" + str(max_confidence) + ";" + str(avg_confidence) + "\n"

		txtFile = open(folder + "/report.txt", "w")
		txtFile.write(content)
		txtFile.close()

		txtFile2 = open("general_report.txt", "a")
		txtFile2.write(general_content)
		txtFile2.close()

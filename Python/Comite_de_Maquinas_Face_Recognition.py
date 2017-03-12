# Importa as todas bibliotecas que serao utilizadas
import cv2, sys, os, time
import numpy as np
from PIL import Image

# Mostra a data/hora de inicio da execucao do algoritmo
print "Started at " + time.strftime("%d-%m-%Y %H-%M-%S")

# Parametros
scaleFactor = 1.1
min_neighbors = 5
min_size = 30

# Numero de participantes
nSubjects = 19

# Variavel que armazena o conteudo que sera salvo no arquivo de texto
content = ""
# Variavel que armazena o conteudo que sera salvo no arquivo geral (onde armazena todos os resultados em cada linha)
general_content = ""

# Caminho para a pasta onde estao as imagens para treinamento
imagesPath = "/home/kelvin/Desktop/Mestrado/Photos/faces_png2/"

# Caminho para a pasta onde esta o video
videosPath = "/home/kelvin/Desktop/Mestrado/Videos/"
# Nome do arquivo de video que sera utilizado para realizar o reconhecimento
video = "grupos_MAOS_1.MPG"
# Caminho do video que sera analisado
video_capture = cv2.VideoCapture(videosPath + video)

# Caminho para a pasta onde esta o arquivo XML do classificador em cascata
classifiersPath = "/home/kelvin/Desktop/Mestrado/Cascade_Classifiers/haarcascades_cuda/"
# Nome do arquivo XML classificador
cascadeClassifier = "haarcascade_frontalface_default.xml"
# Caminho do arquivo XML do classificador
cascPath = classifiersPath + cascadeClassifier

# Cria um objeto para o classificador
faceCascade = cv2.CascadeClassifier(cascPath)

# Nome do algoritmo que sera utilizado (esta variavel eh utilizada no arquivo de texto de saida)
faceRecognizer = "Eigen Faces / Fisher Faces / LBPH"

# Cria o objeto do algoritmo de reconhecimento facial
recognizer1 = cv2.face.createEigenFaceRecognizer()
recognizer2 = cv2.face.createFisherFaceRecognizer()
recognizer3 = cv2.face.createLBPHFaceRecognizer()

# Define o limiar que sera utilizado (depende do algoritmo de reconhecimento)
confidence_threshold1 = 1500
confidence_threshold2 = 1500
confidence_threshold3 = 1500

#recognizer = cv2.face.createLBPHFaceRecognizer()
#faceRecognizer = "LBPH"
#confidence_threshold = 80

#recognizer = cv2.face.createFisherFaceRecognizer()
#faceRecognizer = "Fisher Faces"
#confidence_threshold = 800

# Coloca o nome do algoritmo na variavel de saida
content += "Face Recognizer : " + faceRecognizer + "\n"
# Coloca a data/hora de inicio na variavel de saida
content += "Started at " + time.strftime("%d-%m-%Y %H-%M-%S") + "\n\n"
# Coloca o nome do classificador na variavel de saida
content += "Cascade Classifier : " + cascadeClassifier + "\n"
# Coloca o nome do video na variavel de saida
content += "Video : " + video + "\n"
# Coloca os parametros utilizados na variavel de saida
content += "Face Cascade Parameters : \n - Scale Factor : " + str(scaleFactor) + " \n - Min. Neighbors : " + str(min_neighbors) + " \n - Min. Size : " + str(min_size) + ", " + str(min_size) + " \n - Flags : cv2.CASCADE_SCALE_IMAGE\n\n"

# Guarda o conteudo na variavel de saida para gravar no arquivo geral
general_content = time.strftime("%d-%m-%Y %H-%M-%S") + ";" + cascadeClassifier + ";" + video + ";" + faceRecognizer + ";" + str(scaleFactor) + ";" + str(min_neighbors) + ";" + str(min_size) + ";"

# Variavel global que controla o numero de imagens faciais encontradas para ser utilizado como label
images_count = 1

# Vetores que irao armazenar as imagens e o titulo das imagens para utilizar no treinamento
images = []
labels = []

# Vetor que guarda uma imagem de cada sujeito apenas para gerar a imagem de saida com o sujeito correto
subjects = [None] * nSubjects
subject_count = 1

# Define o nome da pasta com base na data e hora atual
folder = time.strftime("%Y-%m-%d_%H-%M-%S")
# Cria a pasta
os.makedirs(folder)
# Cria duas pastas, dentro da pasta anterior, para salvar as imagens reconhecidas corretamente e incorretamente
os.makedirs(str(folder)+"/CORRECT")
os.makedirs(str(folder)+"/INCORRECT")

# Funcao que recebe o caminho da imagem e busca as faces presentes na imagem, inserindo as
# faces encontradas no vetor images e o nome/titulo no vetor labels
def find_face(path):
    # Utiliza as variaveis globais
    global images_count
    global images
    global labels
    global subjects
    global subject_count

	# Separa o nome da image, criando um vetor de string, e obtem o 3 elemento do vetor
    subject = path.split("_")[2]

    # Le a imagem e carrega ela na variavel image
    image = cv2.imread(path)

    # Transforma a imagem para escala de cinza e cria a nova imagem na variavel gray_image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Chama a funcao que detecta as faces passando como parametro a imagem e
	# alguns outros parametros necessarios definidos no comeco do algoritmo
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
		# Redimensiona a imagem para ficar com tamanho padrao 100x100
        crop_face = cv2.resize(crop_face, (100,100), interpolation = cv2.INTER_CUBIC)

        # Adiciona a imagem recortada ao vetor images
        images.append( crop_face )
        # Adiciona um label/titulo para a imagem
        labels.append( int(subject) )

		# Salva a imagem que ira representar o sujeito
		# Eh utilizada para salvar com juntamente com a face reconhecida
        subjects[int(subject)-1] = crop_face

        # Incrementa o numero de imagens encontradas
        images_count += 1

        # Mostra a imagem na tela
        #cv2.imshow("Adding faces to traning set...", cv2.rectangle(image, (x, y), (x+w, y+h), (250, 250, 250), 1))
        # Sleep/Tempo que a imagem ira aparecer na tela para o usuario
        #cv2.waitKey(50)


# Obtem o nome de todos os arquivos encontrados na pasta imagesPath
image_paths = os.listdir(imagesPath)

# Laco responsavel por carregar todas as imagens para treinamento
# Neste caso esta pegando todas as imagens frontais (das fotos do cadastro)
#for x in xrange(1,20):
    # Chama a funcao que procura as faces na imagem
	# find_face(imagesPath + str(x) + ".jpg")
for x in xrange(0,len(image_paths)):
    # Chama a funcao que procura as faces na imagem
    find_face(imagesPath + image_paths[x])

# Chama a funcao de treinamento passando como parametro as imagens e os titulos
recognizer1.train(images, np.array(labels))
recognizer2.train(images, np.array(labels))
recognizer3.train(images, np.array(labels))

# Variavel que sera utilizada para contar os frames do video
countFrames = 0

# Coloca o numero de frames por segundo na variavel de saida
content += "FPS : " + str(video_capture.get(cv2.CAP_PROP_FPS)) + "\n"

# Variavel que armazena o total de faces detectadas
totalFacesDetected = 0
# Variavel que armazena o total de faces reconhecidas
totalFacesRecognized = 0
# Variavel que armazena o total de faces incorretamente reconhecidas
totalFacesIncorrectRecognized = 0

# Variaval que armazena a menor confianca encontrada ("melhor resultado")
min_confidence = 9999999999999
# Variavel que armazena a maior confianca encontrada ("pior resultado")
max_confidence = -1
# Variavel que armazena a media das confiancas encontradas
avg_confidence = 0

content_results = ""

# Laco principal que ira percorrer o video frame a frame
while True:

    # Captura frame a frame
    ret, frame = video_capture.read()

	# Se nao existe frame quebra o laco
    if frame is None:
        break

	# Conta mais um frame
    countFrames += 1

    # Converte o frame atual para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Procura pelas faces no frame atual (escala de cinza)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = scaleFactor,
        minNeighbors = min_neighbors,
        minSize = (min_size, min_size),
        flags= cv2.CASCADE_SCALE_IMAGE
    )

    # Laco que percorre cada imagem facial encontrada no frame atual
    for (x, y, w, h) in faces:

		# Incrementa a variavel que armazena o total de faces detectadas
    	totalFacesDetected += 1

		# Desenha um retangulo na face encontrada
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (250, 250, 250), 1)

		# Recorta a imagem facial encontrada no frame
		crop_face = frame[y: y + h, x: x + w]
		# Converte a imagem facial para escala de cinza
        gray_image = cv2.cvtColor(crop_face, cv2.COLOR_BGR2GRAY)
		# Redimensiona a imagem para 100x100 para ficar padrao, igual ao treinamento
        gray_image = cv2.resize(gray_image, (100,100), interpolation = cv2.INTER_CUBIC)

        # Chama a funcao que faz a predicao passando como parametro apenas a imagem facial ja recortada e em escala de cinza
        # A funcao retorna o numero da imagem que o algoritmo "reconheceu" e a conficanca encontrada
        nbr_predicted1, conf1 = recognizer1.predict( gray_image )
        nbr_predicted2, conf2 = recognizer2.predict( gray_image )
        nbr_predicted3, conf3 = recognizer3.predict( gray_image )

		# Se a confianca atual eh menor que a menor confianca encontrada, atribui ela a variavel min_confidence
        if conf1 < min_confidence:
        	min_confidence = conf1
		# Senao, se a confianca atual eh maior que a maior confianca encontrada, atribui ela a variavel max_confidence
        elif conf1 > max_confidence:
        	max_confidence = conf1
		# Se a confianca atual eh menor que a menor confianca encontrada, atribui ela a variavel min_confidence
        if conf2 < min_confidence:
        	min_confidence = conf2
		# Senao, se a confianca atual eh maior que a maior confianca encontrada, atribui ela a variavel max_confidence
        elif conf2 > max_confidence:
        	max_confidence = conf2
		# Se a confianca atual eh menor que a menor confianca encontrada, atribui ela a variavel min_confidence
        if conf3 < min_confidence:
        	min_confidence = conf3
		# Senao, se a confianca atual eh maior que a maior confianca encontrada, atribui ela a variavel max_confidence
        elif conf3 > max_confidence:
        	max_confidence = conf3

		# Soma a confianca atual a variavel avg_confidence para ao final calcular a confianca media
		avg_confidence += conf1
        avg_confidence += conf2
        avg_confidence += conf3

        # Se a confianca atual for menor que o limiar, significa que a face foi corretamente reconhecida
        if nbr_predicted1 == nbr_predicted2 or nbr_predicted1 == nbr_predicted3:
			# Mostra na tela o numero do frame, o numero do sujeito e a confianca atual
            print "Subject from frame {} is Correctly Recognized as subject {} with confidence {}".format(countFrames, nbr_predicted1, conf1)
			# Cria uma variavel com o nome do arquivo que sera salvo, contem o numero do frame, o numero do sujeito e a confianca atual
			filename = "Subject_from_Frame_" + str(countFrames) + "_Predicted_as_Subject_" + str(nbr_predicted1) + "_with_Confidence_" + str(round(conf1, 6)) + ".png"

			# Incrementa a variavel que armazena o total de faces corretamente reconhecidas
            totalFacesRecognized += 1

			# Adiciona o nome do arquivo atual (imagem) na variavel content_results que sera utilizada no arquivo de saida (log txt)
            content_results += filename + "\n"
			# Edita a variavel filename inserindo o caminho inteiro onde a imagem sera salva
            filename = folder + "/CORRECT/" + filename

    		# Se o sujeito que foi reconhecido esta entre os sujeitos "conhecidos" (ou seja, que estao no vetor subjects)
            if nbr_predicted1-1 <= len(subjects)-1:
    			# Coloca as duas imagens uma ao lado da outra para salvar
            	vis = np.concatenate((gray_image, subjects[nbr_predicted1-1]), axis=1)
            	# Salva a imagem no caminho especificado pela variavel filename
            	cv2.imwrite(filename, vis)

        elif nbr_predicted2 == nbr_predicted1 or nbr_predicted2 == nbr_predicted3:
			# Mostra na tela o numero do frame, o numero do sujeito e a confianca atual
            print "Subject from frame {} is Correctly Recognized as subject {} with confidence {}".format(countFrames, nbr_predicted2, conf2)
			# Cria uma variavel com o nome do arquivo que sera salvo, contem o numero do frame, o numero do sujeito e a confianca atual
			filename = "Subject_from_Frame_" + str(countFrames) + "_Predicted_as_Subject_" + str(nbr_predicted2) + "_with_Confidence_" + str(round(conf2, 6)) + ".png"

			# Incrementa a variavel que armazena o total de faces corretamente reconhecidas
            totalFacesRecognized += 1

			# Adiciona o nome do arquivo atual (imagem) na variavel content_results que sera utilizada no arquivo de saida (log txt)
            content_results += filename + "\n"
			# Edita a variavel filename inserindo o caminho inteiro onde a imagem sera salva
            filename = folder + "/CORRECT/" + filename

    		# Se o sujeito que foi reconhecido esta entre os sujeitos "conhecidos" (ou seja, que estao no vetor subjects)
            if nbr_predicted2-1 <= len(subjects)-1:
    			# Coloca as duas imagens uma ao lado da outra para salvar
            	vis = np.concatenate((gray_image, subjects[nbr_predicted2-1]), axis=1)
            	# Salva a imagem no caminho especificado pela variavel filename
            	cv2.imwrite(filename, vis)

        else:
			# Mostra na tela o numero do frame, o numero do sujeito e a confianca atual
            print "Subject from frame {} is INCORRECT Recognized as subject {} with confidence {}".format(countFrames, nbr_predicted, conf)
			# Cria uma variavel com o nome do arquivo que sera salvo, contem o numero do frame, o numero do sujeito e a confianca atual
            filename = "INCORRECT_Subject_from_Frame_" + str(countFrames) + "_Predicted_as_Subject_" + str(nbr_predicted1) + "_with_Confidence_" + str(round(conf1, 6)) + ".png"

			# Incrementa a variavel que armazena o total de faces incorretamente reconhecidas
            totalFacesIncorrectRecognized += 1

			# Adiciona o nome do arquivo atual (imagem) na variavel content_results que sera utilizada no arquivo de saida (log txt)
            content_results += filename + "\n"
			# Edita a variavel filename inserindo o caminho inteiro onde a imagem sera salva
            filename = folder + "/INCORRECT/" + filename

    		# Se o sujeito que foi reconhecido esta entre os sujeitos "conhecidos" (ou seja, que estao no vetor subjects)
            if nbr_predicted1-1 <= len(subjects)-1:
    			# Coloca as duas imagens uma ao lado da outra para salvar
            	vis = np.concatenate((gray_image, subjects[nbr_predicted1-1]), axis=1)
            	# Salva a imagem no caminho especificado pela variavel filename
            	cv2.imwrite(filename, vis)

    # Mostra o frame atual na tela
    cv2.imshow('Video', frame)

	# Aguarda ate que a tecla Q seja pressionada e entao fecha a tela
    if( cv2.waitKey(1) & 0xFF == ord('q') ) :
        break

# Quando terminar de passar por todos os frames do video
# Libera a variavel de captura de frame
video_capture.release()
# Destroi todas as janelas atuais do opencv
cv2.destroyAllWindows()

# Se existe pelo menos uma face detectada, calcula a media de confianca
if totalFacesDetected > 0:
	# Soma de todas as confiancas dividido pelo total de faces detectadas
	avg_confidence = avg_confidence/(totalFacesDetected*3)

# Coloca todos os resultados na variavel de saida
# Faces detectadas, reconhecidas corretamente e incorretamente
content += "\nTotal Faces Detected : " + str(totalFacesDetected) + "\n"
content += "Total Faces Recognized : " + str(totalFacesRecognized) + "\n"
content += "Total Faces Incorrectly Recognized : " + str(totalFacesIncorrectRecognized) + "\n\n"

# Limiar de confianca
content += "Confidence Threshold : " + str(confidence_threshold1) + " - "  + str(confidence_threshold2) + " - " + str(confidence_threshold3) + "\n\n"

# Confiancas: minima, maxima e media
content += "Min. Confidence : " + str(min_confidence) + "\n"
content += "Max. Confidence : " + str(max_confidence) + "\n"
content += "Avg. Confidence : " + str(avg_confidence) + "\n\n"

# Coloca a data e hora final na variavel de saida
content += "Finished at " + time.strftime("%d-%m-%Y %H-%M-%S") + "\n\n"

# Coloca os resultados de todas as imagens detectadas na variavel de saida
# (incluindo numero do frame, do sujeito e a confianca)
content += content_results

# Adiciona mais conteudo a variavel geral (que salva todos os resultados em uma tabela)
general_content += str(totalFacesDetected) + ";" + str(totalFacesRecognized) + ";" + str(totalFacesIncorrectRecognized) + ";-;" + str(min_confidence) + ";" + str(max_confidence) + ";" + str(avg_confidence) + "\n"

# Abre o arquivo de report somente escrita, para salvar o conteudo
txtFile = open(folder + "/report.txt", "w")
# Escreve o conteudo que esta na variavel content no arquivo de texto
txtFile.write(content)
# Fecha o arquivo
txtFile.close()

# Abre o arquivo general_report para "append"
txtFile2 = open("general_report.txt", "a")
# Escreve o conteudo da variavel general_content (resultados globais de todos os algoritmos executados)
txtFile2.write(general_content)
# Fecha o arquivo
txtFile2.close()

# Mostra a data e hora do fim da execucao do algoritmo
print "Finished at " + time.strftime("%d-%m-%Y %H-%M-%S")

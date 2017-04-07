
import FaceRecognition as fr
from FaceRecognition import Algorithms, Interpolation

# Define o caminho para a pasta de treinamento
trainPath = "/home/Desktop/FaceRecognition/BASE1/"

# Define o caminho para a pasta de teste
testPath = "/home/Desktop/FaceRecognition/TESTE1/"

faceRec = fr.FaceRecognition()

# Define qual algoritmo sera utilizado (padrao: EIGENFACES)
# [Algorithms.EIGENFACES, Algorithms.FISHERFACES, Algorithms.LBPH, Algorithms.SIFT, Algorithms.SURF]
faceRec.setAlgorithm(Algorithms.EIGENFACES)

# Define qual metodo de interpolacao sera utilizado (padrao: INTER_CUBIC)
# [Interpolation.INTER_CUBIC, Interpolation.INTER_NEAREST, Interpolation.INTER_LINEAR, Interpolation.INTER_AREA, Interpolation.INTER_LANCZOS4]
faceRec.setInterpolation(Interpolation.INTER_CUBIC)

# Define o tamanho padrao para as imagens (padrao 100x100)
# Neste caso 100x100 (larguraXaltura)
faceRec.setDefaultSize(100, 100)

# Define o caminho da pasta de treinamento
faceRec.train(trainPath)

# Define o caminho da pasta de teste
faceRec.predict(testPath)

# Mostra os resultados na tela
faceRec.showResults()

# Salva os resultados em uma pasta
# O caminho da pasta pode ser passado como parametro (ex.: faceRec.save('/home/Desktop/FaceRecognition/Results'))
# Se o parametro estiver vazio a funcao ira criar uma pasta seguindo o
# padrao: ano_mes_dia_hora_minuto_segundo_algoritmo_parametros
faceRec.save()

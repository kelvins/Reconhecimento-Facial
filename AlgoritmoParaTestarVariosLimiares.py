
# Algoritmo desenvolvido para testar o reconhecimento com varios limiares e armazenar os resultados, para futuramente gerar a curva ROC

# Importa todas as bibliotecas que serao utilizadas
import cv2, sys, os, time
import numpy as np
from PIL import Image

########################## DEFINICAO DOS PARAMETROS ##########################

# Caminho para as pastas que contem as imagens para treinamento e testes
caminhoParaPastaTreinamento = "/home/kelvin/Desktop/FaceRecognition/TestesCurvaROC/Base3/"
caminhoParaPastaTeste = "/home/kelvin/Desktop/FaceRecognition/TestesCurvaROC/Teste1/"

# Define o algoritmo de reconhecimento facial que sera utilizado
algoritmo = "lbph"
#eigenface = 1500
#fisherface = 700
#LBPH = 80

# Define os limiares minimo, maximo e o passo
limiarMin = 40
limiarMax = 120
passo = 2

##############################################################################

# Nome do arquivo de texto que sera salvo - nome gerado de acordo com a data e hora
nomeArquivo = time.strftime("%Y-%m-%d_%H-%M-%S")

# Variavel que ira guardar o conteudo que sera salvo no arquivo de texto
conteudo = ""

# Cria uma variavel com a hora de inicio dos testes
horaInicio = "Hora de Inicio  : " + time.strftime("%d/%m/%Y %H:%M:%S")

# Mostra a hora de inicio do teste - apenas para saber o tempo estimado
print horaInicio

# Adiciona os parametros utilizados no conteudo que sera salvo
conteudo += "\n####################### - Parametros Utilizados - ######################\n"
conteudo += "Algoritmo utilizado : " + algoritmo.upper()
conteudo += "\nPasta de treinamento : " + caminhoParaPastaTreinamento
conteudo += "\nPasta de teste : " + caminhoParaPastaTeste
conteudo += "\nLimiar minimo : " + str(limiarMin)
conteudo += "\nLimiar maximo : " + str(limiarMax)
conteudo += "\nPasso : " + str(passo)
conteudo += "\n########################################################################\n"

# Adiciona uma legenda ao conteudo que sera salvo no arquivo de texto
conteudo += "\n################################################ - Legenda - #################################################\n"
conteudo += "Verdadeiro Positivo (VP) : faces reconhecidas (abaixo do limiar) corretamente. O algoritmo reconheceu e realmente eh o sujeito."
conteudo += "\nVerdadeiro Negativo (VN) : faces reconhecidas (abaixo do limiar) incorretamente. O algoritmo reconheceu porem nao eh o sujeito correto."
conteudo += "\nFalso Positivo (FP) : faces nao reconhecidas (acima do limiar) corretamente. Realmente nao eram para ser reconhecidas, ou porque o sujeito nao esta na base de treinamento ou porque nao corresponde a face de um sujeito."
conteudo += "\nFalso Negativo (FN) : faces nao reconhecidas (acima do limiar) incorretamente. Correspondem a faces de sujeitos encontrados no treinamento, porem o algoritmo nao conseguiu reconhecer."
conteudo += "\n##############################################################################################################\n"

# Cria o cabecalho da tabela dos resultados
conteudo += "\nLIMIAR;VP;VN;FP;FN\n";

## AQUI COMECA O LOOP QUE VAI DO LIMIAR MIN ATE O MAX DE ACORDO COM O PASSO ##

# limiarMax recebe +1 para poder incluir o proprio valor do limiarMax no loop (ex. limiar <= limiarMax ou limiar < limiarMax+1)
for limiar in range(limiarMin, limiarMax+1, passo):

	# Mostra o limiar atual, apenas para saber em que passo esta o teste
	print "Limiar : " + str(limiar)

	# Cria o objeto do algoritmo de reconhecimento facial
	if algoritmo.lower() == "eigenface" or algoritmo.lower() == "eigenfaces":
		algReconhecimento = cv2.face.createEigenFaceRecognizer()
	elif algoritmo.lower() == "fisherface" or algoritmo.lower() == "fisherfaces":
		algReconhecimento = cv2.face.createFisherFaceRecognizer()
	elif algoritmo.lower() == "lbph" or algoritmo.lower() == "local binary pattern histogram":
		algReconhecimento = cv2.face.createLBPHFaceRecognizer()

	# Declara as variaveis que irao armazenar os resultados
	VP = 0
	VN = 0
	FP = 0
	FN = 0

	# Vetores que irao armazenar as imagens e o titulo das imagens para utilizar no treinamento
	imagens = []
	titulos = []

	# Obtem o nome de todos os arquivos encontrados na pasta de treinamento
	listaDirTreinamento = os.listdir(caminhoParaPastaTreinamento)

	# Laco responsavel por carregar todas as imagens para treinamento
	for index in xrange(0, len(listaDirTreinamento)):
		# Obtem o caminho para cada imagem da pasta
		caminhoDaImagemTreinamento = caminhoParaPastaTreinamento + listaDirTreinamento[index]

		# Extrai o numero correspondente ao sujeito
		numSujeito = int(listaDirTreinamento[index].split("_")[1])

		# Le a imagem e carrega ela na variavel imagem
		imagem = cv2.imread(caminhoDaImagemTreinamento)
		# Transforma a imagem para escala de cinza
		imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
		# Redimensiona a imagem para ficar com tamanho padrao 100x100
		imagem = cv2.resize(imagem, (100,100), interpolation = cv2.INTER_CUBIC)

		# Adiciona a imagem tratada ao vetor imagens
		imagens.append( imagem )
		# Adiciona um titulo para a imagem que corresponde ao numero do sujeito
		titulos.append( numSujeito )

	# Treina o algoritmo de reconhecimento facial
	algReconhecimento.train(imagens, np.array(titulos))

	# Obtem o nome de todos os arquivos encontrados na pasta de teste
	listaDirTeste = os.listdir(caminhoParaPastaTeste)

	# Laco responsavel por passar por todas as imagens na pasta de teste
	for index in xrange(0, len(listaDirTeste)):
		# Obtem o caminho para cada imagem da pasta
		caminhoDaImagemTeste = caminhoParaPastaTeste + listaDirTeste[index]
		
		# Usado para identificar se alguma imagem esta com o nome incorreto de acordo com o padrao esperado
		#print caminhoDaImagemTeste

		# Extrai o numero correspondente ao sujeito esperado
		sujeitoEsperado = listaDirTeste[index].split("_")[1]
		sujeitoEsperado = int(sujeitoEsperado.split(".")[0])
		
		# Carrega a imagem na variavel imagem
		imagem = cv2.imread(caminhoDaImagemTeste)
		# Transforma a imagem para escala de cinza
		imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
		# Redimensiona a imagem para ficar com tamanho padrao 100x100
		imagem = cv2.resize(imagem, (100,100), interpolation = cv2.INTER_CUBIC)

		# Realiza o reconhecimento facial
		sujeitoCLassificado, confianca = algReconhecimento.predict( imagem )

		# Se a confianca for menor ou igual ao limiar atual significa que o algoritmo reconheceu o sujeito
		if confianca <= limiar:
			# Se o algoritmo classificou o sujeito igual ao sujeito esperado esta correto (verdadeiro positivo)
			if sujeitoCLassificado == sujeitoEsperado:
				VP += 1
			# Caso contrario, se o algoritmo classificou um sujeito diferente esta incorreto (verdadeiro negativo)
			else:
				VN += 1
		# Se a confianca for maior que o limiar atual significa que o algoritmo nao reconheceu o sujeito
		else:
			# Se o sujeito esperado eh igual a -1, significa que nao foi possivel definir visivelmente quem era o sujeito, 
			# ou realmente nao corresponde a face de um sujeito. Neste caso o algoritmo esta de certa forma correto ao nao identificar (falso positivo)
			if sujeitoEsperado == -1:
				FP += 1
			# Caso contrario, se o sujeito esperado eh um sujeito valido e o algoritmo nao conseguiu identificar entao eh falso negativo
			else:
				FN += 1

	conteudo += str(limiar) + ";" + str(VP) + ";" + str(VN) + ";" + str(FP) + ";" + str(FN) + "\n";

# Registra a hora que o teste terminou em uma variavel
horaTermino = "\nHora de Termino : " + time.strftime("%d/%m/%Y %H:%M:%S") + "\n"

# Adiciona a hora de inicio e hora de termino dos testes 
conteudo = horaInicio + horaTermino + conteudo

# Mostra a hora de termino do teste - apenas para saber o tempo estimado
print horaTermino

# Salva o arquivo de texto
arquivoTexto = open(nomeArquivo + ".txt", "w")
arquivoTexto.write(conteudo)
arquivoTexto.close()

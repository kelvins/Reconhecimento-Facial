
import FaceRecognition as fr
from FaceRecognition import Algorithms

# Defines the path to the training folder
trainPath = "/home/kelvin/Desktop/FaceRecognition/TestesCurvaROC/BASE1/"

# Defines the path to the test folder
testPath  = "/home/kelvin/Desktop/FaceRecognition/TestesCurvaROC/TESTE1/"

newFaceRec = fr.FaceRecognition( Algorithms.EIGENFACE )

newFaceRec.setInterpolation( Interpolation.INTER_CUBIC )
newFaceRec.setDefaultSize( 100, 100 )
newFaceRec.train( trainPath )
newFaceRec.predict( testPath )
newFaceRec.showResults()
newFaceRec.save()
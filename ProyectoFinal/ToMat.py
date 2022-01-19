import numpy as np
import time
import glob #para leer los archivos
import matplotlib as plt
from matplotlib import image
import scipy.io as sciOutput
from scipy.io import loadmat


#guarda una serie de imagenes .jpg en una lista
def read_paths(path):
    auxPath = path + "/*.jpg"
    imgsList = glob.glob(auxPath)

    return imgsList


#convierte las imagenes de una lista dada en vectores
def process_img_list(X, y, listImages, size, typeIndex):
    for img in listImages:
        processError = False
        imgPixels = plt.image.imread(img)
        imgValues = []

        #convertimos los valores rgb de cada pixel en un int
        for i in range(size):
            for j in range(size):
                try:
                    rgb = imgPixels[i, j]
                    rgbInt = ((rgb[0]&0x0ff)<<16) | ((rgb[1]&0x0ff)<<8) | (rgb[2]&0x0ff)
                    imgValues.append(rgbInt)
                except:
                    processError = True
            
        #solo las incluimos si se pudieron transformar
        if(not processError):
            X.append(imgValues)
            y.append(typeIndex)


#transforma una serie de imagenes .jpg del mismo tamaño en un archivo .mat
def images_to_mat(path, matName, size):
    imgType = ["CAPUCHINBIRD",
            "COCK OF THE ROCK",
            "FRIGATE",
            "GANG GANG COCKATOO",
            "GO AWAY BIRD",
            "IWI",
            "TIT MOUSE",
            "UMBRELLA BIRD"]
    
    totalImgs = 0

    matX = []
    matY = []
    matXval = []
    matYval = []
    matXtest = []
    matYtest = []
    
    #indices de cada tipo de imagen en el .mat
    y = np.array({0, 1, 2, 3, 4, 5, 6, 7}) #todo esto no es un array [1, 2, 3, etc.]

    #leemos las imagenes por tipo
    for type in imgType:
        dataPaths = [path + "/train/" + type, path + "/val/" + type, path + "/test/" + type]
        imgsTrain = read_paths(dataPaths[0])
        imgsVal = read_paths(dataPaths[1])
        imgsTest = read_paths(dataPaths[2])

        #procesamos las imagenes una a una
            #entrenamiento
        process_img_list(matX, matY, imgsTrain, size, imgType.index(type))
        # matX = np.vstack([matX, auxX]) if matX.size else auxX
        # matY = np.vstack([matY, auxY]) if matY.size else auxY
            #validacion
        process_img_list(matXval, matYval, imgsVal, size, imgType.index(type))
        # matXval = np.vstack([matXval, auxX]) if matXval.size else auxX
        # matYval = np.vstack([matYval, auxY]) if matYval.size else auxY
            #test
        process_img_list(matXtest, matYtest, imgsTest, size, imgType.index(type))
        # matXtest = np.vstack([matXtest, auxX]) if matXtest.size else auxX
        # matYtest = np.vstack([matYtest, auxY]) if matYtest.size else auxY

    dictionary = {
        "X": matX,
        "y": matY,
        "Xval": matXval,

        "yval": matYval,
        "Xtest": matXtest,
        "ytest": matYtest
    }

    sciOutput.savemat(matName, dictionary)
    print(f"----- Archivo .mat guardado como {matName}")  

        
def main():
    #medimos el tiempo que tarda en crearse el .mat
    tic = time.time()

    #img size_ 224x224x3
    images_to_mat("data", "birdData.mat", 224)

    toc = time.time()
    tTotal = toc - tic
    print(f"----- ToMat.py: Tiempo tardado en crear el .mat: {tTotal // 60} min {str(tTotal % 60)[:5]} s")

main()

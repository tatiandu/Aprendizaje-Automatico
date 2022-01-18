import numpy as np
import os

def to_mat(path, matName):
    species = ["CAPUCHINBIRD", "COCK OF THE ROCK", "FRIGATE", "GANG GANG COCKATOO", "GO AWAY BIRD", "IWI", "TIT MOUSE", "UMBRELLA BIRD"]
    
    trainingMatX = []
    trainingMatY = []
    
    validationMatX = []
    validationMatY = []
    
    testMatX = []
    testMatY = []
    
    y = np.array({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) #???
    
    for speciesPath in species:
        files = []
        for r, d, f, in os.walk(path + speciesPath):
            for file in f:
                if '.jpg' in file:
                    files.append(file)
        for f in files:
            os.system(("csl")) #!???
            
            img = Image.open(path + speciesPath + f)
            values = []
            pix = img.load()
            for i in range(20): #Cambiar
                for j in range(20):
                    rgb = pix[i, j]
                    try:
                        rgbInteger = (int)(("%02x%02x%02x"%rgb), 16) #porque %03x%02x%02y sería descabellado
                        values.append(rgbInteger)
            
            aux = paths.index(concretePath)
        
def main():
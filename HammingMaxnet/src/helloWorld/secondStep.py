import cv2
import os, random
import numpy as np
from matplotlib import pyplot as plt
from _sqlite3 import Row
from numpy import dtype, double, float32
#hamming network matrix

mat = np.fromfile("matrix", dtype = double,count=-1)
print(mat.shape)
np.reshape(mat,(9,10000))
res = [[],[],[],[],[],[],[],[],[]]
wrong=[0,0,0,0,0,0,0,0,0]
#test image - grey scale

fdogs = r"C:\Users\David-tower\workspace_eclipse\HelloWorldP\src\helloWorld\New folder (2)\dogs"
fcats = r"C:\Users\David-tower\workspace_eclipse\HelloWorldP\src\helloWorld\New folder (2)\cats"
fcows = r"C:\Users\David-tower\workspace_eclipse\HelloWorldP\src\helloWorld\New folder (2)\cows"
ffighter = r"C:\Users\David-tower\workspace_eclipse\HelloWorldP\src\helloWorld\New folder (2)\fighters"
fjets = r"C:\Users\David-tower\workspace_eclipse\HelloWorldP\src\helloWorld\New folder (2)\jets"
frocks = r"C:\Users\David-tower\workspace_eclipse\HelloWorldP\src\helloWorld\New folder (2)\rocks"
fwhales = r"C:\Users\David-tower\workspace_eclipse\HelloWorldP\src\helloWorld\New folder (2)\whales"
fbiplanes = r"C:\Users\David-tower\workspace_eclipse\HelloWorldP\src\helloWorld\New folder (2)\biplanes"
fdolphin = r"C:\Users\David-tower\workspace_eclipse\HelloWorldP\src\helloWorld\New folder (2)\dolphin"


for g in range(50):
    a = random.choice(os.listdir(fdogs))
    file = fdogs + '\\'+a
                #print(file)
    img = cv2.imread(file,0)
            # find edges
    edges = cv2.Canny(img,100,200)
            #resize to 100,100
                #print(file)
    res1= cv2.resize(edges,(100,100),interpolation = cv2.INTER_CUBIC)
            
            
    a = random.choice(os.listdir(fcats))
    file = fcats + '\\'+a
                #print(file)
    img = cv2.imread(file,0)
            # find edges
    edges = cv2.Canny(img,100,200)
            #resize to 100,100
                #print(file)
    res2= cv2.resize(edges,(100,100),interpolation = cv2.INTER_CUBIC)
                
    a = random.choice(os.listdir(fcows))
    file = fcows + '\\'+a
                #print(file)
    img = cv2.imread(file,0)
            # find edges
    edges = cv2.Canny(img,100,200)
            #resize to 100,100
                #print(file)
    res3= cv2.resize(edges,(100,100),interpolation = cv2.INTER_CUBIC)
            
    a = random.choice(os.listdir(frocks))
    file = frocks + '\\'+a
                #print(file)
    img = cv2.imread(file,0)
            # find edges
    edges = cv2.Canny(img,100,200)
            #resize to 100,100
                #print(file)
    res7= cv2.resize(edges,(100,100),interpolation = cv2.INTER_CUBIC)
                
    a = random.choice(os.listdir(fjets))
    file = fjets + '\\'+a
                #print(file)
    img = cv2.imread(file,0)
            # find edges
    edges = cv2.Canny(img,100,200)
            #resize to 100,100
                #print(file)
    res5= cv2.resize(edges,(100,100),interpolation = cv2.INTER_CUBIC)
                
    a = random.choice(os.listdir(ffighter))
    file = ffighter + '\\'+a
                #print(file)
    img = cv2.imread(file,0)
            # find edges
    edges = cv2.Canny(img,100,200)
            #resize to 100,100
                #print(file)
    res4= cv2.resize(edges,(100,100),interpolation = cv2.INTER_CUBIC)
                
    a = random.choice(os.listdir(fbiplanes))
    file = fbiplanes + '\\'+a
                #print(file)
    img = cv2.imread(file,0)
            # find edges
    edges = cv2.Canny(img,100,200)
            #resize to 100,100
                #print(file)
    res6= cv2.resize(edges,(100,100),interpolation = cv2.INTER_CUBIC)
                
    a = random.choice(os.listdir(fwhales))
    file = fwhales + '\\'+a
                #print(file)
    img = cv2.imread(file,0)
            # find edges
    edges = cv2.Canny(img,100,200)
            #resize to 100,100
                #print(file)
    res8= cv2.resize(edges,(100,100),interpolation = cv2.INTER_CUBIC)
                
    a = random.choice(os.listdir(fdolphin))
    file = fdolphin + '\\'+a
                #print(file)
    img = cv2.imread(file,0)
            # find edges
    edges = cv2.Canny(img,100,200)
            #resize to 100,100
                #print(file)
    res9= cv2.resize(edges,(100,100),interpolation = cv2.INTER_CUBIC)
            #plt.subplot(121),plt.imshow(res,cmap = 'gray')
            #plt.title('Res Image'), plt.xticks([]), plt.yticks([])
            #plt.subplot(122),plt.imshow(edges,cmap = 'gray')
            #plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
            
            #plt.show()
            #turn it into a vector
    res[0]=res1.ravel()
    res[1]=res2.ravel()
    res[2]=res3.ravel()
    res[3]=res4.ravel()
    res[4]=res5.ravel()
    res[5]=res6.ravel()
    res[6]=res7.ravel()
    res[7]=res8.ravel()
    res[8]=res9.ravel()
    
    for t in range(9):
        for g in range(len(res[t])):
            if(res[t][g]==0):
                res[t][g] = -1.0
            else:
                res[t][g] = 1.0
            # applying image to matrix
    matnet=np.array([[1,-.1,-.1,-.1,-.1,-.1,-.1,-.1,-.1],[-.1,1,-.1,-.1,-.1,-.1,-.1,-.1,-.1],[-.1,-.1,1,-.1,-.1,-.1,-.1,-.1,-.1],[-.1,-.1,-.1,1,-.1,-.1,-.1,-.1,-.1],[-.1,-.1,-.1,-.1,1,-.1,-.1,-.1,-.1],[-.1,-.1,-.1,-.1,-.1,1,-.1,-.1,-.1],[-.1,-.1,-.1,-.1,-.1,-.1,1,-.1,-.1],[-.1,-.1,-.1,-.1,-.1,-.1,-.1,1,-.1],[-.1,-.1,-.1,-.1,-.1,-.1,-.1,-.1,1]])
            
    for i in range(9):
        mat2=mat.dot(res[i])
            
        mat2=mat2/2
        
        count = 9
        winner = 0
                    
        while count>1 :
            count=0
            mat2=matnet.dot(mat2)
            for x in range(0,9):
                if mat2[x] < 0:
                    mat2[x] = 0
                else:
                    count=count+1
                    winner=x 
                     
                   
        if winner != i :
            wrong[i] += 1


print((9*(50))-sum(wrong))/(9 *(50))
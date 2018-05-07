'''
Created on Apr 14, 2018

@author: David-tower
'''
import cv2
import os, random
import numpy as np
from matplotlib import pyplot as plt
from _sqlite3 import Row
from numpy import dtype, double
#hamming network matrix
mat = np.zeros(shape=(9,10000))



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

file = fdogs +'\\dog2.jpg'
img = cv2.imread(file,0)
edges = cv2.Canny(img,100,200)

res5= cv2.resize(edges,(100,100),interpolation = cv2.INTER_CUBIC)
res2=res5.ravel()

for row in range(len(res2)):
    if res2[row] > 0:
        mat[0][row]= 1.0
    else:
        mat[0][row]= -1.0
        

file = fcats +'\\KittenProgression-Darling-Week5.jpg'
img = cv2.imread(file,0)
edges = cv2.Canny(img,100,200)

res6= cv2.resize(edges,(100,100),interpolation = cv2.INTER_CUBIC)
res2=res6.ravel()

for row in range(len(res2)):
    if res2[row] > 0:
        mat[1][row]= 1.0
    else:
        mat[1][row]= -1.0

file = fcows +'\\cow.jpg'
img = cv2.imread(file,0)
edges = cv2.Canny(img,100,200)

res7= cv2.resize(edges,(100,100),interpolation = cv2.INTER_CUBIC)
res2=res7.ravel()

for row in range(len(res2)):
    if res2[row] > 0:
        mat[2][row]= 1.0
    else:
        mat[2][row]= -1.0
        
file = ffighter +'\\images.jpg'
img = cv2.imread(file,0)
edges = cv2.Canny(img,100,200)

res8= cv2.resize(edges,(100,100),interpolation = cv2.INTER_CUBIC)
res2=res8.ravel()

for row in range(len(res2)):
    if res2[row] > 0:
        mat[3][row]= 1.0
    else:
        mat[3][row]= -1.0
        
file = fjets +'\\71cPfeL6ASL._SX522_.jpg'
img = cv2.imread(file,0)
edges = cv2.Canny(img,100,200)

res9= cv2.resize(edges,(100,100),interpolation = cv2.INTER_CUBIC)
res2=res9.ravel()

for row in range(len(res2)):
    if res2[row] > 0:
        mat[4][row]= 1.0
    else:
        mat[4][row]= -1.0
        
file = fbiplanes +'\\IMG_5625.jpg'
img = cv2.imread(file,0)
edges = cv2.Canny(img,100,200)

res10= cv2.resize(edges,(100,100),interpolation = cv2.INTER_CUBIC)
res2=res10.ravel()

for row in range(len(res2)):
    if res2[row] > 0:
        mat[5][row]= 1.0
    else:
        mat[5][row]= -1.0
        
file = frocks +'\\images.jpg'
img = cv2.imread(file,0)
edges = cv2.Canny(img,100,200)

res11= cv2.resize(edges,(100,100),interpolation = cv2.INTER_CUBIC)
res2=res11.ravel()

for row in range(len(res2)):
    if res2[row] > 0:
        mat[6][row]= 1.0
    else:
        mat[6][row]= -1.0
        
file = fwhales +'\\64511.jpg'
img = cv2.imread(file,0)
edges = cv2.Canny(img,100,200)

res12= cv2.resize(edges,(100,100),interpolation = cv2.INTER_CUBIC)
res2=res12.ravel()

for row in range(len(res2)):
    if res2[row] > 0:
        mat[7][row]= 1.0
    else:
        mat[7][row]= -1.0

file = fdolphin +'\\images.jpg'
img = cv2.imread(file,0)
edges = cv2.Canny(img,100,200)

res12= cv2.resize(edges,(100,100),interpolation = cv2.INTER_CUBIC)
res2=res12.ravel()

for row in range(len(res2)):
    if res2[row] > 0:
        mat[8][row]= 1.0
    else:
        mat[8][row]= -1.0
      
#here 
print(mat)
mat3 = mat
res = [[],[],[],[],[],[],[],[],[]]
bpercent = 0
besttraining = 0
wrong=[0,0,0,0,0,0,0,0,0]

for o in range(60):
    for s in range(15):
        wrong=[0,0,0,0,0,0,0,0,0]
        mat = mat3
        for f in range(s):
            
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
        
        #maxnet
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
                    for c in range(10000):
                        mat[i][c]= mat[i][c]+.1*(res[i][c]-mat[i][c])
        
        
        
        
            
            #print(((9*(f+1))-sum(wrong))/(9 *(f+1)))
            if(bpercent < ((9*(f+1))-sum(wrong))/(9 *(f+1))):
                bpercent = ((9*(f+1))-sum(wrong))/(9 *(f+1))
                besttraining = s
                mat4 = mat
                
            #print("i exited correctly ")    
            if(bpercent>.5):
                break
        if(bpercent>.5):
            break
    if(bpercent>.5):
        break
            
print(bpercent,besttraining)
#mat4.tofile("matrix",' ', "%i") 

mat = mat4
print(mat)
wrong=[0,0,0,0,0,0,0,0,0]


for g in range(50):
    print(g)
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
        mat2=mat4.dot(res[i])
            
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


print((9*(50)-sum(wrong))/(9 *(50)))

print(sum(wrong))
print(wrong)


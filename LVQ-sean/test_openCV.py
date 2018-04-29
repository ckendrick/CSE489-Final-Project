from import_images import unpickle #unpacks cifar-10 images to dictionary
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#Exctracts 8-bit RGB image from arr of size nxm starting at i0
def extract_img(arr, n,m, i0):
    im = np.zeros((n,m,3),dtype=np.uint8)
    for i in range(3): 
        for j in range(n):
            im[j,:,i] = arr[i0,i*n*m + j*m : i*n*m + (j+1)*m]
    return im
                
        


ts1 = unpickle('./images/data_batch_1')["data"]
img = cv.imread('./images/elevator.jpg',cv.IMREAD_GRAYSCALE) #note that cv is in BGR format, plt.imshow expects RGB format



#drop an image from ts1 into appropriately sized container
img_n = 420
ts1_img = extract_img(ts1,32,32,img_n)
ts1_img = cv.cvtColor(ts1_img,cv.COLOR_RGB2GRAY) #change image to grayscale
#ts1_img = img #make this for comparison
ts1_img = cv.resize(ts1_img,(512,512),interpolation=0) #resize image
plt.imshow(ts1_img,interpolation="nearest",cmap=plt.get_cmap('gray'))
plt.show()
#cv.imshow('input',ts1_img)

#img2 = mpimg.imread('./images/elevator.jpg')
#cv.imshow('img', img)
#plt.imshow(img, cmap=plt.get_cmap('gray'))
#plt.show()

## Run SURF on image
#surf = cv.xfeatures2d.SURF_create(400) #input argument is Hessian Threshold
orb = cv.ORB_create(nfeatures=128)

#Find keypoints and descriptors
kp = orb.detect(ts1_img,None)

kp,des = orb.compute(ts1_img, kp)

ts1_img2 = cv.drawKeypoints(ts1_img,kp,None,color=(0,255,0),flags=0)
cv.imshow('keypoints',ts1_img2)
#plt.imshow(ts1_img2,interpolation="nearest")
#plt.show()
#img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0),flags=0)
#cv.imshow('img2', img2)
cv.imwrite('./im_out/ts1_'+str(img_n)+'.jpg',ts1_img2)
print(des.shape)


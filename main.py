#encoding: utf-8
import matplotlib
#matplotlib.use("WXAgg", warn=True)
import matplotlib.pyplot as plt
import skimage.io as io
import cv2
import numpy as np
import matplotlib.cm as cm
import scipy.signal as signal
from skimage import img_as_ubyte
"load image data"
Img_Original =  io.imread('C:/Users/a2310o/PycharmProjects/Skeletonization-by-Zhang-Suen-Thinning-Algorithm-master/data/name.jpg', as_grey=True)     # Gray image, rgb images need pre-conversion
"Convert gray images to binary images using Otsu's method"
from skimage.filters import threshold_otsu
Otsu_Threshold = threshold_otsu(Img_Original)
BW_Original = Img_Original < Otsu_Threshold    # must set object region as 1, background region as 0 !     < means black

def neighbours(x,y,image): #returm 8 numbers
    img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
    return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],     # P2,P3,P4,P5
                img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]    # P6,P7,P8,P9

def transitions(neighbours):
    "No. of 0,1 patterns (transitions from 0 to 1) in the ordered sequence"
    n = neighbours + neighbours[0:1]      # P2, P3, ... , P8, P9, P2      [0:1] P2
    return sum( (n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]) )  # (P2,P3), (P3,P4), ... , (P8,P9), (P9,P2) 0->1

def zhangSuen(image):
    "the Zhang-Suen Thinning Algorithm"
    Image_Thinned = image.copy()  # protect the original image
    changing1 = changing2 = 1        #  the points to be removed (set as 0)
    while changing1 or changing2:   #  iterates until no further changes occur in the image
        # Step 1
        changing1 = []
        rows, columns = Image_Thinned.shape
        for x in range(1, rows - 1):                     # No. of  rows #edge 不用掃
            for y in range(1, columns - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1     and    # Conditions:
                    2 <= sum(n) <= 6   and
                    transitions(n) == 1 and
                    P2 * P4 * P6 == 0  and
                    P4 * P6 * P8 == 0):
                    changing1.append((x,y))
        for x, y in changing1: 
            Image_Thinned[x][y] = 0   #delete
        # Step 2
        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1   and
                    2 <= sum(n) <= 6  and
                    transitions(n) == 1 and
                    P2 * P4 * P8 == 0 and
                    P2 * P6 * P8 == 0):
                    changing2.append((x,y))    
        for x, y in changing2: 
            Image_Thinned[x][y] = 0
    return Image_Thinned

"Apply the algorithm on images"
BW_Skeleton = zhangSuen(BW_Original)
# opencv :binary images   :  numpy.uint8   scikit-image numpy.float64
cv_image = img_as_ubyte(BW_Skeleton)
cv2.imwrite('./results/skel.jpg',cv_image)




#hist
img = cv2.imread('./data/name.jpg',0)

sobelx = cv2.Sobel(img,cv2.CV_32F,1,0,ksize=3) #Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]]) -> dst    #32bit float

sobely= cv2.Sobel(img,cv2.CV_32F,0,1,ksize=3)

phase=cv2.phase(sobelx,sobely,True,angleInDegrees=True)

p= np.arange(0,360,0.3515625)

plt.subplot(2,1,1)
plt.imshow(BW_Skeleton,cmap=cm.gray)
plt.subplot(2,1,2)
plt.hist(phase.flatten(),p)
plt.show()

## 스케치 효과를 주었을때 CT 사진 효과
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math 
kernel_size = 3
def sketch(image):
    height , width = image.shape[ :2]
    imageC = image.copy()
    for y in range(height):
        for x in range(width):
            divK = kernel_size//2
            if x-divK<0 or y-divK <0 or x+divK+1>width or y+divK+1>height:
                lis= []
                if x-divK<0:
                    dx = abs(x-divK)
                    for i in range(dx):
                        lis.append(imageC[y, x])    
                if y-divK<0:
                    dy = abs(y-divK)
                    for i in range(dy):
                        lis.append(imageC[y, x]) 
                if x+divK+1>width:
                    dx = x+divK+1-width
                    for i in range(dx):
                        lis.append(imageC[y, x]) 
                if y+divK+1>height:
                    dy = y+divK+1-height
                    for i in range(dy):
                        lis.append(imageC[y, x]) 
                for los1 in (-divK, divK+1, 1):
                    for los2 in (-divK, divK+1, 1):
                        try:
                            lis.append(imageC[y-los1, x-los2])
                        except:
                            continue
                
                kernel = lis
            else:
                lis = []
                for los1 in (-divK, divK+1, 1):
                    for los2 in (-divK, divK+1, 1):
                        try:
                            lis.append(imageC[y-los1, x-los2])
                        except:
                            continue
                kernel = lis
            maximum = maximumFilter(kernel)
            image[y, x] = 255*imageC[y, x]/maximum
                    
    return image


def maximumFilter(kernel):
    return max(kernel)
    

image = cv2.imread("./testBackground.jpeg")
dimension = image.shape[2]
if dimension == 1:
    pass
elif dimension ==3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
elif dimension ==4:
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

image2 = sketch(image.copy())

image3 = cv2.GaussianBlur(image2.copy(), (5,5), 0)

ret, image4 = cv2.threshold(image2, 0, 255, cv2.THRESH_TOZERO+cv2.THRESH_OTSU)

kernel = np.ones((3, 3), dtype = "uint8" )
image5 = cv2.dilate(image4, kernel, iterations=1)
image5 = cv2.erode(image5, kernel, iterations=1)

ret, image6 = cv2.threshold(image2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)



plt.figure(figsize=(15,6))
plt.subplot(2,3,1)
plt.imshow(image, cmap = "gray")
plt.subplot(2,3,2)
plt.imshow(image2, cmap = "gray")
plt.subplot(2,3,3)
plt.imshow(image3, cmap = "gray")
plt.subplot(2,3,4)
plt.imshow(image4, cmap = "gray")
plt.subplot(2,3,5)
plt.imshow(image5, cmap = "gray")
plt.subplot(2,3,6)
plt.imshow(image6, cmap = "gray")
plt.show()
cv2.imwrite("1.jpeg", image)
cv2.imwrite("2.jpeg", image2)
cv2.imwrite("3.jpeg", image3)
cv2.imwrite("4.jpeg", image4)
cv2.imwrite("5.jpeg", image5)
cv2.imwrite("6.jpeg", image6)



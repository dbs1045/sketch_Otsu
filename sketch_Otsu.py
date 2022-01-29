## 스케치 효과를 주었을때 CT 사진 효과
import cv2
import numpy as np
from matplotlib import pyplot as plt
kernel_size = 5
def sketch(image):
    width , height = image.shape[ :2]
    imageC = image.copy()
    for x in range(width):
        for y in range(height):
            divK = kernel_size//2
            if x-divK<0 or y-divK <0 or x+divK+1>=width or y+divK+1>=height:
                randX= x-divK
                ranpX = x+divK+1
                randY = y-divK
                ranpY = y+divK+1
                if x-divK<0:
                    dx = abs(x-divK)
                    randX = randX+dx
                if y-divK<0:
                    dy = abs(y-divK)
                    randY = randY+dy
                if x+divK+1>=width:
                    dx = x+divK+1-width
                    ranpX = ranpX-dx
                if y+divK+1>=height:
                    dy = y+divK+1-height
                    ranpY = ranpY-dy
                kernel = np.reshape(imageC[randX:ranpX, randY:ranpY], ((ranpX-randX)*(ranpY-randY)))
            else:
                kernel = np.reshape(imageC[x-divK:x+divK+1, y-divK:y+divK+1], (kernel_size**2))
            maximum = maximumFilter(kernel)
            image[x, y] = 255*imageC[x, y]/maximum
                    
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
image5 = cv2.dilate(image4, kernel,iterations=1)
image5 = cv2.erode(image5, kernel, iterations=1)



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
plt.show()




#Aim: Write a program to Apply edge detection techniques such as Sobel and Canny to extract meaningful information from the given image samples

import numpy as np
from scipy import signal, misc, ndimage
from skimage import filters, feature, img_as_float
from skimage.io import imread 
from skimage.color import rgb2gray 
from PIL import Image, ImageFilter 
import matplotlib.pylab as pylab


def plot_image (image,title): 
    pylab.imshow(image)
    pylab.title(title,size=20)
    pylab.axis('off')
im=Image.open("../Downloads/test.jpg").convert('L') 
pylab.gray()
pylab.figure(figsize=(15,15))
pylab.subplot(3,2,1)
plot_image(im,'original') 
edges=filters.roberts(im) 
pylab.subplot(3,2,2)
plot_image(edges,'roberts') 
edges=filters.scharr(im) 
pylab.subplot(3,2,3)
plot_image(edges,'scharr') 
edges=filters.sobel(im)
pylab.subplot(3,2,4)
plot_image(edges,'sobel') 
edges=filters.prewitt(im) 
pylab.subplot(3,2,5)
plot_image(edges,'prewitt') 
edges=np.clip(filters.laplace(im),0,1) 
pylab.subplot(3,2,6)
plot_image(edges,'laplace') 
pylab.subplots_adjust(wspace=0.1,hspace=0.1) 
pylab.show()


im=Image.open('../Downloads/test.jpg').convert('L')
pylab.gray()
pylab.figure(figsize=(15,15)) 
pylab.subplot(2,2,1) 
plot_image(im,'original') 
pylab.subplot(2,2,2) 
edges_x=filters.sobel_h(im)
plot_image(np.clip(edges_x,0,1),'sobel_x') 
pylab.subplot(2,2,3)
edges_y=filters.sobel_v(im)
plot_image(np.clip(edges_y,0,1),'sobel_y') 
pylab.subplot(2,2,4) 
edges=filters.sobel(im)
plot_image(np.clip(edges,0,1),'sobel')
pylab.subplots_adjust(wspace=0.1,hspace=0.1)
pylab.show()


import numpy as np
import matplotlib.pyplot as plt
 
from scipy import ndimage as ndi
from skimage.util import random_noise
from skimage import feature #Generate noisy image of a square 
image=np.zeros((128,128),dtype=float) 
image[32:-32,32:-32]=1
image=ndi.rotate(image,15,mode='constant') 
image=ndi.gaussian_filter(image,4) 
image=random_noise(image,mode='speckle',mean=0.05) #ComputetheCannyfilterfortwovaluesofsigma 
edges1=feature.canny(image) 
edges2=feature.canny(image,sigma=3)
#displayresults 
fig,ax=plt.subplots(nrows=1,ncols=3,figsize=(8,3)) 
ax[0].imshow(image,cmap='gray') 
ax[0].set_title('noisyimage',fontsize=10) 
ax[1].imshow(edges1,cmap='gray') 
ax[1].set_title(r'Cannyfilter,$\sigma=1$',fontsize=10)
ax[2].imshow(edges2,cmap='gray') 
ax[2].set_title(r'Cannyfilter,$\sigma=3$',fontsize=10) 
for a in ax:
    a.axis('off') 
fig.tight_layout()
plt.show()

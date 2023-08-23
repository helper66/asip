#Aim: Write program to implement point/pixel intensity transformations such as
#1.Log and Power-law transformations
#2.Contrast adjustments
#3.Histogram equalization
#4.Thresholding, and halftoning operations


#Importing libraries
import numpy as np
from skimage import data,img_as_float,img_as_ubyte,exposure,io,color
from skimage.io import imread
from skimage.exposure import cumulative_distribution
from skimage.restoration import denoise_bilateral,denoise_nl_means,estimate_sigma
#from skimage.measure import compare_psnr 
from skimage.util import random_noise 
from skimage.color import rgb2gray
from PIL import Image,ImageEnhance,ImageFilter
from scipy import ndimage,misc
import matplotlib.pylab as pylab


#load image and plot histogram 
#function to display image 
def plot_image (image,title = "" ):
    pylab.title(title,size=20)
    pylab.imshow(image) 
    pylab.axis('off')
#function to display histogram
def plot_hist (r,g,b,title =''):
    r,g,b = img_as_ubyte(r)
    img_as_ubyte(g)
    img_as_ubyte(b) 
    pylab.hist(np.array(r).ravel(),bins=256, range=(0,256),color='r',alpha=0.5) 
    pylab.hist(np.array(g).ravel(),bins=256,range=(0,256),color='g',alpha=0.5) 
    pylab.hist(np.array(b).ravel(),bins=256,range=(0,256),color='b',alpha=0.5) 
    pylab.xlabel('PixelValues',size=10)
    pylab.ylabel('Frequency',size=10)
    pylab.title(title,size=10)
im=Image.open("../Downloads/test.jpg") 
im_r,im_g,im_b=im.split() 
pylab.style.use('ggplot') 
pylab.figure(figsize=(15,5)) 
pylab.subplot(121)
plot_image(im,'OriginalImage')
pylab.subplot(122)
plot_hist(im_r,im_g,im_b,'HistpgramofRGBImage') 
pylab.show()


#Log and Power-law transformations 
#log transformation
im=im.point(lambda i:255*np.log(1+i/255)) 
im_r,im_g,im_b=im.split() 
pylab.style.use('ggplot') 
pylab.figure(figsize=(15,5))
pylab.subplot(121)
plot_image(im,'ImageafterLogTransformation') 
pylab.subplot(122)
plot_hist(im_r,im_g,im_b,'HistogramofLogtransformforRGBchannel') 
pylab.show()


#power law transform 
im=img_as_float(imread("../Downloads/test.jpg")) 
#im_r, im_g, im_b = im.split()
gamma=3.5 
im1=im**gamma 
pylab.style.use('ggplot') 
pylab.figure(figsize=(10,5))
pylab.subplot(121)
plot_hist(im[...,0],im[...,1],im[...,2],'HistogramforRGBchannel(Input)')
pylab.subplot(122)
plot_hist(im1[...,0],im1[...,1],im1[...,2],'Histogramfor,RGBOutput')
pylab.show()
pylab.figure(figsize=(10,5))
pylab.subplot(121)
plot_image(im,'Imageoriginal') 
pylab.subplot(122)
plot_image(im1,'Output') 
pylab.show()


#constrast streching
im=Image.open("../Downloads/test.jpg") 
im_r,im_g,im_b=im.split() 
pylab.style.use('ggplot')
pylab.figure(figsize=(15,5)) 
pylab.subplot(121) 
plot_image(im)
pylab.subplot(122) 
plot_hist(im_r,im_g,im_b) 
pylab.show()


#contrast
def contrast (c):
    return 0 if c<50 else (255 if c>150 else(255*c-22950)/48) 
im1=im.point(contrast)
im_r,im_g,im_b=im1.split() 
pylab.style.use('ggplot') 
pylab.figure(figsize=(15,5))
pylab.subplot(121) 
plot_image(im1) 
pylab.subplot(122) 
plot_hist(im_r,im_g,im_b) 
pylab.yscale('log',basey=10)
pylab.show()


#thresholding operations 
im=Image.open("../Downloads/test.jpg").convert('L')
pylab.hist(np.array(im).ravel(),bins=256,range=(0,256),color='g') 
pylab.xlabel('Pixelvalues')
pylab.ylabel('Frequency')
pylab.title('HistogramofPixelvalues')
pylab.show() 
pylab.figure(figsize=(12,18))
pylab.gray()
pylab.subplot(221)
plot_image(im,'OriginalImage') 
pylab.axis('off')
th=[0,50,100,150,200]
for i in range (2,5):
    im1=im.point(lambda x:x>th[i]) 
pylab.subplot(2,2,i)
plot_image(im1,'binaryImagewiththreshold='+str(th[i])) 
pylab.show()

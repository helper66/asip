#Aim: Write program to demonstrate the following aspects of signal processing on suitable data
#1. Upsampling and downsampling on Image/speech signal
#2. Fast Fourier Transform to compute DFT


#importing libraries
from PIL import Image
from skimage.io import imread, imshow, show
import scipy.fftpack as fp
from scipy import ndimage, misc, signal 
from skimage import data, img_as_float 
from skimage.color import rgb2gray 
from skimage.transform import rescale 
import matplotlib.pylab as pylab
import numpy as np 
import numpy.fft 
import timeit


im=Image.open("../Downloads/test.jpg")
pylab.imshow(im)
pylab.show()


#upsampling nearest 
im1=im.resize((im.width*5,im.height*5), Image.NEAREST) 
pylab.figure(figsize=(7,7))
pylab.imshow(im1)
pylab.show()


#upsampling bi-linear
im1=im.resize((im.width*5,im.height*5), Image.BILINEAR) 
pylab.figure(figsize=(8,8))
pylab.imshow(im1)
pylab.show()


#upsampling bi-cubic 
im1=im.resize((im.width*5,im.height*5), Image.BICUBIC) 
pylab.figure(figsize=(9,9))
pylab.imshow(im1)
pylab.show()

#downsampling
im=Image.open("../Downloads/test.jpg") 
im=im.resize((im.width//5,im.height//5))
pylab.figure(figsize=(10,5))
pylab.imshow(im)
pylab.show()


im=im.resize((im.width//5,im.height//5), Image.ANTIALIAS) 
pylab.figure(figsize=(10,5))
pylab.imshow(im)
pylab.show()


im=imread('../Downloads/test.jpg') 
im1=im.copy()
pylab.figure(figsize=(10,10))
for i in range(6):
    pylab.subplot(2,3,i+1)
    pylab.imshow(im1,cmap='Spectral_r')
    pylab.axis('on')
    pylab.title('Image size= '+str(im1.shape[1])+'x'+str(im1.shape[0])) 
    im1=rescale(im1,scale=0.5,multichannel=True, anti_aliasing=True) 
pylab.subplots_adjust(wspace=0.3, hspace=0.3)
pylab.show()


im=Image.open("../Downloads/test.jpg")
def signaltonoise(a,axis=0,ddof=0):
    a=np.asanyarray(a) 
    n=a.mean(axis) 
    sd=a.std(axis=axis,ddof=ddof) 
    return np.where (sd==0, 0, n/sd) 
pylab.figure(figsize=(10,10))
num_colors_list=[1<<n for n in range (8,0,-1)] 
snr_list=[]
i=1
for num_colors in num_colors_list: 
    im1=im.convert('P',palette=Image.ADAPTIVE, colors=num_colors) 
    pylab.subplot(4,2,i),pylab.imshow(im1)
    pylab.axis('off') 
    snr_list.append(signaltonoise(im1,axis=None))
    pylab.title('Image with $ colors = '+str(num_colors)+ 'SNR=' + str(np.round(snr_list[i- 1],3)),size=10)
    i+=1
pylab.subplots_adjust(wspace=0.2, hspace=0.2) 
pylab.show()


pylab.plot(num_colors_list,snr_list,'r.-')
pylab.xlabel('#colorsintheimage') 
pylab.ylabel('SNR') 
pylab.title('ChangeinSNRw.r.t.#colors') 
pylab.xscale('log',base=2) 
pylab.gca().invert_xaxis() 
pylab.show()


# Fast Fourier Transform to compute DFT 
# FFT Operations
im=np.array(Image.open("../Downloads/test.jpg").convert('L')) 
snr=signaltonoise(im, axis=None)
print("SNR for the original Image =" +str(snr)) 
freq=fp.fft2(im)
im1=fp.ifft2(freq).real 
snr=signaltonoise(im1,axis=None) 
print('SNR for the original Image =' +str(snr)) 
assert(np.allclose(im,im1)) 
pylab.figure(figsize=(10,10))
pylab.subplot(121)
pylab.imshow(im, cmap='gray')
pylab.axis('on') 
pylab.title('original image' ,size=10)
pylab.subplot(122)
pylab.imshow(im, cmap='gray')
pylab.axis('on') 
pylab.title('reconstructed image' ,size=10)


im=np.array(Image.open("../Downloads/test.jpg").convert('L')) 
snr=signaltonoise(im, axis=None)
print("SNR for the original Image =" +str(snr)) 
freq=fp.fft2(im)
im1=fp.ifft2(freq).real 
snr=signaltonoise(im1,axis=None) 
print('SNR for the original Image =' +str(snr)) 
assert(np.allclose(im,im1)) 
pylab.figure(figsize=(10,10))
pylab.subplot(121)
pylab.imshow(im, cmap='twilight_shifted')
pylab.axis('on') 
pylab.title('original image' ,size=10)
pylab.subplot(122)
pylab.imshow(im, cmap='twilight_shifted_r')
pylab.axis('on') 
pylab.title('reconstructed image' ,size=10)


freq2=fp.fftshift(freq) 
pylab.figure(figsize=(5,5))
pylab.imshow(20*np.log10(0.1+freq2).astype(int))
pylab.show()


#FFT with numpy
import numpy.fft as fp 
im1=rgb2gray(imread("../Downloads/test.jpg")) 
pylab.figure(figsize=(12,10))
freq1=fp.fft2(im1) 
im1=fp.ifft2(im1).real
pylab.subplot(3,2,1)
pylab.imshow(im1, cmap='gray') 
pylab.title('Original Image', size=10)
pylab.subplot(3,2,2)
pylab.imshow(20*np.log10(0.01 + np.abs(fp.fftshift(freq1)))) 
pylab.title('FFT spectrum Magnitude', size=10)
pylab.subplot(3,2,3) 
pylab.imshow(np.angle(fp.fftshift(freq1)), cmap='gray') 
pylab.title('FFT Phase' , size=10)
pylab.subplot(3,2,4)
pylab.imshow(np.clip(im,0,255), cmap='gray') 
pylab.title('reconstructed Image', size=10)
pylab.show()

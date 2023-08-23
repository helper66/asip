#Aim: Write the program to implement various morphological image processing techniques.


#Importing Libraries
from skimage.io import imread
import numpy as np
from skimage.color import rgb2gray
import matplotlib.pylab as pylab
from skimage.morphology import binary_erosion,rectangle


#erosion
def plot_image(image,title=''):
    pylab.title(title,size=20)
    pylab.imshow(image)
    pylab.axis('off') #comment this line if you want axis ticks 
im=rgb2gray(imread('../Downloads/test.jpg')) 
im[im<=0.5]=0
#create binary image with fixed threshold 0.5
im[im>0.5]=1 
pylab.gray()
pylab.figure(figsize=(20,10)) 
pylab.subplot(1,3,1)
plot_image(im,'original') 
im1=binary_erosion(im,rectangle(1,5))
pylab.subplot(1,3,2)
plot_image(im1,'erosionwithrectanglesize(1,5)') 
im1=binary_erosion(im,rectangle(1,15)) 
pylab.subplot(1,3,3)
plot_image(im1,'erosionwithrectanglesize(1,15)')
pylab.show()


#Dialation
from skimage.morphology import binary_dilation,disk
from skimage import img_as_float
im=img_as_float(imread('../Downloads/test.jpg')) 
im=1-im[...,2]
im[im<=0.5]=0 
im[im>0.5]=1 
pylab.gray()
pylab.figure(figsize=(18,9)) 
pylab.subplot(131)
pylab.imshow(im)
pylab.title('original',size=20) 
pylab.axis('off')
for d in range(1,3):
    pylab.subplot(1,3,d+1) 
    im1=binary_dilation(im,disk(2*d)) 
    pylab.imshow(im1)
    pylab.title('dilationwithdisksize'+str(2*d),size=20)
    pylab.axis('off')
pylab.show()


from skimage.morphology import binary_opening,binary_closing,binary_erosion,binary_dilation,disk 
im=rgb2gray(imread('../Downloads/test.jpg')) 
im[im<=0.5]=0
im[im>0.5]=1 
pylab.gray()
pylab.figure(figsize=(20,10)) 
pylab.subplot(1,3,1)
plot_image(im,'original') 
im1=binary_opening(im,disk(12)) 
pylab.subplot(1,3,2)
plot_image(im1,'openingwithdisksize'+str(12)) 
im1=binary_closing(im,disk(6)) 
pylab.subplot(1,3,3)
plot_image(im1,'closingwithdisksize'+str(6)) 
pylab.show()


def plot_images_horizontally(original,filtered,filter_name,sz=(10,3)): 
    pylab.gray()
    pylab.figure(figsize=sz) 
    pylab.subplot(1,2,1)
    plot_image(original,'original')
    pylab.subplot(1,2,2)
    plot_image(filtered,filter_name) 
    pylab.show()
from skimage.morphology import skeletonize 
im=img_as_float(imread('../Downloads/test.jpg')[...,2]) 
threshold=0.5
im[im<=threshold]=0 
im[im>threshold]=1 
skeleton=skeletonize(im)
plot_images_horizontally(im,skeleton,'skeleton',sz=(10,5))


from skimage.morphology import convex_hull_image 
im=rgb2gray(imread('../Downloads/test.jpg')) 
threshold=0.5
im[im<threshold]=0 
#converttobinaryimage 
im[im>=threshold]=1 
chull=convex_hull_image(im)
plot_images_horizontally(im,chull,'convexhull',sz=(18,9))


#Removing small objects
from skimage.morphology import remove_small_objects 
im=rgb2gray(imread('../Downloads/test.jpg')) 
im[im>0.5]=1
#create binary image by thresholding with fixed threshold 0.5
im[im<=0.5]=0 
im=im.astype(np.bool) 
pylab.figure(figsize=(15,15))
pylab.subplot(2,2,1)
plot_image(im,'original') 
i=2
for osz in[50,200,500]: 
    im1=remove_small_objects(im, osz, connectivity=1)
    pylab.subplot(2,2,i),plot_image(im1,'removing small objects below size'+str(osz)) 
    i+=1
pylab.show()


#Whiteandblacktop-hats
from skimage.morphology import white_tophat,black_tophat,square 
im=imread('../Downloads/test.jpg')[...,2] 
im[im<=0.5]=0
im[im>0.5]=1 
im1=white_tophat(im,square(5)) 
im2=black_tophat(im,square(5))
pylab.figure(figsize=(15,15))
pylab.subplot(1,2,1)
plot_image(im1,'whitetophat') 
pylab.subplot(1,2,2)
plot_image(im2,'blacktophat') 
pylab.show()


from skimage.morphology import binary_erosion 
im=rgb2gray(imread('../Downloads/test.jpg')) 
threshold=0.5
im[im<threshold]=0
im[im>=threshold]=1 
boundary=im - binary_erosion(im)
plot_images_horizontally(im,boundary,'boundary',sz=(18,9))


#Fingerprintcleaningwithopeningandclosing 
im=rgb2gray(imread('../Downloads/test.jpg')) 
im[im<=0.5]=0

#binarize
im[im>0.5]=1 
im_o=binary_opening(im,square(2)) 
im_c=binary_closing(im,square(2))
im_oc=binary_closing(binary_opening(im,square(2)),square(2)) 
pylab.figure(figsize=(15,15)) 
pylab.subplot(221)
plot_image(im,'original') 
pylab.subplot(222)
plot_image(im_o,'opening') 
pylab.subplot(223)
plot_image(im_c,'closing') 
pylab.subplot(224)
plot_image(im_oc,'opening+closing') 
pylab.show()


#Grayscale operations
from skimage.morphology import dilation,erosion,closing,opening,square
im=imread('../Downloads/test.jpg') 
im=rgb2gray(im)
struct_elem=square(5) 
eroded=erosion(im,struct_elem) 
plot_images_horizontally(im,eroded,'erosion')


dilated=dilation(im,struct_elem) 
plot_images_horizontally(im,dilated,'dilation')


opened=opening(im,struct_elem) 
plot_images_horizontally(im,opened,'opening')


closed=closing(im,struct_elem) 
plot_images_horizontally(im,closed,'closing')


#Morphological contrast enhancement
from skimage.filters.rank import enhance_contrast
from skimage import exposure
def plot_gray_image(ax,image,title): 
    ax.imshow(image,cmap=pylab.cm.gray)
    ax.set_title(title)
    ax.axis('off') 
    ax.set_adjustable('box')

image=rgb2gray(imread('../Downloads/test.jpg')) 
sigma=0.05 
noisy_image=np.clip(image+sigma*np.random.standard_normal(image.shape),0,1) 
enhanced_image=enhance_contrast(noisy_image,disk(5)) 
equalized_image=exposure.equalize_adapthist(noisy_image) 
fig,axes=pylab.subplots(1,3,figsize=[18,7],sharex='row',sharey='row') 
axes1,axes2,axes3=axes.ravel()
plot_gray_image(axes1,noisy_image,'Original') 
plot_gray_image(axes2,enhanced_image,'Localmorphologicalcontrast enhancement') 
plot_gray_image(axes3,equalized_image,'AdaptiveHistogramequalization')


#Filling holes inbinary objects
from scipy.ndimage.morphology import binary_fill_holes
im =rgb2gray(imread('../Downloads/test.jpg')) 
im[im<=0.5]=0
im[im>0.5]=1 
pylab.figure(figsize=(15,15))
pylab.subplot(221)
pylab.imshow(im)
pylab.title('original',size=20)
pylab.axis('off')
i=2
for n in [3,5,7]: 
    pylab.subplot(2,2,i)
    im1=binary_fill_holes(im,structure=np.ones((n,n))) 
    pylab.imshow(im1)
    pylab.title('binary_fill_holes with structure square side'+str(n),size=15) 
    pylab.axis('off') 
    i+=1
pylab.show()


#Noiseremovalwiththemedianfilter
from skimage.filters.rank import median
from skimage.morphology import disk 
noisy_image=(rgb2gray(imread('../Downloads/test.jpg'))*255).astype(np.uint8) 
noise=np.random.random(noisy_image.shape) 
noisy_image[noise>0.9]=255
noisy_image[noise<0.1]=0 
fig,axes=pylab.subplots(2,2,figsize=(10,10),sharex=True,sharey=True) 
axes1,axes2,axes3,axes4=axes.ravel() 
plot_gray_image(axes1,noisy_image,'Noisyimage') 
plot_gray_image(axes2,median(noisy_image,disk(1)),'Median$r=1$')
plot_gray_image(axes3,median(noisy_image,disk(5)),'Median$r=5$')
plot_gray_image(axes4,median(noisy_image,disk(20)),'Median$r=20$')



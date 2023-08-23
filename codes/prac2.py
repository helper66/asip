#Aim: Write program to demonstrate the following aspects of signal on sound/image data
#1. Convolution operation
#2. Template Matching


#Importing libraries
from skimage.io import imread,imshow,show 
from skimage.color import rgb2gray
import numpy as np
from scipy import ndimage,misc,signal
import matplotlib.pylab as pylab


#Convolution on grey and Color Images
im = rgb2gray(imread("../Downloads/test.jpg")).astype(float) 
print(np.max(im))
print(im.shape) 
blur_box_kernel=np.ones((3,3))/9
edge_laplace_kernel=np.array([[0,1,0],[1,-4,1],[0,1,0]]) 
im_blurred = signal.convolve2d(im,blur_box_kernel) 
im_edges=np.clip(signal.convolve2d(im,edge_laplace_kernel),0,1)
fig,axes=pylab.subplots(ncols=3,sharex=True,sharey=True,figsize=(18,6)) 
axes[0].imshow(im,cmap=pylab.cm.gray) 
axes[0].set_title('OriginalImage',size=20) 
axes[1].imshow(im_blurred,cmap=pylab.cm.gray) 
axes[1].set_title('BoxBlur',size=20) 
axes[2].imshow(im_edges,cmap=pylab.cm.gray) 
axes[2].set_title('LaplaceEdgeDetection',size=20)
for ax in axes:
    ax.axis('off') 
pylab.show()


#Applying convolution to acolor (RGB) image
im=imread("../Downloads/test.jpg").astype(np.float)


#read as float
print(np.max(im))
sharpen_kernel=np.array([0,-1,0,-1,5,-1,0,-1,0]).reshape((3,3,1))
emboss_kernel=np.array(np.array([[-2,-1,0],[-1,1,1],[0,1,2]])).reshape((3,3,1)) 
im_sharp=ndimage.convolve(im,sharpen_kernel,mode='nearest') 
im_sharp=np.clip(im_sharp,0,255).astype(np.uint8)


#clip(0 to 255) and convert to unsigned int 
im_emboss=ndimage.convolve(im,emboss_kernel,mode='nearest') 
im_emboss=np.clip(im_emboss,0,255).astype(np.uint8) 
pylab.figure(figsize=(10,15))
pylab.subplot(131)
pylab.imshow(im.astype(np.uint8))
pylab.axis('off') 
pylab.title('OriginalImage',size=25)
pylab.subplot(132)
pylab.imshow(im_sharp)
pylab.axis('off') 
pylab.title('SharpenedImage',size=25) 
pylab.subplot(133), pylab.imshow(im_emboss)
pylab.axis('off') 
pylab.title('EmbossedImage',size=25) 
pylab.tight_layout()
pylab.show()
im_gray=ndimage.convolve(im,emboss_kernel,mode='nearest') 
im_gray=np.clip(im_gray,0,255).astype(np.uint8) 
pylab.figure(figsize=(10,15))
pylab.subplot(133)
pylab.imshow(im_gray)
pylab.axis('off') 
pylab.title('grayImage',size=25) 
pylab.tight_layout() 
pylab.show()


#Template matching with cross-correlation between the image and template 
face_image=misc.face(gray=True)-misc.face(gray=True).mean() 
template_image=np.copy(face_image[300:365,670:750])
#righteye
template_image-=template_image.mean() 
face_image=face_image+np.random.randn(*face_image.shape)*50 #add random noise
correlation=signal.correlate2d(face_image,template_image,boundary='symm',mode='same') 
y,x=np.unravel_index(np.argmax(correlation),correlation.shape)
#find the match 
fig,(ax_original,ax_template,ax_correlation)=pylab.subplots(3,1,figsize=(6,15)) 
ax_original.imshow(face_image,cmap='gray') 
ax_original.set_title('Original',size=20)
ax_original.set_axis_off() 
ax_template.imshow(template_image,cmap='gray') 
ax_template.set_title('Template',size=20)
ax_template.set_axis_off()
ax_correlation.imshow(correlation,cmap='afmhot') 
ax_correlation.set_title('Cross-correlation',size=20) 
ax_correlation.set_axis_off() 
ax_original.plot(x,y,'ro')
fig.show()

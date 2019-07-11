import numpy as np
from skimage.transform import resize

#normalize R G B values to ensure all images have similar temperature and hue
def standardize(img):
    temp = np.zeros(img.shape)
    
    #R
    x = img[:,:,0]
    mean = np.mean(x.flatten())
    std = np.std(x.flatten())
    temp[:,:,0] = (x-mean)/std
 
    #G
    x = img[:,:,1]
    mean = np.mean(x.flatten())
    std = np.std(x.flatten())
    temp[:,:,1] = (x-mean)/std
 
    #B
    x = img[:,:,2]
    mean = np.mean(x.flatten())
    std = np.std(x.flatten())
    temp[:,:,2] = (x-mean)/std
    
    #bring the pixel values between 0 and 255
    temp = (temp-np.min(temp.flatten()))/(np.max(temp.flatten()) - np.min(temp.flatten()))
    temp = (temp*255).astype('uint8')
    
    return temp
    
#normalize intensity levels to make very dark images lighter and very light images darker 
def histogram_equalization(img):
   
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    temp = cdf[img]
    return temp

#preprocessing step applied to all images    
def preprocess(img,img_w = 100, img_h = 100, img_ch = 3): 
    temp = standardize(img)
    temp = histogram_equalization(temp)
    temp = resize(temp, (img_w, img_h, img_ch))
    return temp
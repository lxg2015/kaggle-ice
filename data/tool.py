#codeing=-utf8
import random
import cv2
import numpy as np
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
from skimage import morphology, measure

def randomCrop(img, w, h):
    '''
    img: WxHxC
    '''
    (img_h, img_w) = img.shape[:2]
    left = random.randrange(0, img_w-w)
    top = random.randrange(0, img_h-h)
    return img[top:top+w, left:left+h]

def rotate(img, angle, scale=1.0):
    '''
    img: object to rotate
    angle: degree not radian to ratate clockwise
    scale: Isotropic scale factor

    Args:
    BORDER_REFLECT: border is reflected (copied) http://answers.opencv.org/question/50706/border_reflect-vs-border_reflect_101/
    BORDER_REFLECT101: same as BORDER_REFLECT, but the outer side pixel is not copied
    '''
    (h,w) = img.shape[:2]
    rotated = cv2.getRotationMatrix2D((w/2,h/2), angle, scale)
    img = cv2.warpAffine(img, rotated, (w,h), borderMode=cv2.BORDER_REFLECT_101)
    return img

def lee_filter(img):
    # speckle filter https://stackoverflow.com/questions/39785970/speckle-lee-filter-in-python
    w,h,c = img.shape
    img_mean = uniform_filter(img, (w, w, 1))
    img_sqr_mean = uniform_filter(img**2, (w, w, 1))
    img_variance = img_sqr_mean - img_mean**2
    overall_variance = variance(img)
    img_weights = img_variance**2 / (img_variance**2 + overall_variance**2)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output

def object_crop(img, train, label=None):
    h,w,_ = img.shape
    max_area_0, box_0 = getmaxmask(img[:,:,0])
    max_area_1, box_1 = getmaxmask(img[:,:,1])
    max_area = max(max_area_0, max_area_1)
    if max_area_0 > max_area_1:
        box = box_0
    else:
        box = box_1
    # print(box)
    max_size_half = max(box[2]-box[0], box[3]-box[1])/2
    max_size_half = max(max_size_half*1.5, 15)

    cx, cy = (box[2]+box[0])/2, (box[3]+box[1])/2
    
    # add random for data-augmentation
    # if train and label==1: # ice
    #     cx += random.randrange(-5,5)
    #     cy += random.randrange(-5,5)
    # elif train and label==0: #ship
    #     cx += random.randrange(-13,13)
    #     cy += random.randrange(-13,13)

    minx = int(max(cx - max_size_half, 0))
    miny = int(max(cy - max_size_half, 0))
    maxx = int(min(cx + max_size_half, w))
    maxy = int(min(cy + max_size_half, h))
    
    return img[minx:maxx, miny:maxy,:], max_area

def getmaxmask(img):
    max_area = 0
    mask = img > img.mean() + 2*img.std()
    label = measure.label(mask)
    properity = measure.regionprops(label)
    box = None
    for region in properity:
        if region.area > max_area:
            max_area = region.area
            box = region.bbox
    # print(max_area, box)
    return max_area, box

def getMaskImg(img):
    img1 = img[:,:,0]
    mask1 = img1 > img1.mean() + 2*img1.std()
    img2 = img[:,:,1]
    mask2 = img2 > img2.mean() + 2*img2.std()
    mask = (mask1 + mask2) > 0
    mask = mask.astype(np.uint8)
    mask[mask==0] = 255

    return mask
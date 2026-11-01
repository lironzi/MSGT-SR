import math
import json
import cv2
import numpy as np
from PIL import Image
from math import radians, cos, sin, asin, sqrt
from torchvision import transforms
from glob import glob
import os
import tifffile
import matplotlib.pyplot as plt
import torch
from scipy.stats import multivariate_normal

def rot_mat(theta):
    return np.array([[math.cos(math.radians(theta)),-1*math.sin(math.radians(theta))],[math.sin(math.radians(theta)),math.cos(math.radians(theta))]])

def check_rotation(rot,img,bb,center=True):
    """"
    bb - a bounding box of [xmin,ymin,xmax,ymax]
    rot - a rotation matrix taking degrees
    img - the image to check if the bb after rotation is within the image
    center - True if rotation is done from the center of the image (0,0) otherwise

    Note that inorder to figure out if random xyxy coords are landing in the image after rotation of the image we need to see if those xyxy will return true for the inverse rotation.
    meaning check_rotation should be activated with -1*angle and if true than xyxy lands on the rotated image
    """
    y_max=img.shape[0]
    x_max=img.shape[1]
    xyxy=np.copy(bb)
    xyxy[1]=xyxy[1]*-1 #the y'is are being multiplied by -1 for the image coordinates
    xyxy[3]=xyxy[3]*-1
    if center: #convert the coordinate system to the center of the image
        xyxy[0]=xyxy[0]-x_max//2
        xyxy[1]=xyxy[1]+y_max//2
        xyxy[2]=xyxy[2]-x_max//2
        xyxy[3]=xyxy[3]+y_max//2

    rotated_points=[np.matmul(rot,np.array([xyxy[0],xyxy[1]])),np.matmul(rot,np.array([xyxy[2],xyxy[1]])),np.matmul(rot,np.array([xyxy[2],xyxy[3]])),np.matmul(rot,np.array([xyxy[0],xyxy[3]]))]


    flag=True
    for i,point in enumerate(rotated_points):
        if center:   #convert coordinate system back to (0,0)
            point[0]=point[0]+x_max//2
            point[1]=point[1]-y_max//2
            rotated_points[i]=point
        if point[0]<0 or point[0]>x_max or point[1]>0 or point[1]<-1*y_max:
            flag=False
            break

    return flag,rotated_points

def rotation(bb,img):
    p = np.random.rand()
    if p > 0:
        theta = int(np.random.rand() * 360)
        rotation_flag, _ = check_rotation(rot_mat(-1 * theta), img,bb)
        if rotation_flag:
            img = Image.fromarray(img)
            img=img.rotate(theta)
            img=np.array(img)
            print('rotated')
    return img

def change_to_uint8(img):
    if type(img)==np.ndarray:
        img=img-np.min(img)
        img=img/np.max(img)
        img=img*255
        img=img.astype('uint8')
    else:
        # image_flat=img.clone()
        # image_flat=image_flat.view(img.size(0),-1)
        # #image_flat -= image_flat.min(1, keepdim=True)[0]
        # image_flat /= image_flat.max(1, keepdim=True)[0]
        # img=image_flat.view(img.size())
        img=img/2**13
        img=img*255
        img=img.byte()
    return img

def degree_to_decimal(coord):
    temp=coord/100
    temp=int(temp)
    degree=temp
    temp=temp*100
    residue=coord-temp
    decimal=residue/60
    return degree+decimal

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in DMS degrees)
    """
    # convert to decimal:
    lon1=degree_to_decimal(lon1)
    lat1=degree_to_decimal(lat1)
    lon2=degree_to_decimal(lon2)
    lat2=degree_to_decimal(lat2)

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r*1000


def get_mean(path):

    tiff_path_list=glob(os.path.join(path,'*.tiff'))
    bands=['B02','B03','B04','B08']
    imgs=[]
    for tiff_path in tiff_path_list:
        if os.path.basename(tiff_path).split('.')[0].split('_')[-2] in bands:
           imgs.append(tifffile.imread(tiff_path))
    img=np.mean(np.array(imgs),axis=0)
    return img

def rotate(pt, radians, origin):
    x, y = pt
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
    return qx, qy

def get_new_point(original_shape,rotated_shape,x,y,angle):
    h, w = original_shape[:2]
    origin = (w/2, h/2)
    x1_new, y1_new = rotate([x, y], np.radians(angle), origin)

    h_new, w_new = rotated_shape[:2]
    xoffset, yoffset = (w_new - w)/2, (h_new - h)/2

    x1_new, y1_new = x1_new+xoffset, y1_new+yoffset

    return x1_new,y1_new

def get_new_shape(old_shape,theta):
    width,height=old_shape

    if (theta>=90 and theta<180) or (theta>=270 and theta<360):
        temp=width
        width=height
        height=temp
    theta = np.mod(theta, 90)
    new_height=np.ceil(np.ceil(np.sin(math.radians(theta))*width)+np.ceil(np.cos(math.radians(theta))*height))
    new_width = np.ceil(np.ceil(np.sin(math.radians(theta))*height)+np.ceil(np.cos(math.radians(theta))*width))
    return(new_width,new_height)

def get_x_bounderies(theta,target_height=256):
    # width,height=old_shape
    #
    # if (theta>=90 and theta<180) or (theta>=270 and theta<360):
    #     temp=width
    #     width=height
    #     height=temp
    theta = np.mod(theta, 90)
    if theta>45:
        theta=90-theta
    x_min=target_height*np.sin(math.radians(theta))*np.cos(math.radians(theta))
    # alpha=target_height/(height*np.cos(math.radians(theta)))
    # xmin=alpha*np.sin(math.radians(theta))*height
    return x_min

def get_y_bounderies(old_shape,new_shape,theta,x):
    width,height=old_shape
    new_width,new_height=new_shape
    if (theta>=90 and theta<180) or (theta>=270 and theta<360):
        temp=width
        width=height
        height=temp
    theta = np.mod(theta, 90)

    if x<new_width-np.ceil(np.sin(math.radians(theta)) * height):
        d = new_width - np.ceil(np.sin(math.radians(theta)) * height) - x
        y_min=np.tan(math.radians(theta))*d
    else:
        d=((new_width-x)/ np.ceil(np.sin(math.radians(theta)) * height))*(new_height-np.ceil(np.sin(math.radians(theta)) * width))
        y_min=new_height-np.ceil(np.sin(math.radians(theta)) * width)-d
    if x < np.ceil(np.sin(math.radians(theta)) * height):
        d=(x*np.ceil(np.cos(math.radians(theta)) * height))/np.ceil(np.sin(math.radians(theta)) * height)
        y_max=np.ceil(np.sin(math.radians(theta)) * width)+d
    else:
        d=x-np.ceil(np.sin(math.radians(theta)) * height)
        y=np.ceil(np.tan(math.radians(theta))*d)
        y_max=new_height-y

    return y_min,y_max

def get_y_boundaries(old_shape, new_shape, theta_deg, x):
    """
    Compute valid y_min and y_max crop coordinates for a given x coordinate,
    ensuring the crop remains within the rotated (expanded=True) image and
    does not include padded pixels.

    Args:
        old_shape: tuple (width, height) of the original image
        new_shape: tuple (new_width, new_height) after rotation with expand=True
        theta_deg: rotation angle in degrees
        x: current x coordinate in the rotated image

    Returns:
        (y_min, y_max): valid y coordinate bounds
    """
    width, height = old_shape
    new_width, new_height = new_shape

    # Normalize angle to [0, 360)
    theta_deg = np.mod(theta_deg, 360)

    # For angles in the 2nd or 4th quadrants, swap width/height
    if 90 <= theta_deg < 180 or 270 <= theta_deg < 360:
        width, height = height, width

    # Reduce to first quadrant for symmetry
    theta = math.radians(theta_deg % 90)

    # Precompute trigonometric values
    sin_t = math.sin(theta)
    cos_t = math.cos(theta)
    tan_t = math.tan(theta)

    # Avoid degenerate cases
    if abs(sin_t) < 1e-8 or abs(cos_t) < 1e-8:
        return 0.0, float(new_height)

    # Compute y_min
    if x < new_width - math.ceil(sin_t * height):
        d = new_width - math.ceil(sin_t * height) - x
        y_min = tan_t * d
    else:
        denom = math.ceil(sin_t * height)
        if denom == 0:
            return 0.0, float(new_height)
        d = ((new_width - x) / denom) * (new_height - math.ceil(sin_t * width))
        y_min = new_height - math.ceil(sin_t * width) - d

    # Compute y_max
    if x < math.ceil(sin_t * height):
        denom = math.ceil(sin_t * height)
        y_max = math.ceil(sin_t * width) + (x * math.ceil(cos_t * height)) / denom
    else:
        d = x - math.ceil(sin_t * height)
        y = math.ceil(tan_t * d)
        y_max = new_height - y

    # Clamp to valid image range
    y_min = max(0, y_min)
    y_max = min(new_height, y_max)
    return float(y_min), float(y_max)

def get_bands(path,bands,rotate_flag=0):
    l=glob(os.path.join(path,'*.tiff'))
    l=[p for i,p in enumerate(l) if i in bands]
    img_list=[]
    for p in l:
        img=tifffile.imread(p)
        img=cv2.rotate(img,rotate_flag)
        img=cv2.resize(img, (168,128), cv2.INTER_CUBIC)
        img_list.append(img)
    img_list=np.array([img_list])

    return img_list


def get_laplace_min(loss_map):
    search_area=20
    laplace=cv2.Laplacian(change_to_uint8(loss_map),3,8)
    size=loss_map.shape[0]

    y=np.argmax(laplace)//size
    x=np.argmax(laplace)-y*size

    search_area_left = np.min([search_area,x])
    search_area_up=np.min([search_area,y])

    min=np.argmin(loss_map[y-search_area_up:y+search_area,x-search_area_left:x+search_area])
    y_min=min//(search_area_left+search_area)
    x_min=min-y_min*(search_area_left+search_area)

    true_x=x-search_area_left+x_min
    true_y=y-search_area_up+y_min

    return true_x,true_y

def generate_grid(sigma):
    rv = multivariate_normal([16, 16], sigma)
    grid=np.zeros(32,32)
    for i in range(32):
        for j in range(32):
            grid[j,i]=rv.pdf((i,j))
    return grid

def change_to_jpg(img):
    img=change_to_uint8(img)
    img=np.expand_dims(img,axis=-1)
    img=np.repeat(img,3,axis=-1)
    return img
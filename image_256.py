import random
import os
from PIL import Image,ImageFilter,ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2
import pdb
img_patchs=[]
def load_data(img_path,train = True):
    gt_path = img_path.replace('.jpg','.h5').replace('images','ground_truth')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
#   try:

    #print('image.py line 15',gt_file)
    target = np.asarray(gt_file['density'])
#   except:
#        pdb.set_trace()
    if train == True:
        crop_size = (256,256)
        if random.randint(0,9)<= -1:
            
            
            dx = int(random.randint(0,1)*img.size[0]*1./2)
            dy = int(random.randint(0,1)*img.size[1]*1./2)
        else:
            dx = int(random.random()*img.size[0]*1./2)
            dy = int(random.random()*img.size[1]*1./2)
        
#        print('dx=',dx,'dy=',dy)    
        img = img.crop((dx,dy,crop_size[0]+dx,crop_size[1]+dy))
        target = target[dy:crop_size[1]+dy,dx:crop_size[0]+dx]
        target = cv2.resize(target,(32,32),interpolation = cv2.INTER_CUBIC)*(img.size[1]//32)*(img.size[0]//32)
    return img,target
#        print(img.size[1],img.size[0])       
#        print(target.shape[0],target.shape[1])       
#        if random.random()>0.8:
#            target = np.fliplr(target)
#            img = img.transpose(Image.FLIP_LEFT_RIGHT)

#    img = img.resize((img.size[0]//8*8,img.size[1]//8*8)) 
#    img = img.resize((256,256)) 

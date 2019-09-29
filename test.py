import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter 
import scipy
import json
import math
import torchvision.transforms.functional as F
from matplotlib import cm as CM
from image_git import *
from model_VGG import CSRNet
import torch
import sys
from torchvision import datasets, transforms
transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])
path='/home/timing/Git_Repos_Summer/CSRNet-pytorch/data/ShanghaiTech/merge_AB/test_data/'

#print('test_mergeAB_testA.py is running~~~')
print('path of test images is',path)

img_paths = []
gt_path = []
gt_paths = []
for img_path in glob.glob(os.path.join(path,'images', '*.jpg')):
    img_paths.append(img_path)
    gt_path = img_path.replace('.jpg','.h5').replace('images','ground_truth')
    gt_paths.append(gt_path)
#print(gt_paths)
model = CSRNet()
model = model.cuda()
checkpoint = torch.load('DA_AB_190924model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

mae = 0
mse=0
for i in range(len(img_paths)):
#    print('i= ',i)
    #img = 255.0 * F.to_tensor(Image.open(img_paths[i]).convert('RGB'))
    #img = 255.0 * F.to_tensor(Image.open(img_paths[i]).convert('RGB'))
    #img[0,:,:]=img[0,:,:]-92.8207477031
    #img[1,:,:]=img[1,:,:]-95.2757037428
    #img[2,:,:]=img[2,:,:]-104.877445883
    img = transform(Image.open(img_paths[i]).convert('RGB')).cuda()
    #gt_file = h5py.File(img_paths[i].replace('.jpg','.h5').replace('images','ground_truth'),'r')
    gt_file = h5py.File(gt_paths[i])
    groundtruth = np.asarray(gt_file['density'])
    output,_ = model(img.unsqueeze(0))
#    print('i= ',i,'output=',output.detach().cpu().sum().numpy())
    error_per =abs(output.detach().cpu().sum().numpy()-np.sum(groundtruth)/np.sum(groundtruth))
    print('%s,%f,%f,%f'%(img_paths[i],np.sum(groundtruth),output.detach().cpu().sum(),abs(error_per)))
    mae += abs(np.sum(groundtruth) - abs(output.detach().cpu().sum().numpy()))
    mse += math.pow(np.sum(groundtruth) - output.detach().cpu().sum().numpy(),2)
    #print(i,mae)
print('number of images',len(img_paths))
print('MAE=',mae/len(img_paths))
print('MSE=',mse/len(img_paths))

import sys
import os

import warnings

from model_git import CSRNet

from utils import save_checkpoint

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

import numpy as np
import argparse
import json
import cv2
import dataset_git as dataset
import time
import mmd
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

parser = argparse.ArgumentParser(description='PyTorch CSRNet')

parser.add_argument('train_jsonA', metavar='train_A',
                    help='path to trainA json')
parser.add_argument('train_jsonB', metavar='TRAIN_B',
                    help='path to trainB json')
parser.add_argument('test_json', metavar='TEST',
                    help='path to test json')

parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None,type=str,
                    help='path to the pretrained model')

parser.add_argument('gpu',metavar='GPU', type=str,
                    help='GPU id to use.')

parser.add_argument('task',metavar='TASK', type=str,
                    help='task id to use.')
#i=0
#j=0
#img_tg=[]
#target_tg=[]
def main():
    
    global args,best_prec1
    
    best_prec1 = 1e6
    
    args = parser.parse_args()
    args.original_lr = 1e-7
    args.lr = 1e-7
    args.batch_size    = 1
    args.momentum      = 0.95
    args.decay         = 5*1e-4
    args.start_epoch   = 0
    args.epochs = 400
    args.steps         = [-1,1,100,150]
    args.scales        = [1,1,1,1]
    args.workers = 4
    args.seed = time.time()
    args.print_freq = 50
    with open(args.train_jsonA, 'r') as outfile1:        
        train_list_target = json.load(outfile1)
    with open(args.train_jsonB, 'r') as outfile2:        
        train_list_source = json.load(outfile2)
    with open(args.test_json, 'r') as outfile:       
        val_list = json.load(outfile)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.manual_seed(args.seed)
    print('GPU=',args.gpu)
    sys.stdin.readline()
    model = CSRNet()
    print(model)
    model = model.cuda()
    
    criterion = nn.MSELoss(size_average=False).cuda()
    
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.decay)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1: 
        model = nn.DataParallel(model,device_ids=[0,1,2])
        model.to(device)

    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))
            
    for epoch in range(args.start_epoch, args.epochs):
        
        adjust_learning_rate(optimizer, epoch)
        
        train(train_list_source, train_list_target, model, criterion, optimizer, epoch)
        prec1 = validate(val_list, model, criterion)
        
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(' * best MAE {mae:.3f} '
              .format(mae=best_prec1))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best,args.task)

def train(train_list_source, train_list_target, model, criterion, optimizer, epoch):
    
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    train_loader_source = torch.utils.data.DataLoader(
        dataset.listDataset(train_list_source,
                       shuffle=True,
                       transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]), 
                       train=True, 
                       seen=model.seen,
                       batch_size=args.batch_size,
                       num_workers=args.workers),
        batch_size=args.batch_size)

    train_loader_target = torch.utils.data.DataLoader(
        dataset.listDataset(train_list_target,
                       shuffle=True,
                       transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]), 
                       train=True, 
                       seen=model.seen,
                       batch_size=args.batch_size,
                       num_workers=args.workers),
        batch_size=args.batch_size)


    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader_source.dataset), args.lr))
    
    model.train()
    end = time.time()

    iter_source = iter(train_loader_source)
    iter_target = iter(train_loader_target)
    num_iter = len(train_loader_source)

    for i in range(1, num_iter):
#    for i,(img_sr, target_sr)in enumerate(train_loader_source) and j,(img_tg, target_tg)in enumerate(train_loader_target):
        data_time.update(time.time() - end)

        img_sr,target_sr = iter_source.next()
        img_tg,target_tg = iter_target.next()

        img_sr = img_sr.cuda()
        img_tg = img_tg.cuda()
        img_sr = Variable(img_sr)        
        img_tg = Variable(img_tg)
        output_sr = model(img_sr)       
        output_tg = model(img_tg)          
        target_sr = target_sr.type(torch.FloatTensor).unsqueeze(0).cuda()
        target_tg = target_tg.type(torch.FloatTensor).unsqueeze(0).cuda()
        target_sr = Variable(target_sr)        
        target_tg = Variable(target_tg)
        
        loss_sr = criterion(output_sr, target_sr)
        loss_tg = criterion(output_tg, target_tg)
        loss_mmd = mmd.mmd_linear(output_sr, output_tg)
        loss = loss_sr + loss_tg + loss_mmd

        losses.update(loss.item(), img_sr.size(0))
        losses.update(loss.item(), img_tg.size(0))        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                   epoch, i, len(train_loader_source), batch_time=batch_time,
                   data_time=data_time, loss=losses))
    
def validate(val_list, model, criterion):
    print ('begin test')
    test_loader = torch.utils.data.DataLoader(
    dataset.listDataset(val_list,
                   shuffle=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]),  train=False),
    batch_size=args.batch_size)    
    
    model.eval()
    
    mae = 0
    #mae_list=[]
    for i,(img, target) in enumerate(test_loader):
        img = img.cuda()
        img = Variable(img)
        output = model(img)
        
        mae += abs(output.data.sum()-target.sum().type(torch.FloatTensor).cuda())       
        #mae_list.append(abs(output.data.sum()-target.sum().type(torch.FloatTensor).cuda()))

    print(' * MAE {mae:.3f} '
              .format(mae=mae))

    mae = mae/len(test_loader)    
    #np.savetxt('mae_list200.txt',mae_list)
    return mae    
        
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    
    
    args.lr = args.original_lr
    
    for i in range(len(args.steps)):
        
        scale = args.scales[i] if i < len(args.scales) else 1
        
        
        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    
    
if __name__ == '__main__':
    main()        

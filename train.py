#(M_pytorch1.0python3.6)
#python train.py --model unet
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from multiprocessing import Pool 
from loss import LossSelector
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig
from pathlib import Path
from dataset import camvid, joint_transforms
import utils.imgs
from models import ModelSelector
from models.DeepLabV3 import deeplabv3
from models.DeepLabV3_plus import deeplabv3_plus
from models.PSPNet import pspnet
from models.Unet import Unet3
from models.Unet_AE import unet_ae
from models.AttentionR2Unet import R2AttUnet
from models.AttentionUnet import AttUnet
from models.RecurrentUnet import R2Unet
from models.Unet import Unet1, Unet2
from models.Segnet import segnet
from models.CENet import cenet
from models.Unet_nested import UNet_Nested
from models.DenseASPP import denseaspp
from models.RefineNet import RefineNet
from models.RDFNet import rdfnet
################Model Cofig

ce_params = {'weight': None}
dice_params = {'smooth': 1}
focal_params = {'weight': None, 'gamma': 2, 'alpha': 0.5}
lovasz_params = {'multiclasses': True}
parser = argparse.ArgumentParser(description='PyTorch AV Training')
# Optimization options
parser.add_argument('--epochs', default=2000, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float, metavar='LR', help='initial learning rate')

parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
parser.add_argument('--val-iteration', type=int, default=1, help='Number of labeled data')
parser.add_argument('--out', default='result', help='Directory to output the result')
parser.add_argument('--model', default='unet', choices=['unet', 'unet_ae', 'R2unet', 'Attunet', 'R2Attunet', 'deeplabv3', 'deeplabv3_plus', 'pspnet', 'segnet', 'cenet', 'unet_nested', 'denseaspp', 'refinenet', 'rdfnet'], type=str)
parser.add_argument('--loss', default='focal', choices=['ce', 'dice', 'focal', 'lovasz'], type=str)
parser.add_argument('--loss_params', default={'ce': ce_params,
                                             'dice': dice_params,
                                             'focal': focal_params,
                                             'lovasz': lovasz_params})
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
# Use CUDA
use_cuda = torch.cuda.is_available()
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)
###################
va= 'train'
PATH = args.model
start_epoch = 0
inch = 3
class_num = 4
LoadData = True
Gray_Flag = False
batch_size = 8
###################
best_dice = 0.  # best test accuracy
best_acc = 0.  # best test accuracy
def main():
    ##################################################################data
    CAMVID_PATH = Path('/home/zhaojie/zhaojie/Mixmatch/Data/dataprocess/RITE-RGB-40-True/')
    transform_train = transforms.Compose([transforms.ToTensor()])
    train_joint_transformer = transforms.Compose([joint_transforms.JointRandomHorizontalFlip()])
	#train
    train_dset = camvid.CamVid(CAMVID_PATH, Gray = Gray_Flag, index_start = 0, index_end = 20, joint_transform=train_joint_transformer, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size, shuffle=True)
	#val
    val_dset = camvid.CamVid(CAMVID_PATH, Gray = Gray_Flag, index_start = 20, index_end = 40, joint_transform=None, transform=transform_train)
    val_loader = torch.utils.data.DataLoader(val_dset, batch_size=batch_size, shuffle=True)
    makedir_Flag = True
    if makedir_Flag:
        dirs = ['./result/' +PATH +'/visulizeT0/','./result/' +PATH+'/visulizeT1/']
        for dir_num in range(len(dirs)):
            os.makedirs(dirs[dir_num], exist_ok = True)
    ########################
    global best_dice
    global best_acc
    print(args.model)
    if args.model =='unet':
        model = Unet3.UNet(in_channels = inch, num_classes = class_num, filter_scale=1)
    elif args.model =='deeplabv3':
        model = deeplabv3.DeepLabV3(class_num = class_num)
    elif args.model =='deeplabv3_plus':
        model = deeplabv3_plus.DeepLabv3_plus(in_channels = inch, num_classes = class_num, backend = 'mobilenet_v2', os = 16, pretrained = 'imagenet')
    elif args.model =='pspnet':
        model = pspnet.PSPNet(in_channels = inch, num_classes = class_num, backend = 'resnet101',pool_scales = (1, 2, 3, 6), pretrained = 'imagenet')
    elif args.model =='unet_ae':
        model = unet_ae.UnetResnetAE(in_channels = inch, num_classes = class_num, backend = 'resnet101', pretrained = 'imagenet')
    elif args.model =='U_Net':
        model = Unet2.U_Net(img_ch= inch,output_ch = class_num)
    elif args.model =='R2U_Net':
        model = R2Unet.R2U_Net(img_ch = inch,output_ch = class_num,t = args.t)
    elif args.model =='Attunet':
        model = AttUnet.AttU_Net(img_ch = inch,output_ch = class_num)
    elif args.model == 'R2Attunet':
        model = R2AttUnet.R2AttU_Net(img_ch = inch,output_ch = class_num,t = args.t)
    elif args.model == 'segnet':
        model = segnet.SegNet(num_classes = class_num, in_channels= inch)
    elif args.model == 'cenet':
        model = cenet.CE_Net(num_classes = class_num, num_channels= inch)
    elif args.model == 'unet_nested':
        model = UNet_Nested.UNet_Nested(in_channels = inch, n_classes= class_num)
    elif args.model == 'denseaspp':
        model = denseaspp.DenseASPP(class_num = class_num)
    elif args.model == 'refinenet':
        model = RefineNet.get_refinenet(input_size = 256, num_classes = class_num)
    # elif args.model == 'rdfnet':#out = net(left, right)
        # model = rdfnet.RDF(input_size = 256, num_classes = class_num)
    device = torch.device("cuda")
    model = torch.nn.DataParallel(model).to(device).cuda()
    print('model',model)
    print('loss',args.model, args.loss)
    criterion = LossSelector[args.loss](**args.loss_params[args.loss])
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    test_dices = []
    test_aucs = []
    model_fold = "./PKL/"+PATH + '/'
    os.makedirs(model_fold, exist_ok = True)
    for epoch in range(start_epoch, args.epochs):
        print('--------------------------------------')
        ###########################
        
        #####################
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        print('val', va)
        train_loss, train_dice_x1, train_dice_x2, train_dice_x3, train_acc = train(train_loader, model, optimizer, epoch, dirs[0], criterion)
        val_loss, val_dice1, val_dice2, val_dice3, val_acc = validate(val_loader, model, epoch, dirs[1])
            # test(train_test_loader, model, epoch, dirs[2])
        # save model
        monitor_dice = val_dice1 + val_dice2 + val_dice3
        monitor_acc = val_acc
        is_best_dice = monitor_dice > best_dice
        is_best_acc = monitor_acc > best_acc
        best_dice = max(monitor_dice, best_dice)
        best_acc = max(monitor_acc, best_acc)
        if is_best_dice or is_best_acc:
            name = model_fold + str(epoch) + '_D1-' + str(round(val_dice1,4)) + '_D2-' + str(round(val_dice2,4))+ '_D3-' + str(round(val_dice3,4))+ '_A-' + str(round(val_acc,4)) + '.pkl'
            print(name)
            # torch.save(model, name)
    print('Best dice:', best_dice)
    print('Best acc:', best_acc)

def train(trainloader, model, optimizer, epoch, dir, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    dices_x1 = AverageMeter()
    dices_x2 = AverageMeter()
    dices_x3 = AverageMeter()
    accs = AverageMeter()
    Threshod = 0.7
    train_iter = iter(trainloader)
    model.train()
    for batch_idx in range(args.val_iteration):
        try:
            inputs_x, targets_x, name_x = train_iter.next()
        except:
            train_iter = iter(trainloader)
            inputs_x, targets_x, name_x = train_iter.next()
        batch_size = inputs_x.size(0)

        inputs_x, targets_x = torch.FloatTensor(inputs_x.float()).cuda(), torch.FloatTensor(targets_x.float()).cuda().long()
        outputs = model(inputs_x)

        if not isinstance(outputs, tuple):
            loss = criterion(outputs, targets_x)
            pred = get_predictions(outputs)
        elif len(outputs) == 2:
            # For pspnet outputs
            loss = criterion(outputs[1], targets_x)
            pred = get_predictions(outputs[1])
        x_dice1, x_dice2, x_dice3 = get_dice(pred, targets_x, Threshod)
        # print('dice', x_dice1, x_dice2, x_dice3)
        ###############
        acc = 1 - error(pred, targets_x.data.cpu())
        # record loss
        dices_x1.update(x_dice1, inputs_x.size(0))
        dices_x2.update(x_dice2, inputs_x.size(0))
        dices_x3.update(x_dice3, inputs_x.size(0))
        losses.update(loss.item(), inputs_x.size(0))
        accs.update(acc, inputs_x.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #save output
        if random.random() > 1:
            visulize(name_x, inputs_x, pred, targets_x, epoch, dir, Threshod)
    print('train---Loss: {loss:.4f} | Dice_x1: {dice_x1:.4f}| Dice_x2: {dice_x2:.4f} | Dice_x3: {dice_x3:.4f} | acc: {acc:.4f}'.format(loss=losses.avg, dice_x1=dices_x1.avg, dice_x2=dices_x2.avg, dice_x3=dices_x3.avg, acc=accs.avg))
    return (losses.avg, dices_x1.avg, dices_x2.avg, dices_x3.avg, accs.avg)

def validate(valloader, model, epoch, dir):
    losses = AverageMeter()
    dices1 = AverageMeter()
    dices2 = AverageMeter()
    dices3 = AverageMeter()
    accs = AverageMeter()
    model.eval()
    Threshod = 0.2
    end = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets, names) in enumerate(valloader):
            # measure data loading time
            inputs, targets = torch.FloatTensor(inputs.float()).cuda(), torch.FloatTensor(targets.float()).cuda().long()
            outputs = model(inputs)
            loss = loss_calc(outputs, targets)
            # print('shape', inputs.shape, outputs.shape, targets.shape)#torch.Size([4, 3, 512, 512]) torch.Size([4, 4, 512, 512]) torch.Size([4, 512, 512])
            pred = get_predictions(outputs)
            x_dice1, x_dice2, x_dice3 = get_dice(pred, targets, Threshod)
            # print('dice', x_dice1, x_dice2, x_dice3)
            ###############
            acc = 1 - error(pred, targets.data.cpu())
            # record loss
            dices1.update(x_dice1, inputs.size(0))
            dices2.update(x_dice2, inputs.size(0))
            dices3.update(x_dice3, inputs.size(0))
            losses.update(loss.item(), inputs.size(0))
            accs.update(acc, inputs.size(0))
            #save output
            if random.random() > 1:
                visulize(names, inputs, pred, targets, epoch, dir, Threshod)
        print('val---Loss: {loss:.4f} | Dice1: {dice1:.4f}| Dice2: {dice2:.4f} | Dice3: {dice3:.4f}| acc: {acc:.4f}'.format(loss=losses.avg, dice1=dices1.avg, dice2=dices2.avg, dice3=dices3.avg, acc=accs.avg))
    return (losses.avg, dices1.avg, dices2.avg, dices3.avg, accs.avg)
def test(valloader, model, epoch, dir):
    losses = AverageMeter()
    dices1 = AverageMeter()
    dices2 = AverageMeter()
    dices3 = AverageMeter()
    accs = AverageMeter()
    model.eval()
    Threshod = 0.2
    end = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, names) in enumerate(valloader):
            # measure data loading time
            inputs = torch.FloatTensor(inputs.float()).cuda()
            outputs = model(inputs)
            
            # print('shape', inputs.shape, outputs.shape, targets.shape)#torch.Size([4, 3, 512, 512]) torch.Size([4, 4, 512, 512]) torch.Size([4, 512, 512])
            pred = get_predictions(outputs)
            if random.random() > 1:
                visulize_test(names, inputs, pred, epoch, dir, Threshod)

def visulize(a, b, c, d, epoch, dir, Threshod):
    
    #a, b, c, d= names, inputs, outputs, targets
    if epoch % 1 == 0:
        for batch_num in range(len(a)):
            batch_num = 0
            name_save = a[batch_num]
            inputs_save = b[batch_num].data.squeeze(0).cpu().numpy()
            if inputs_save.shape[0] == 3:
                inputs_save = np.transpose(inputs_save, (1,2,0))[:,:,::-1]
            outputs_save = c[batch_num].data.squeeze(0).cpu().numpy()
            targets_save = d[batch_num].data.squeeze(0).cpu().numpy()
            # print('shape0:', inputs_save.shape, outputs_save.shape, targets_save.shape, np.max(inputs_save), np.max(outputs_save))#(3, 512, 512) (512, 512) (512, 512) 0.9725647 1.0
            outputs_save = colorize_mask(outputs_save, class_num).astype(np.int32)
            targets_save = colorize_mask(targets_save, class_num).astype(np.int32)
            # print('shape1:', inputs_save.shape, outputs_save.shape, targets_save.shape, np.max(inputs_save), np.max(outputs_save), np.max(targets_save))
            if outputs_save.shape[0] == 3:
                outputs_save = np.transpose(outputs_save, (1,2,0))#[:,:,::-1]
                targets_save = np.transpose(targets_save, (1,2,0))#[:,:,::-1]
            cv2.imwrite(dir + str(epoch) + '_' + str(name_save) + '_Inp.png', 255 * inputs_save)
            cv2.imwrite(dir + str(epoch) + '_' + str(name_save) + '_Out.png', outputs_save)
            
            cv2.imwrite(dir + str(epoch) + '_' + str(name_save) + '_Tar.png', targets_save)
def visulize_test(a, b, c, epoch, dir, Threshod):
    
    #a, b, c, d= names, inputs, outputs, targets
    if epoch % 1 == 0:
        for batch_num in range(len(a)):
            batch_num = 0
            name_save = a[batch_num]
            inputs_save = b[batch_num].data.squeeze(0).cpu().numpy()
            if inputs_save.shape[0] == 3:
                inputs_save = np.transpose(inputs_save, (1,2,0))[:,:,::-1]
            outputs_save = c[batch_num].data.squeeze(0).cpu().numpy()
            outputs_save = colorize_mask(outputs_save, class_num).astype(np.int32)
            # print('shape1:', inputs_save.shape, outputs_save.shape, targets_save.shape, np.max(inputs_save), np.max(outputs_save), np.max(targets_save))
            if outputs_save.shape[0] == 3:
                outputs_save = np.transpose(outputs_save, (1,2,0))
                
            cv2.imwrite(dir + str(epoch) + '_' + str(name_save) + '_Inp.png', 255 * inputs_save)
            cv2.imwrite(dir + str(epoch) + '_' + str(name_save) + '_Out.png', outputs_save)

def For_dice(y_true, y_pred, N, index):
    pred_copy = torch.zeros((N, y_pred.shape[1],y_pred.shape[2]))
    true_copy = torch.zeros((N, y_true.shape[1],y_true.shape[2]))
    pred_copy[y_pred == index] = 1
    true_copy[y_true == index] = 1
    true_copy, pred_copy = true_copy.data.cpu().numpy(), pred_copy.data.cpu().numpy()
    # print('max1',np.max(true_copy), np.max(pred_copy))#1 1
    dice = 2 * (pred_copy * true_copy).sum(1).sum(1) / (pred_copy.sum(1).sum(1) + true_copy.sum(1).sum(1) + 1e-5)
    dice = dice.sum() / N
    # print(dice)
    return dice
def get_dice(y_true, y_pred, Threshod):
    smooth = 1e-5
    dices = []
    N = y_true.shape[0]
    # print('max0',N, np.max(y_true.data.cpu().numpy()), np.max(y_pred.data.cpu().numpy()))#3 3
    for index in range(1,4):#1,2,3 classnum
        
        dice = For_dice(y_true, y_pred, N, index)
        dices.append(dice)
    return dices[0], dices[1], dices[2]

def loss_calc(pred, label):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    return criterion(pred, label)

def get_predictions(output_batch):
    bs,c,h,w = output_batch.size()
    tensor = output_batch.data
    values, indices = tensor.cpu().max(1)
    indices = indices.view(bs,h,w)
    return indices

def colorize_mask(mask, class_num):#R=3,G=2,B=1
    # mask: numpy array of the mask
    
    # print('mask.shape', mask.shape)
    mask_save = np.zeros((class_num -1, mask.shape[0], mask.shape[1]))
    # print('mask_save.shape', mask_save.shape)
    for index in range(1, class_num):
        mask_copy = np.zeros((mask.shape[0], mask.shape[1]))
        # print('mask_copy.shape', mask_copy.shape)
        mask_copy[mask == index] = 255
        mask_save[index-1,:,:] = mask_copy

    return mask_save

def error(preds, targets):
    assert preds.size() == targets.size()
    bs,h,w = preds.size()
    n_pixels = bs*h*w
    # n_pixels = 1
    incorrect = preds.ne(targets).cpu().sum().numpy()
    err = incorrect/n_pixels
    # print(incorrect,n_pixels,err)
    # return round(err,5)
    return err
	
def init(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(module.weight.data, 0.15)
        nn.init.constant_(module.bias.data, 0)
if __name__ == '__main__':
    main()

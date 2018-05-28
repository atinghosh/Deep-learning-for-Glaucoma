#This can take patient ID with any no. of slices
#This code will work in delta, to make it work in gamma change the data path e.g. root_dir and seg_loc
##This has 32x1 vector in the last layer

from sys import path
path.insert(0,'/home/atin/DeployedProjects/Glaucoma-DL/')

import pickle
from Atin.Russian.helper import *
import Huy.transforms as imgT

import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torch.utils.data.sampler import WeightedRandomSampler
import torch.optim as optim
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
import torchsample.transforms as tr

from pytorch_monitor import monitor_module, init_experiment
from sklearn import metrics
from tensorboardX import SummaryWriter
import datetime

import random
import os
import matplotlib.pyplot as plt
from scipy.misc import imresize
from scipy.io import loadmat
from scipy import signal
from random import shuffle
from PIL import Image

import math
import sys
import warnings

from collections import Counter
import pandas as pd

if not sys.warnoptions:
    warnings.simplefilter("ignore")

os.environ["CUDA_VISIBLE_DEVICES"]="3"
CUDA = True
LOG_INTERVAL = 10



def preprocess_image(loc, resize_dim = (143,400), transform = None, diag = True):
    '''
    :param loc:
    :param resize_dim: must be a tuple like (143,400)
    :param transform:
    :return:
    '''
    im = plt.imread(loc)
    im = imresize(im, resize_dim,interp='nearest') #original image size is
    # im = imresize(im, (200, 400), interp='nearest')
    #im = imresize(im, (125, 350), interp='nearest')
    im = im / 255
    if not diag:
        im = np.expand_dims(im, 0)
        im = im.transpose((1, 2, 0))

    if transform is not None:
        im = transform(im)

    return im


def seg_map_convert_one_hot(sample_im, channel_first=True, no_of_class=8):
    '''convert HxW segented map to one hot CxHxW
    '''
    shape = sample_im.shape
    class_in_the_seg_map = np.unique(sample_im)
    if len(class_in_the_seg_map) != no_of_class:
        missing_class = list(set(np.arange(1, 8)) - set(class_in_the_seg_map))
    else:
        missing_class = []

    returned_array = np.zeros((no_of_class, shape[0], shape[1]))
    sample_im_flattened = np.reshape(sample_im, (-1))
    one_hot = pd.get_dummies(sample_im_flattened)

    j = 0
    for i in range(no_of_class):
        if i + 1 in missing_class:
            returned_array[i, :, :] = np.zeros((shape[0], shape[1]))
        else:
            flattened_array = np.array(one_hot[one_hot.columns[j]])
            returned_array[i, :, :] = flattened_array.reshape(shape[0], shape[1])
            j += 1

    if channel_first:
        return returned_array
    else:
        returned_array = np.transpose(returned_array, (1, 2, 0))
        return returned_array


def make_balance(small_list, big_list):
    ratio = math.floor(len(big_list) / len(small_list))
    if ratio > .5:
        final_list = small_list * (ratio + 1) + big_list
    else:
        final_list = small_list * ratio + big_list

    shuffle(final_list)
    return final_list


# Transformation
def create_dark_mask():
    """create dark mask for adding black box
    """
    black_dx = 60
    black_dy = 20
    dark_mask = np.zeros((black_dx, black_dy))
    for k in range(black_dy):
        dark_mask[:, k] = (np.abs(k - black_dy // 2) / (black_dy / 2.)) ** 2
    return dark_mask


def create_elastic_indices():
    """create indices for elastic deformation
    used once at the start epoch
    """
    # initial values
    alpha, alpha2, sigma = 10, 10, 50
    shape = (96, 288)  # same as shape of input images
    x_mesh, y_mesh = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

    # below is used once per epoch for the elastic deformation
    g_1d = signal.gaussian(300, sigma)
    kernel_deform = np.outer(g_1d, g_1d)
    dx = signal.fftconvolve(np.random.rand(*shape) * 2 - 1, kernel_deform, mode='same')
    dy = signal.fftconvolve(np.random.rand(*shape) * 2 - 1, kernel_deform, mode='same')
    dx = alpha * (dx - np.mean(dx)) / np.std(dx)
    dy = alpha2 * (dy - np.mean(dy)) / np.std(dy)
    indices_x, indices_y = x_mesh + dx, y_mesh + dy
    indices_x_clipped = np.clip(indices_x, a_min=0, a_max=shape[1] - 1)
    indices_y_clipped = np.clip(indices_y, a_min=0, a_max=shape[0] - 1)
    return indices_x_clipped, indices_y_clipped


def train_transform(dark_mask, indices_x_clipped, indices_y_clipped):
    return imgT.EnhancedCompose([
        # random rotation
        # random flip
        T.Lambda(imgT.randomlyFlip),
        # intensity nonlinear
        [T.Lambda(imgT.intensityNonliearShift), None],
        # add blackbox
        [imgT.AddBlackBox(dark_mask), None],
        # imgT.RandomRotate(),
        # elastic deformation
        imgT.ElasticDeformation(indices_x_clipped, indices_y_clipped),
        # multiplicative gauss
        [imgT.AddGaussian(ismulti=True), None],
        # additive gauss
        [imgT.AddGaussian(ismulti=False), None],
        # for non-pytorch usage, remove to_tensor conversion
        [T.Lambda(imgT.to_tensor), T.Lambda(imgT.to_tensor_target)]
    ])


def val_transform():
    return imgT.EnhancedCompose([
        # for non-pytorch usage, remove to_tensor conversion
        [T.Lambda(imgT.to_tensor), T.Lambda(imgT.to_tensor_target)]
    ])


class Beijing_Seg_Dataset(Dataset):

    '''
    :return diag_output, seg_output where diag_output is torch.cat(diag_image_array),train_y and seg_output is im, seg_im
    '''

    def __init__(self, seg_patient_id, diag_root, seg_root, resize_dim, transform_seg = None):

        '''
        :param seg_patient_id: must be a list with each element as e.g. ['12956', '4', 'G']
        :param diag_root: root_dir for diagnosis
        :param seg_root:  root_dir for segmented map
        :param transform: composed transformation
        '''

        self.patient_id = seg_patient_id
        self.diag_root = diag_root
        self.seg_root = seg_root
        self.resize_dim = resize_dim
        self.transform_seg = transform_seg

    def __len__(self):
        return len(self.patient_id)

    def __getitem__(self, idx):
        id, slice_id, h_or_g = self.patient_id[idx]


        #Segmentation data preprocess

        if h_or_g == "H":
            im_path = 'healthy/'+ '_'.join(self.patient_id[idx]) + '.png'
        else:
            im_path = 'glaucoma/' + '_'.join(self.patient_id[idx]) + '.png'

        im_path_full = os.path.join(self.diag_root, im_path)

        im = preprocess_image(im_path_full, self.resize_dim, diag=False)

        seg_path = os.path.join(self.seg_root,'_'.join(self.patient_id[idx])+'_STDC-Ei4.mat')
        seg_im = loadmat(seg_path)['segmented_data']
        seg_im = imresize(seg_im, self.resize_dim, interp='nearest')
        seg_im = seg_map_convert_one_hot(seg_im, channel_first=False)

        if self.transform_seg is not None:
            seg_output = self.transform_seg([im,seg_im])
        else:
            seg_output = im, seg_im

        return seg_output

# a simple custom collate function, just to show the idea
def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]





class Beijing_Diag_Dataset(Dataset):

    '''
    :return diag_output, seg_output where diag_output is torch.cat(diag_image_array),train_y and seg_output is im, seg_im
    '''

    def __init__(self, patient_id_H, patient_id_G, balanced_patient_id, diag_root, resize_dim, transform_diag=None):

        '''
        :param seg_patient_id: must be a list with each element as e.g. ['12956', '4', 'G']
        :param diag_root: root_dir for diagnosis
        :param seg_root:  root_dir for segmented map
        :param transform: composed transformation
        '''

        self.patient_id_H = patient_id_H
        self.patient_id_G = patient_id_G
        self.patient_id = balanced_patient_id
        self.diag_root = diag_root
        self.resize_dim = resize_dim
        self.transform_diag = transform_diag

    def __len__(self):
        return len(self.patient_id)

    def __getitem__(self, idx):

        #Diagnosis data preprocess
        # slices = ["_1", "_2", "_3", "_4", "_5", "_6"]
        current_id = self.patient_id[idx]
        if current_id in self.patient_id_H:
            path = os.path.join(self.diag_root,'healthy/')
        else:
            path = os.path.join(self.diag_root, 'glaucoma/')

        # all_files_with_current_id = [filename for filename in os.listdir(path) if filename.startswith(current_id)]
        all_files_with_current_id = [filename for filename in os.listdir(path) if filename.split("_")[0]==current_id]

        slices = ['_' + a.split('_')[1] for a in sorted(all_files_with_current_id)]


        diag_image_array = []
        for s in slices:
            if current_id in self.patient_id_H:
                slice_file_name = 'healthy/'+ current_id + s +'_H.png'
            else:
                slice_file_name = 'glaucoma/'+ current_id + s +'_G.png'
            slice_path = os.path.join(self.diag_root, slice_file_name)
            im = preprocess_image(slice_path, self.resize_dim, self.transform_diag)
            diag_image_array.append(im.unsqueeze(0))
            # diag_image_array.append(im)

        if current_id in self.patient_id_H:
            train_y = 0
        else:
            train_y = 1

        diag_output = torch.cat(diag_image_array), train_y  # data to be passed to diagnosis cnn

        return diag_output


def _init_weight(modules):
    for m in modules:
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            init.xavier_uniform(m.weight.data)
            if m.bias is not None:
                init.constant(m.bias.data, 0)


def conv(dims, inplanes, outplanes, kernel_size, stride, dilation, bias):
    """convolution with flexible options"""
    padding = math.floor((dilation * (kernel_size - 1) + 2 - stride) / 2)
    if dims == 2:
        return nn.Conv2d(inplanes, outplanes, kernel_size, stride,
                         padding, dilation, bias=bias)
    elif dims == 3:
        return nn.Conv3d(inplanes, outplanes, kernel_size, stride,
                         padding, dilation, bias=bias)
    else:
        raise ValueError('dimension of conv must be 2 or 3')


def deconv(dims, inplanes, outplanes, kernel_size, stride, bias, dilation):
    """deconvolution with flexible options
    note: currently, dilation is unclear in pytorch doc,
    thus we use default dilation
    """
    padding = math.floor((kernel_size-stride+1)/2)
    if dims==2:
        return nn.ConvTranspose2d(inplanes, outplanes, kernel_size, stride,
                                  padding=padding, bias=bias) #, dilation=1)
    elif dims==3:
        return nn.ConvTranspose3d(inplanes, outplanes, kernel_size, stride,
                                  padding = padding, bias=bias) #, dilation=1)
    else:
        raise ValueError('dimension of deconv must be 2 or 3')

def batchnorm(dims, planes):
    if dims == 2:
        return nn.BatchNorm2d(planes)
    elif dims == 3:
        return nn.BatchNorm3d(planes)
    else:
        raise ValueError('dimension of batchnorm must be 2 or 3')


def maxpool(dims, kernel_size=2, stride=2):
    if dims == 2:
        return nn.MaxPool2d(kernel_size, stride)
    elif dims == 3:
        return nn.MaxPool3d(kernel_size, stride)
    else:
        raise ValueError('dimension of maxpool must be 2 or 3')


# ------------------------------------------------------------------------------
# Single Block
# ------------------------------------------------------------------------------
class SharedBlock(nn.Module):
    """
    Basic convblock for Unet:
    conv(inplane-outplane) - conv(outplane-outplane) - convconv(outplane-outplane)
    """

    def __init__(self, dims, inplanes, outplanes, kernel_size, stride, dilation, bias, relu_type):
        super(SharedBlock, self).__init__()
        self.conv1 = conv(dims, inplanes, outplanes, kernel_size, stride=1, dilation=dilation, bias=True)
        self.relu = getattr(nn, relu_type)(inplace=True)
        self.conv2 = conv(dims, outplanes, outplanes, kernel_size, stride=1, dilation=dilation, bias=True)
        self.conv3 = conv(dims, outplanes, outplanes, kernel_size, stride=1, dilation=dilation, bias=True)


    def forward(self, input):
        output = self.conv1(input)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.relu(output)
        output = self.relu(self.conv3(output))
        return output



class BasicBlock(nn.Module):
    """
    Basic convblock for Unet:
    conv(inplane-outplane) - conv(outplane-outplane) - convconv(outplane-outplane)
    """

    def __init__(self, dims, inplanes, outplanes, kernel_size, stride, dilation, bias, relu_type):
        super(BasicBlock, self).__init__()
        self.conv1 = conv(dims, inplanes, outplanes, kernel_size, stride=1, dilation=dilation, bias=True)
        self.relu = getattr(nn, relu_type)(inplace=True)
        self.conv2 = conv(dims, outplanes, outplanes, kernel_size, stride=1, dilation=dilation, bias=True)

    def forward(self, input):
        output = self.conv1(input)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.relu(output)

        return output


class ResBasicBlock(nn.Module):
    """
    Re-inplementation of residual basis convblock
    note: bias always be False"""

    def __init__(self, dims, inplanes, outplanes, kernel_size, stride, dilation, bias, relu_type):
        super(ResBasicBlock, self).__init__()
        self.dims = dims
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        self.conv1 = conv(dims, inplanes, outplanes, kernel_size, stride=1, dilation=dilation, bias=True)
        self.relu = getattr(nn, relu_type)(inplace=True)
        self.conv2 = conv(dims, outplanes, outplanes, kernel_size, stride=1, dilation=dilation, bias=True)

        self.residualsample = conv(dims, inplanes, outplanes, kernel_size=1, stride=1, dilation=1, bias=False)

    def forward(self, input):
        residual = self.residualsample(input)

        output = self.conv1(input)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.relu(output)

        output = output + residual
        return output


class UpBlock(nn.Module):
    """
    Upsampe convblock for U-net
    if option deconv, use de-conv layer
    else use simple upsample followed by a 1x1 conv layer
    """

    def __init__(self, dims, inplanes, outplanes, kernel_size, stride, bias, dilation, up_mode):
        super(UpBlock, self).__init__()
        if up_mode == 'deconv':
            self.upblock = deconv(dims, inplanes, outplanes, kernel_size, stride, bias, dilation)
        else:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            conv1x1 = conv(dims, inplanes, outplanes, kernel_size=1, stride=1, dilation=1, bias=False)
            self.upblock = nn.Sequential(upsample, conv1x1)

    def forward(self, input):
        output = self.upblock(input)
        return output


# ------------------------------------------------------------------------------
# Unet
# ------------------------------------------------------------------------------

class DownModule(nn.Module):
    def __init__(self, dims, block, inplanes, outplanes, kernel_size, stride, dilation, bias, relu_type):
        super(DownModule, self).__init__()
        self.convblock = block(dims, inplanes, outplanes, kernel_size, stride, dilation=dilation, bias=True,
                               relu_type=relu_type)
        self.pool = maxpool(dims, kernel_size=2, stride=2)

    def forward(self, input):
        before_pool = self.convblock(input)
        output = self.pool(before_pool)
        return output, before_pool


class UpModule(nn.Module):
    def __init__(self, dims, block, inplanes, outplanes, kernel_size, stride, bias, dilation, relu_type,
                 up_mode='upsample', merge_mode='concat'):
        super(UpModule, self).__init__()
        self.upblock = nn.Upsample(scale_factor=2, mode='bilinear')
        self.convblock = block(dims, inplanes, outplanes, kernel_size, stride, dilation=dilation, bias=True,
                               relu_type=relu_type)

    def forward(self, before_pool, from_up):
        from_up = self.upblock(from_up)
        x = torch.cat((from_up, before_pool), dim=1)
        output = self.convblock(x)
        return output


class Unet(nn.Module):
    def __init__(self, dims=2, inplanes=1, outplanes=8, relu_type='ELU'):
        super(Unet, self).__init__()
        # blocks
        self.block1 = DownModule(dims, SharedBlock, inplanes, outplanes=16, kernel_size=3, stride=1, dilation=1,
                                 bias=True, relu_type=relu_type)

        self.block2 = DownModule(dims, ResBasicBlock, inplanes=16, outplanes=16, kernel_size=3, stride=1, dilation=2,
                                 bias=True, relu_type=relu_type)

        self.block3 = DownModule(dims, ResBasicBlock, inplanes=16, outplanes=16, kernel_size=3, stride=1, dilation=4,
                                 bias=True, relu_type=relu_type)

        #         self.block4 = DownModule(dims, ResBasicBlock, inplanes=16, outplanes=16, kernel_size=3, stride=1, dilation=8, bias=True, relu_type=relu_type)

        self.block5 = ResBasicBlock(dims, inplanes=16, outplanes=32, kernel_size=3, stride=1, dilation=18, bias=True,
                                    relu_type=relu_type)

        #         self.block6 = UpModule(dims, ResBasicBlock, inplanes=48, outplanes=32, kernel_size=3, stride=1, dilation=8, bias=True, relu_type=relu_type)

        self.block7 = UpModule(dims, ResBasicBlock, inplanes=48, outplanes=16, kernel_size=3, stride=1, dilation=4,
                               bias=True, relu_type=relu_type)

        self.block8 = UpModule(dims, ResBasicBlock, inplanes=32, outplanes=16, kernel_size=3, stride=1, dilation=2,
                               bias=True, relu_type=relu_type)

        self.block9 = UpModule(dims, ResBasicBlock, inplanes=32, outplanes=16, kernel_size=3, stride=1, dilation=1,
                               bias=True, relu_type=relu_type)

        self.block10 = conv(dims, inplanes=16, outplanes=outplanes, kernel_size=1, stride=1, dilation=1, bias=True)

        # init weight
        _init_weight(self.modules())

    def forward(self, x):
        out1, cv1 = self.block1(x)
        out2, cv2 = self.block2(out1)
        out3, cv3 = self.block3(out2)
        #         out4, cv4 = self.block4(out3)
        out5 = self.block5(out3)

        #         out6 = self.block6(cv4, out5)

        out7 = self.block7(cv3, out5)
        out8 = self.block8(cv2, out7)
        out9 = self.block9(cv1, out8)

        out = self.block10(out9)

        return out


class CustomSoftmax(nn.Module):
    """modified Softmax
    input: N*C*H*W*(Z)
    output: exp(xi-max)/sum(exp(xi-max))
    """

    def __init__(self, dim, logit=False):
        super(CustomSoftmax, self).__init__()
        self.dim = dim
        self.softmax = nn.LogSoftmax(dim=dim) if logit else nn.Softmax(dim=dim)

    def forward(self, x):
        max, _ = torch.max(x, dim=self.dim, keepdim=True)  # DONE: check again later
        x = x - max
        return self.softmax(x)


class Seg_Model(nn.Module):
    """Model Implementation
    input: N*C*H*W
    output: N*Classes*H*W
    """

    def __init__(self):
        super(Seg_Model, self).__init__()
        self.unet = Unet()
        # softmax on classes
        self.squash = CustomSoftmax(dim=1)  # nn.Softmax(dim=1)

    def forward(self, x):
        x = self.unet(x)
        x = self.squash(x)
        return x

class Net(nn.Module):
    def __init__(self, block_shared):
        super(Net, self).__init__()
        self.block_shared = block_shared
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size= (3,3), padding=(2,2), dilation=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size= (3,3), padding=(2,2), dilation=2)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), padding=(2,2), dilation=2)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), padding=(2,2), dilation=2)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), padding=(2,2), dilation=2)
        self.conv5_drop = nn.Dropout(p=.2)
        # self.fc1 = nn.Linear(in_features=32, out_features=10)
        # self.fc2 = nn.Linear(in_features=10, out_features=2)

        # for m in self.modules():
        #
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #         #init.xavier_normal(m.weight.data, gain=nn.init.calculate_gain('relu'))
        #         init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
        #         #init.kaiming_uniform(m.weight.data)
        #         init.constant(m.bias, .1)
        #
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()




    def forward(self, x):
        x, _ = self.block_shared(x)
        x = F.avg_pool2d(x, (2,2))
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.max_pool2d(x, (2,2))
        x = F.elu(self.conv3(x))
        x = F.max_pool2d(x, (2,2))
        x = F.elu(self.conv4(x))
        x = F.max_pool2d(x, (2,2))
        x = F.elu(self.conv5(x))
        x = F.adaptive_avg_pool2d(x,output_size=1)
        x = self.conv5_drop(x)
        x = F.elu(x)
        x = x.view(-1, 32)
        # x = F.elu(self.fc1(x))
        # x = self.fc2(x)
        return x
        #return F.log_softmax(x, dim=1)


#Diagnosis Model starts here
class Net_final(nn.Module):

    def __init__(self, block_shared):
        super(Net_final, self).__init__()
        self.nn_block = Net(block_shared)
        self.fc1 = nn.Linear(in_features=32, out_features=10)
        self.fc2 = nn.Linear(in_features=10, out_features=2)

        # for m in self.modules():
        #
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #         #init.xavier_normal(m.weight.data, gain=nn.init.calculate_gain('relu'))
        #         init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
        #         #init.kaiming_uniform(m.weight.data)
        #         init.constant(m.bias, .1)
        #
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def forward(self,y):
        output = []
        for x in y:
            current_output = self.nn_block(x)
            current_output,_ = torch.max(current_output,0)
            output.append(current_output.unsqueeze(0))

        final_output = torch.cat(output)
        # final_output, _ = torch.max(final_output,0)

        x = F.elu(self.fc1(final_output))
        x = self.fc2(x)

        return x



class Combined_Model(nn.Module):

    def __init__(self):
        super(Combined_Model,self).__init__()
        self.unet = Unet()
        # softmax on classes
        self.squash = CustomSoftmax(dim=1)  # nn.Softmax(dim=1)
        #self.sharebolock = self.unet.block1
        self.diag_block = Net_final(self.unet.block1)

    def forward(self,y,diagnosis=True):

        if diagnosis:
            return self.diag_block(y)
        else:
            y = self.unet(y)
            y = self.squash(y)
            return y




class JaccardCriterion(nn.Module):
    """Jaccard Criterion for Segmentation
    jaccard_coef = [(X==Y)/(X+Y-(X==Y)] = TP/(2TP+FP+FN-TP)
    loss = 1 - jaccard_coef(input, target)
    """

    def __init__(self, dims=2):
        super(JaccardCriterion, self).__init__()
        self.eps = 1e-5

    def forward(self, output, target):
        if target.data.type() != output.data.type():
            target.data = target.data.type(output.data.type())  # should be ok when target contains binary values
        m = output * target
        m = m.sum(dim=3).sum(dim=2)
        s = output + target
        s = s.sum(dim=3).sum(dim=2)
        loss = 1. - m / (s - m + self.eps)  # size N*C
        loss = loss.mean(dim=1)
        return loss.mean()

def train(epoch, model, seg_train_loader, diag_train_loader, criterion, optimizer, writer, count_writer, alpha=.9):
    model.train()
    train_loss = 0
    train_diag_loss = 0
    train_seg_loss = 0
    count = 0
    count_diag = count
    count_seg = count

    factor = [0]
    for batch_id, diag_data in enumerate(diag_train_loader):
        diag_train, diag_target = diag_data
        # print(len(diag_train))
        # print(diag_train[0].size())
        # diag_train = torch.transpose(diag_train, 0, 1)


        if CUDA:
            diag_target = Variable(diag_target.type(torch.LongTensor).cuda())
            for i in range(len(diag_train)):
                diag_train[i] = diag_train[i].type(torch.FloatTensor).cuda()
                diag_train[i] = Variable(diag_train[i])
        else:
            diag_target = Variable(diag_target)
            for i in range(len(diag_train)):
                diag_train[i] = Variable(diag_train[i])




        # if CUDA: diag_train, diag_target = diag_train.type(torch.FloatTensor).cuda(), diag_target.type(torch.LongTensor).cuda()
        # diag_train, diag_target = Variable(diag_train), Variable(diag_target)

        # diag_output = model(diag_train)
        # loss_diag = F.cross_entropy(input=diag_output, target=diag_target)

        for seg_batch_id, seg_data in enumerate(seg_train_loader):
            seg_train, seg_target = seg_data
            if CUDA: seg_train, seg_target = seg_train.type(torch.FloatTensor).cuda(), seg_target.type(torch.FloatTensor).cuda()
            seg_train, seg_target = Variable(seg_train), Variable(seg_target)

            seg_output = model(seg_train, diagnosis=False)
            loss_seg = criterion(seg_output, seg_target)
            writer.add_scalar('data/Segmentation_Loss', loss_seg.data[0], count_writer)
            train_seg_loss +=loss_seg.data[0]
            count_seg +=1

            diag_output = model(diag_train)
            loss_diag = F.cross_entropy(input=diag_output, target=diag_target)
            writer.add_scalar('data/Diagnosis_Loss',loss_diag.data[0],count_writer)
            train_diag_loss +=loss_diag.data[0]
            count_diag +=1

            # print("Diag_loss: {} and Seg_loss: {}".format(loss_diag.data[0], loss_seg.data[0]))

            # final_loss = loss_diag + 1.6 * loss_seg

            # Adaptive change in loss

            # factor = loss_diag.data[0]/loss_seg.data[0]
            # final_loss = loss_diag + factor * loss_seg

            new_factor = loss_diag.data[0] / loss_seg.data[0]
            final_factor = alpha*factor[-1]  + (1-alpha)*new_factor
            factor.append(final_factor)
            final_loss = loss_diag + final_factor * loss_seg
            writer.add_scalar('data/Combined_Loss', final_loss.data[0], count_writer)

            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()

            train_loss += final_loss.data[0]
            count +=1

            count_writer +=1


        if batch_id % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_id * len(diag_train), len(diag_train_loader.dataset),
                       100. * batch_id / len(diag_train_loader),
                       final_loss.data[0]))

        if batch_id % 20 == 0:
            print("Diag_loss: {:.4f} and Seg_loss: {:.4f}".format(loss_diag.data[0], loss_seg.data[0]))

    writer.add_scalar('Average_loss/Combined Loss', train_loss /count, epoch)
    writer.add_scalar('Average_loss/Diagnosis Loss', train_diag_loss / count_diag, epoch)
    writer.add_scalar('Average_loss/Segmentation Loss', train_seg_loss/count_seg, epoch)

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss /count))
    print('====> Epoch: {} Average diag loss: {:.4f} and Average seg loss: {:.4f}'.format(epoch, train_diag_loss / count_diag, train_seg_loss/count_seg))

    return count_writer, train_loss /count #Required to plot the error in tensorboard

def test(model,testloader):
    model.eval()
    test_loss = 0
    correct = 0

    count = 0
    for _, (data, target) in enumerate(testloader):
        # data = torch.transpose(data, 0, 1)

        if CUDA:
            target = Variable(target.type(torch.LongTensor).cuda())
            for i in range(len(data)):
                data[i] = data[i].type(torch.FloatTensor).cuda()
                data[i] = Variable(data[i], volatile=True)
        else:
            target = Variable(target)
            for i in range(len(data)):
                data[i] = Variable(data[i], volatile=True)

        # data, target = data.type(torch.FloatTensor).cuda(), target.type(torch.LongTensor).cuda()
        # data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss = test_loss + F.cross_entropy(output, target, size_average=False).data[0]

        pred = output.data.max(1, keepdim=True)[1]
        correct = correct + pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(testloader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))

def auc_cal(epoch, model, testloader, writer):
    model.eval()
    test_loss = 0
    correct = 0

    for _, (data, target) in enumerate(testloader):
        # data = torch.transpose(data, 0, 1)

        if CUDA:
            target = Variable(target.type(torch.LongTensor).cuda())
            for i in range(len(data)):
                data[i] = data[i].type(torch.FloatTensor).cuda()
                data[i] = Variable(data[i], volatile=True)
        else:
            target = Variable(target)
            for i in range(len(data)):
                data[i] = Variable(data[i], volatile=True)

        # data, target = data.type(torch.FloatTensor).cuda(), target.type(torch.LongTensor).cuda()
        # data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # seg_output = model(data[0],diagnosis = False)

        # vutils.make_grid()
        test_loss = test_loss + F.cross_entropy(output, target, size_average=False).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct = correct + pred.eq(target.data.view_as(pred)).cpu().sum()

        output = np.exp(output.data.cpu().numpy())
        predicted = output / np.sum(output, axis=1).reshape(len(testloader.dataset),1)
        target = target.data.cpu().numpy()
        fpr, tpr, thresholds = metrics.roc_curve(1 + target, predicted[:, 1], pos_label=2)
        writer.add_pr_curve('PR curve',target,predicted[:, 1],epoch)


    test_loss /= len(testloader.dataset)
    writer.add_scalar('Average_loss/Combined Loss', test_loss)


    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))
    return metrics.auc(fpr, tpr)



if __name__ == "__main__":

    # seg_loc = "/home/atin/Glaucoma-DL/Atin/Beijing/data/segmentation"
    # root_dir = '/data1/glaucoma/Beijing_Eye_Study/rescaled_slices'
    old_root_dir = '/home/atin/DeployedProjects/data/Beijing'
    root_dir = '/home/atin/rescaled_march_2018'
    seg_loc = '/home/atin/DeployedProjects/Glaucoma-DL/Atin/Beijing/segmentation_data'


    patient_id_G = [x.split('_')[0] for x in os.listdir(root_dir + "/glaucoma/")]
    patient_id_H = [x.split('_')[0] for x in os.listdir(root_dir + "/healthy/")]

    # freq_G = Counter(patient_id_G)
    # freq_H = Counter(patient_id_H)
    #
    # print(len(freq_G))
    # G_exclude = [k for k, v in freq_G.items() if v != 6]
    # H_exclude = [k for k, v in freq_H.items() if v != 6]
    # # len(freq_G_exclude), len(freq_H_exclude) =(9, 76)
    #
    # # Remove those patients which don't have 6 slices.
    # patient_id_G = set(patient_id_G) - set(G_exclude)
    # patient_id_H = set(patient_id_H) - set(H_exclude)

    # print("Before removing segmentation images")
    # print(len(patient_id_G), len(patient_id_H))
    #
    #
    seg_patient_id = [x.split('_')[:3] for x in os.listdir(seg_loc)]
    seg_id = [p[0] for p in seg_patient_id]
    patient_id_G = set(patient_id_G) - set(seg_id)
    patient_id_H = set(patient_id_H) - set(seg_id)

    #Divide the data in train and test 120 total images in test 20 G and 100 H
    print("After removing segmentation images")
    print(len(patient_id_G), len(patient_id_H))

    # patient_id_G = set(patient_id_G)
    # patient_id_H = set(patient_id_H)

    #random.seed(231)
    # random.seed(500)
    random.seed(8956)

    # with open(os.path.join("/home/atin/Glaucoma_DL_train_test", 'test_data'), 'rb') as fp:
    #     patient_id_G_test, patient_id_H_test = pickle.load(fp)

    patient_id_G_test = random.sample(patient_id_G, 20)
    patient_id_H_test = random.sample(patient_id_H, 200)

    patient_id_G_train = list(patient_id_G.difference(patient_id_G_test))
    patient_id_H_train = list(patient_id_H.difference(patient_id_H_test))

    patient_id_G_test, patient_id_H_test = list(patient_id_G_test), list(patient_id_H_test)

    print("no. of Glaucoma patients in train {} and no. of Healthy patients in train {}".format(len(patient_id_G_train),len(patient_id_H_train)))
    print("no. of Glaucoma patients in test {} and no. of Healthy patients in test {}".format(len(patient_id_G_test),len(patient_id_H_test)))

    #Balance the data
    final_train = make_balance(patient_id_G_train, patient_id_H_train)
    final_test = patient_id_G_test + patient_id_H_test

    transform_pipeline_train = tr.Compose([tr.ToTensor(), tr.AddChannel(axis=0), tr.TypeCast('float'),
                                           # tr.RangeNormalize(0,1),
                                           tr.RandomBrightness(-.2, .2),
                                           tr.RandomGamma(.9, 1.1),
                                           tr.RandomFlip(),
                                           tr.RandomAffine(rotation_range=5, translation_range=0.2,
                                                           zoom_range=(0.9, 1.1))])

    transform_pipeline_test = tr.Compose([tr.ToTensor(), tr.AddChannel(axis=0), tr.TypeCast('float')
                                          # tr.RangeNormalize(0, 1)
                                          ])

    diag_train_data = Beijing_Diag_Dataset(patient_id_H_train, patient_id_G_train, final_train, root_dir, (96, 288),
                                           transform_diag= transform_pipeline_train)

    # diag_train_data = Beijing_Diag_Dataset(patient_id_H_train, patient_id_G_train, final_train, root_dir, (96, 288),
    #                                        transform_diag=T.Compose([Horizontalflip(),RandomRotate(angle_range=(0, 5), prob=0,mode="constant"),
    #                                                                  Translation(h_shift_range=(0, 4),w_shift_range=(0, 8),
    #                                                                              prob=0,mode="constant"),
    #                                                                  ToTensor()]))
    #After adding new transformation for diagnosis namely nonlinearintensity shift and addgaussian
    # diag_train_data = Beijing_Diag_Dataset(patient_id_H_train, patient_id_G_train, final_train, root_dir, (96, 288),
    #                                        transform_diag=T.Compose([Horizontalflip(),
    #                                                                  IntensityNonliearShift(),
    #                                                                  AddGaussian(ismulti=True),
    #                                                                  AddGaussian(ismulti=False),
    #                                                                  RandomRotate(angle_range=(0, 5), prob=0,
    #                                                                               mode="constant"),
    #                                                                  Translation(h_shift_range=(0, 4),
    #                                                                              w_shift_range=(0, 8),
    #                                                                              prob=0, mode="constant"),
    #                                                                  ToTensor()]))

    diag_test_data = Beijing_Diag_Dataset(patient_id_H_test, patient_id_G_test, final_test, root_dir, (96,288),
                                          transform_diag= transform_pipeline_test)



    dark_mask = create_dark_mask()
    indices_x_clipped, indices_y_clipped = create_elastic_indices()
    seg_train_data = Beijing_Seg_Dataset(seg_patient_id, old_root_dir, seg_loc, (96, 288),
                                         train_transform(dark_mask, indices_x_clipped, indices_y_clipped))

    # diag_train_dataloader = DataLoader(diag_train_data, batch_size=16, num_workers=10, shuffle=True)

    diag_train_dataloader = DataLoader(diag_train_data, batch_size=16, num_workers=20, shuffle=True, collate_fn=my_collate, pin_memory=True)
    testloader = DataLoader(diag_test_data, batch_size=len(final_test), num_workers=20, shuffle=True, collate_fn=my_collate, pin_memory=True)

    seg_train_loader = DataLoader(seg_train_data, batch_size=16, num_workers=10, shuffle=True)

    model = Combined_Model()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    nb_params = sum([np.prod(p.size()) for p in model_parameters])
    print("no. of trainable parametes is: {}".format((nb_params)))

    if CUDA: model.cuda()

    criterion = JaccardCriterion()
    if CUDA: criterion.cuda()

    nb_epoch = 200
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0001)
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01/nb_epoch)

    root_log_dir = "/home/atin/DeployedProjects/Glaucoma-DL-log"
    log_dir = os.path.join(root_log_dir, "co_training_with_any_slices" + str(datetime.datetime.now()))
    os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)

    test_auc = []
    count = 0
    for epoch in range(1, nb_epoch + 1):
        # print(count)
        new_count, _ = train(epoch, model, seg_train_loader, diag_train_dataloader, criterion, optimizer, writer, count, alpha=.4)
        count = new_count
        test_auc.append(auc_cal(epoch, model, testloader, writer))
        print("Test AUC:", test_auc[-1])
        writer.add_scalar('Average_loss_per_epoch/AUC', test_auc[-1], epoch)
    writer.close()

    print("test_auc over different epochs:")
    print(test_auc)

































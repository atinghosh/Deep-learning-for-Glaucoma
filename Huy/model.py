import math
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torchvision as vision
import torchvision.models as M
from torch.autograd import Function
from torch.autograd import Variable

#------------------------------------------------------------------------------ 
#base functions
#------------------------------------------------------------------------------ 
def _init_weight(modules):
    for m in modules:
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            init.xavier_uniform(m.weight.data)
            if m.bias is not None:
                init.constant(m.bias.data, 0)

def conv(dims, inplanes, outplanes, kernel_size, stride, dilation, bias):
    """convolution with flexible options"""
    padding = math.floor((dilation*(kernel_size-1)+2-stride)/2)
    if dims ==2:
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
    if dims==2:
        return nn.BatchNorm2d(planes)
    elif dims==3:
        return nn.BatchNorm3d(planes)
    else:
        raise ValueError('dimension of batchnorm must be 2 or 3')

def maxpool(dims, kernel_size=2, stride=2):
    if dims==2:
        return nn.MaxPool2d(kernel_size, stride)
    elif dims==3:
        return nn.MaxPool3d(kernel_size, stride)
    else:
        raise ValueError('dimension of maxpool must be 2 or 3')

#------------------------------------------------------------------------------ 
#Single Block
#------------------------------------------------------------------------------ 
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
    def __init__(self,dims, inplanes, outplanes, kernel_size, stride, bias, dilation, up_mode):
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

#------------------------------------------------------------------------------ 
#Unet
#------------------------------------------------------------------------------ 

class DownModule(nn.Module):
    def __init__(self, dims, block, inplanes, outplanes, kernel_size, stride, dilation, bias, relu_type):
        super(DownModule, self).__init__()
        self.convblock = block(dims, inplanes, outplanes, kernel_size, stride, dilation=dilation, bias=True, relu_type=relu_type)
        self.pool = maxpool(dims, kernel_size=2, stride=2)
    
    def forward(self, input):
        before_pool = self.convblock(input)
        output = self.pool(before_pool)
        return output, before_pool

class UpModule(nn.Module):
    def __init__(self, dims, block, inplanes, outplanes, kernel_size, stride, bias, dilation, relu_type, up_mode='upsample', merge_mode='concat'):
        super(UpModule, self).__init__()
        self.upblock = nn.Upsample(scale_factor=2, mode='bilinear')
        self.convblock = block(dims, inplanes, outplanes, kernel_size, stride, dilation=dilation, bias=True, relu_type=relu_type)
        
    def forward(self, before_pool, from_up):
        from_up = self.upblock(from_up)
        x = torch.cat((from_up, before_pool), dim=1)
        output = self.convblock(x)
        return output
        
class Unet(nn.Module):
    def __init__(self, depth, dims, block, inplanes, outplanes, planes=64, 
                 kernel_size=3, stride=1, dilation=1, bias=True,
                 relu_type='ReLU', up_mode='upsample', merge_mode='concat', planes_expansion=1):
        super(Unet, self).__init__()
        self.depth = depth
        self.dims = dims
        self.block = block
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.bias = bias
        self.relu_type = relu_type
        self.up_mode = up_mode
        self.merge_mode = merge_mode
        self.planes_expansion = planes_expansion
        self.expansion = planes_expansion**(depth-1)
        
        #blocks
        self.block1 = DownModule(dims, BasicBlock, inplanes, outplanes=16, kernel_size=3, stride=1, dilation=1, bias=True, relu_type=relu_type)
        
        self.block2 = DownModule(dims, block, inplanes=16, outplanes=16, kernel_size=3, stride=1, dilation=2, bias=True, relu_type=relu_type)
        
        self.block3 = DownModule(dims, block, inplanes=16, outplanes=16, kernel_size=3, stride=1, dilation=4, bias=True, relu_type=relu_type)
        
#         self.block4 = DownModule(dims, block, inplanes=16, outplanes=16, kernel_size=3, stride=1, dilation=8, bias=True, relu_type=relu_type)
        
        self.block5 = block(dims, inplanes=16, outplanes=32, kernel_size=3, stride=1, dilation=18, bias=True, relu_type=relu_type)
        
#         self.block6 = UpModule(dims, block, inplanes=48, outplanes=32, kernel_size=3, stride=1, dilation=8, bias=True, relu_type=relu_type)
        
        self.block7 = UpModule(dims, block, inplanes=48, outplanes=16, kernel_size=3, stride=1, dilation=4, bias=True, relu_type=relu_type)
        
        self.block8 = UpModule(dims, block, inplanes=32, outplanes=16, kernel_size=3, stride=1, dilation=2, bias=True, relu_type=relu_type)
        
        self.block9 = UpModule(dims, block, inplanes=32, outplanes=16, kernel_size=3, stride=1, dilation=1, bias=True, relu_type=relu_type)
        
        self.block10 = conv(dims, inplanes=16, outplanes=outplanes, kernel_size=1, stride=1, dilation=1, bias=True)
        
        #init weight
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
        max, _ = torch.max(x, dim=self.dim, keepdim=True) #DONE: check again later
        x = x - max
        return self.softmax(x)

class Model(nn.Module):
    """Model Implementation
    input: N*C*H*W
    output: N*Classes*H*W
    """
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.unet = Unet(depth=opt.depth, dims=opt.dims, block=opt.block,
                          inplanes=opt.inplanes, outplanes=opt.nClasses, planes=opt.planes,
                          dilation=opt.dilation, bias=opt.bias, relu_type=opt.relu_type,
                          up_mode=opt.up_mode, merge_mode=opt.merge_mode, planes_expansion=1)
        #softmax on classes
        self.squash = CustomSoftmax(dim=1) #nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.unet(x)
        x = self.squash(x)
        return x

#------------------------------------------------------------------------------ 
#Criterion
#------------------------------------------------------------------------------ 

class JaccardCriterion(nn.Module):
    """Jaccard Criterion for Segmentation
    jaccard_coef = [(X==Y)/(X+Y-(X==Y)] = TP/(2TP+FP+FN-TP)
    loss = 1 - jaccard_coef(input, target)
    """
    def __init__(self, dims=2):
        super(JaccardCriterion, self).__init__()
        self.eps = 1e-5
        self.dims = dims
        self.size_average = size_average #if average: mean over minibatch
        self.reduce = reduce
        
    def forward(self, output, target):
        if target.data.type() != output.data.type():
            target.data = target.data.type(output.data.type()) #should be ok when target contains binary values
        m = output * target
        m = m.sum(dim=3).sum(dim=2)
        s = output + target
        s = s.sum(dim=3).sum(dim=2)
        loss = 1. - m / (s - m + self.eps)#size N*C
        loss = loss.mean(dim=1)
        return loss.mean()

import numpy as np
import scipy
import scipy.ndimage
from scipy import signal
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import collections
from PIL import Image
import numbers
import torch

try:
    import accimage
except ImportError:
    accimage = None
import scipy.io as sio

__author__ = "Wei OUYANG"
__license__ = "GPL"
__version__ = "0.1.0"
__status__ = "Development"


def center_crop(x, center_crop_size):
    assert x.ndim == 3
    centerw, centerh = x.shape[1] // 2, x.shape[2] // 2
    halfw, halfh = center_crop_size[0] // 2, center_crop_size[1] // 2
    return x[:, centerw - halfw:centerw + halfw, centerh - halfh:centerh + halfh]

def to_tensor(x):
    x = np.clip(x, 0., 1.)
    x = x.transpose((2, 0, 1))
#     x = torch.from_numpy(x.transpose((2, 0, 1)).copy())
    return torch.from_numpy(x.copy()).float()

#------------------------------------------------- @huy add to manipulate target
def to_tensor_target(x):
    x = np.clip(x, 0., 1.)
    x = np.rint(x)
    x = x.transpose((2, 0, 1))
    return torch.from_numpy(x.copy()).float()

class Merge(object):
    """Merge a group of images
    """

    def __init__(self, axis=-1):
        self.axis = axis

    def __call__(self, images):
        if isinstance(images, collections.Sequence) or isinstance(images, np.ndarray):
            assert all([isinstance(i, np.ndarray)
                        for i in images]), 'only numpy array is supported'
            shapes = [list(i.shape) for i in images]
            for s in shapes:
                s[self.axis] = None
            assert all([s == shapes[0] for s in shapes]
                       ), 'shapes must be the same except the merge axis'
            return np.concatenate(images, axis=self.axis)
        else:
            raise Exception("obj is not a sequence (list, tuple, etc)")

class Split(object):
    """Split images into individual arraies
    """

    def __init__(self, *slices, **kwargs):
        assert isinstance(slices, collections.Sequence)
        slices_ = []
        for s in slices:
            if isinstance(s, collections.Sequence):
                slices_.append(slice(*s))
            else:
                slices_.append(s)
        assert all([isinstance(s, slice) for s in slices_]
                   ), 'slices must be consist of slice instances'
        self.slices = slices_
        self.axis = kwargs.get('axis', -1)

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            ret = []
            for s in self.slices:
                sl = [slice(None)] * image.ndim
                sl[self.axis] = s
                ret.append(image[sl])
            return ret
        else:
            raise Exception("obj is not an numpy array")

class MaxScaleNumpy(object):
    """scale with max and min of each channel of the numpy array i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, range_min=0.0, range_max=1.0):
        self.scale = (range_min, range_max)

    def __call__(self, image):
        mn = image.min(axis=(0, 1))
        mx = image.max(axis=(0, 1))
        if mx == mn:
            return image
        return self.scale[0] + (image - mn) * (self.scale[1] - self.scale[0]) / (mx - mn)

class NormalizeNumpy(object):
    """Normalize each channel of the numpy array i.e.
    channel = (channel - mean) / std
    """

    def __call__(self, image):
        image -= image.mean(axis=(0, 1))
        s = image.std(axis=(0, 1))
        s[s == 0] = 1.0
        image /= s
        return image

class RandomRotate(object):
    """Rotate a PIL.Image or numpy.ndarray (H x W x C) randomly
    """

    def __init__(self, angle_range=(-15.0, 15.0), axes=(0, 1), mode='reflect', random_state=np.random):
        assert isinstance(angle_range, tuple)
        self.angle_range = angle_range
        self.random_state = random_state
        self.axes = axes
        self.mode = mode

    def __call__(self, image):
        if self.random_state.uniform() > 0.5:
            angle = self.random_state.uniform(
                self.angle_range[0], self.angle_range[1])
            if isinstance(image, np.ndarray):
                mi, ma = image.min(), image.max()
                image = scipy.ndimage.interpolation.rotate(
                    image, angle, reshape=False, axes=self.axes, mode=self.mode)
                return np.clip(image, mi, ma)
            elif isinstance(image, Image.Image):
                return image.rotate(angle)
            else:
                raise Exception('unsupported type')
        else:
            return image

class EnhancedCompose(object):
    """Composes several transforms together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> transforms.Compose([
                    transforms.CenterCrop(10),
                    transforms.ToTensor(),
        >>> ])
        >>> transform = EnhancedCompose([
                    Merge(),              # merge input and target along the channel axis
                    ElasticTransform(),
                    RandomRotate(),
                    Split([0,1],[1,2]),  # split into 2 images
                    [CenterCropNumpy(size=input_shape), CenterCropNumpy(size=target_shape)],
                    [NormalizeNumpy(), None],
                    [Lambda(to_tensor), Lambda(to_tensor)]
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            if isinstance(t, collections.Sequence):
                assert isinstance(img, collections.Sequence) and len(img) == len(
                    t), "size of image group and transform group does not fit"
                tmp_ = []
                for i, im_ in enumerate(img):
                    if callable(t[i]):
                        tmp_.append(t[i](im_))
                    else:
                        tmp_.append(im_)
                img = tmp_
            elif callable(t):
                img = t(img)
            elif t is None:
                continue
            else:
                raise Exception('unexpected type')
        return img

#------------------------------------------------------------------------------ 
#NUS-Glaucoma
#------------------------------------------------------------------------------ 
def randomlyFlip(images):
    """Randomly left-right flip numpy.ndarray(HxWxC)
    """
    if not isinstance(images, list):
        raise ValueError('input must be list of images')
    ax = images[0]
    ay = images[1]
    if np.random.randint(0, 1):
        ax = ax[:, ::-1, :]
        ay = ay[:, ::-1, :]
    return [ax, ay]

def intensityNonliearShift(ax):
    """Intensity nonlinear shift for numpy.ndarray(HxWx1)
    """
    p = np.random.uniform(low=0.6, high=1.4)
    ax[:,:,0] = ax[:,:,0]**p
    a = np.random.uniform(low=0, high=0.1)
    b = np.random.uniform(low=0, high=0.1)
    ax[:,:,0] = -a + (1+a+b) * ax[:,:,0]
    ax = np.clip(ax, 0., 1.)
    return ax

def addBlackBox(ax, dark_mask, black_dx = 60, black_dy = 20):
    """Randomly adding black box to numpy.ndarry(HxWx1)
    dark_mask is created once at beginning of epoch
    """
    for k in range(20):
        black_x, black_y = np.random.randint(0, ax.shape[0] -  black_dx), np.random.randint(0, ax.shape[1] -  black_dy)
        #ax[black_x:(black_x+black_dx), black_y:(black_y+black_dy), 0] = 0
        intensity_dark = np.random.uniform(low=0.2, high=0.8)
        window_to_darken = ax[black_x:(black_x+black_dx), black_y:(black_y+black_dy), 0]
        ax[black_x:(black_x+black_dx), black_y:(black_y+black_dy), 0] = window_to_darken * (intensity_dark + (1.-intensity_dark)*dark_mask)
    return ax

def elasticDeformation(images, indices_x_clipped, indices_y_clipped):
    """elastic deformation for numpy.ndarray(HxWx1)
    indices_x_clipped, indices_y_clipped is create once at begining of epoch
    """
    if not isinstance(images, list):
        raise ValueError('input must be list of images')
    ax = images[0]
    ay = images[1]
    ax[:,:,0] = ax[indices_y_clipped.astype(int),indices_x_clipped.astype(int),0]
    for k in range(ay.shape[2]):
        ay[:,:,k] = ay[indices_y_clipped.astype(int),indices_x_clipped.astype(int),k]
    return [ax, ay]

def addGaussian(ax, ismulti):
    """Add Gaussian with noise
    input is numpy array H*W*C
    """
    shape = (96, 288) #ax.shape[:2]
    intensity_noise = np.random.uniform(low=0, high=0.1)
    if ismulti:
        ax[:,:,0] = ax[:,:,0]*(1+ intensity_noise*np.random.normal(loc=0, scale=1, size=shape[0]*shape[1]).reshape(shape[0],shape[1]))
    else:
        ax[:,:,0] = ax[:,:,0]  + intensity_noise*np.random.normal(loc=0, scale=1, size=shape[0]*shape[1]).reshape(shape[0],shape[1])
    return ax

#Classes
class AddBlackBox(object):
    """Manipulate method addBlackBox
    """
    def __init__(self, dark_mask):
        self.dark_mask = dark_mask
    
    def __call__(self, img):
        return addBlackBox(img, self.dark_mask)

class ElasticDeformation(object):
    """Manipulate method elasticDeformation
    """
    def __init__(self, indices_x_clipped, indices_y_clipped):
        self.indices_x_clipped = indices_x_clipped
        self.indices_y_clipped = indices_y_clipped
    
    def __call__(self, imgs):
        return elasticDeformation(imgs, self.indices_x_clipped, self.indices_y_clipped)

class AddGaussian(object):
    """Manipulate method addGaussian
    """
    def __init__(self, ismulti=True):
        self.ismulti = ismulti
    
    def __call__(self, img):
        return addGaussian(img, self.ismulti)

#------------------------------------------------------------------------------ 
# end of @huy
#------------------------------------------------------------------------------ 

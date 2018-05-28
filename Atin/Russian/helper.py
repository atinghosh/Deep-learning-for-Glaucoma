import numpy as np
import scipy
import scipy.ndimage
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.interpolation import shift
import collections
from PIL import Image
import numbers
import torch




class RandomRotate(object):
    """Rotate a PIL.Image or numpy.ndarray (H x W x C) randomly
    """

    def __init__(self, angle_range=(0.0, 360.0), prob=.5, axes=(0, 1), mode='reflect', random_state=np.random):
        assert isinstance(angle_range, tuple)
        self.angle_range = angle_range
        self.random_state = random_state
        self.axes = axes
        self.mode = mode
        self.prob = prob

    def __call__(self, image):

        if np.random.binomial(1, self.prob) == 1:
            return image
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


class Horizontalflip(object):
    """It flips the image horizontally with a given probability
    """

    def __init__(self, prob=.5):
        self.prob = prob

    def __call__(self, image):

        if np.random.binomial(1, self.prob) == 1:
            return np.fliplr(image).copy()
        else:
            return image


class Translation(object):
    """It does horizontal and vertical translation with h_shift_range(tuple) and w_shift_range(tuple)
    """

    def __init__(self, h_shift_range, w_shift_range, prob=.2, random_state=np.random, mode = "nearest"):
        self.h_shift_range = h_shift_range
        self.w_shift_range = w_shift_range
        self.random_state = random_state
        self.prob = prob
        self.mode = mode

    def __call__(self, image):
        h_shift = self.random_state.uniform(self.h_shift_range[0], self.h_shift_range[1])
        w_shift = self.random_state.uniform(self.w_shift_range[0], self.w_shift_range[1])

        if np.random.binomial(1, self.prob) == 1:
            return image

        return shift(image, (h_shift, w_shift, 0), mode= self.mode)


class IntensityNonliearShift():

    def __call__(self, image):
        p = np.random.uniform(low=0.6, high=1.4)
        image[:, :, 0] = image[:, :, 0] ** p
        a = np.random.uniform(low=0, high=0.1)
        b = np.random.uniform(low=0, high=0.1)
        image[:, :, 0] = -a + (1 + a + b) * image[:, :, 0]
        image = np.clip(image, 0., 1.)
        return image


def addGaussian(ax, ismulti):
    """Add Gaussian with noise
    input is numpy array H*W*C
    """
    shape = (96, 288) #ax.shape[:2]
    intensity_noise = np.random.uniform(low=0, high=0.05)
    if ismulti:
        ax[:,:,0] = ax[:,:,0]*(1+ intensity_noise*np.random.normal(loc=0, scale=1, size=shape[0]*shape[1]).reshape(shape[0],shape[1]))
    else:
        ax[:,:,0] = ax[:,:,0]  + intensity_noise*np.random.normal(loc=0, scale=1, size=shape[0]*shape[1]).reshape(shape[0],shape[1])
    return ax


class AddGaussian(object):
    """Manipulate method addGaussian
    """

    def __init__(self, ismulti=True):
        self.ismulti = ismulti

    def __call__(self, img):
        return addGaussian(img, self.ismulti)



class ToTensor(object):
    """Convert ndarrays in sample to Tensors. numpy image: H x W x C, torch image: C X H X W
    """

    def __call__(self, image, invert_arrays=True):

        if invert_arrays:
            image = image.transpose((2, 0, 1))

        return torch.from_numpy(image)


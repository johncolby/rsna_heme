
import glob
import numpy as np
import os
import pydicom

from matplotlib import pyplot as plt

class Dicom:
    def __init__(self, path):
        self.path = path
        self.dcm = pydicom.read_file(self.path)

    def _rescale(self, img):
        slope = self.dcm.RescaleSlope
        intercept = self.dcm.RescaleIntercept
        return img * slope + intercept

    def _parse_wl(self, field):
        x = getattr(self.dcm, field)
        return int(x[0]) if isinstance(x, pydicom.multival.MultiValue) else int(x)

    def _window(self, img, center=None, width=None):
        # import pdb; pdb.set_trace()
        center = center or self._parse_wl('WindowCenter')
        width = width or self._parse_wl('WindowWidth')
        img_min = center - width // 2
        img_max = center + width // 2
        img[img < img_min] = img_min
        img[img > img_max] = img_max
        return img

    def _norm(self, img):
        img -= img.min()
        img *= 255.0 / img.max()
        img = img.astype(np.uint8)
        return img

    def img_for_plot(self, **kwargs):
        img = self.dcm.pixel_array
        img = self._rescale(img)
        img = self._window(img, **kwargs)
        img = self._norm(img)
        return img

    def plot(self, **kwargs):
        img = self.img_for_plot(**kwargs)
        plt.imshow(img, cmap='gray')
        plt.axis('off')

    def img_for_plot3(self, wl):
        img1 = self.img_for_plot(center=wl[0][0], width=wl[0][1])
        img2 = self.img_for_plot(center=wl[1][0], width=wl[1][1])
        img3 = self.img_for_plot(center=wl[2][0], width=wl[2][1])
        img = np.stack([img1, img2, img3], axis=2)
        return img

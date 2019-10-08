
import glob
from matplotlib import pyplot as plt
import os
import pydicom

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
        return int(x[0]) if len(x) > 1 else int(x)

    def _window(self, img, center=None, width=None):
        # import pdb; pdb.set_trace()
        center = center or self._parse_wl('WindowCenter')
        width = width or self._parse_wl('WindowWidth')
        img_min = center - width // 2
        img_max = center + width // 2
        img[img < img_min] = img_min
        img[img > img_max] = img_max
        return img

    def img_for_plot(self, **kwargs):
        img = self.dcm.pixel_array
        img = self._rescale(img)
        img = self._window(img, **kwargs)
        return img

    def plot(self, **kwargs):
        img = self.img_for_plot(**kwargs)
        plt.imshow(img, cmap='gray')
        plt.axis('off')

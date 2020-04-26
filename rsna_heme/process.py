import argparse
import logging
import mxnet as mx
import os
import pandas as pd
import pickle
import re
import requests

from radstudy import RadStudy

from . import dicom
from . import labels
from . import transforms

class HemeStudy(RadStudy):
    def __init__(self, acc='', zip_path='', model_path='', wl=[(40, 80), (80, 200), (40, 380)]):
        super().__init__(acc, zip_path, model_path)
        self.app_name = 'heme' 
        self.wl = wl
        self.channels = 'axial CT'
        self.series_picks = pd.DataFrame({'class': ['axial CT'], 'prob': '', 'SeriesNumber': 2, 'series': ''})

    def process(self, endpoint, save=True):
        self.series_picks.series = self.series_to_path(2)
        dir_series = self.series_picks.series[0]
        dcm_names = os.listdir(dir_series)
        dcm_names = sorted(dcm_names, key = lambda x: int(re.sub(".*\.(.*)\.dcm", "\\1", x)))
        probs_all = []
        for dcm_name in dcm_names:
            dcm_path = os.path.join(dir_series, dcm_name)
            data = self._load_dcm(dcm_path)
            data_str = pickle.dumps(data[0][0])
            prob = requests.post(endpoint, files = {'data': data_str}).json()
            probs_all.append(prob)
        probs = pd.DataFrame(probs_all, columns=labels.heme_types)
        if save is True:
            probs.to_csv(os.path.join(self.dir_tmp, 'output', 'probs.csv'), index=False)
        else:
            return probs

    def _load_dcm(self, dcm_path):
        dcm = dicom.Dicom(dcm_path)
        img = dcm.img_for_plot3(self.wl)
        img, _ = transforms.common_transform(mx.nd.array(img), 0)
        img  = img.flip(axis=2)
        data = mx.gluon.data.SimpleDataset([(img, 0)])
        data = data.transform_first(transforms.train_transform)
        return data

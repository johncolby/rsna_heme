import mxnet as mx
import numpy as np
import os
import pandas as pd
import pydicom

from mxnet.gluon.data.vision import transforms
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from .labels import read_labels

def pack_rec(base_dir, mode, out_dir = None):
    if out_dir == None:
        out_dir = base_dir
    dcm_dir = os.path.join(base_dir, f'stage_1_{mode}_images')

    # Load class labels
    if mode == 'train':
        df = read_labels(base_dir)
    elif mode == 'test':
        dcm_names = sorted([os.path.splitext(x)[0] for x in os.listdir(dcm_dir)])
        df = pd.DataFrame({'ID': dcm_names})
        df.set_index('ID', inplace = True)

    # Open recordIO file for writing
    rec_path = os.path.join(out_dir, mode + '.rec')
    idx_path = os.path.join(out_dir, mode + '.idx')
    record = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'w')

    # Loop over subjects
    i = 0
    for ID, row in tqdm(df.iterrows(), total=len(df)):
        # Get DICOM path
        dcm_name = ID + '.dcm'
        dcm_path = os.path.join(dcm_dir, dcm_name)

        # Load DICOM and extract image data
        dcm = Dicom(dcm_path)
        img = dcm.img_for_plot(center = 40, width = 80)

        # Generate recordIO header
        if mode == 'train':
            label = row.array
        else:
            label = 0
        header = mx.recordio.IRHeader(0, label, i, 0)

        # Pack data into binary recordIO format as compressed jpg
        img_packed = mx.recordio.pack_img(header, img)

        # Append record to recordIO file
        record.write_idx(i, img_packed)
        i += 1
    record.close()

class CVSampler(mx.gluon.data.sampler.Sampler):
    def __init__(self, groups, n_splits, i_fold, mode = 'train', shuffle = True, seed = 1):
        self.n_splits = n_splits
        self.seed = seed
        self.groups = groups
        self.folds = self.get_folds()
        self._indices = self.folds[i_fold][mode]
        self._shuffle = shuffle
        self._length = len(self._indices)
    def __iter__(self):
        if self._shuffle:
            np.random.shuffle(self._indices)
        return iter(self._indices)
    def __len__(self):
        return self._length
    def get_folds(self):
        skf = StratifiedKFold(n_splits = self.n_splits, shuffle = True, random_state = self.seed)
        folds = [{'train': tr, 'test': te} for tr, te in skf.split(X = np.zeros(len(self.groups)), y = self.groups)]
        return folds

train_transform = transforms.Compose([transforms.Resize(512),
                                      transforms.RandomFlipLeftRight(),
                                      transforms.ToTensor()])

val_transform   = transforms.Compose([transforms.Resize(512),
                                      transforms.ToTensor()])

def save_params(net, best_metric, current_metric, epoch, save_interval, prefix):
    """Logic for if/when to save/checkpoint model parameters"""
    if current_metric < best_metric:
        best_metric = current_metric
        net.save_parameters('{:s}_best.params'.format(prefix, epoch, current_metric))
        with open(prefix+'_best.log', 'a') as f:
            f.write('\n{:04d}:\t{:.4f}'.format(epoch, current_metric))
    if save_interval and (epoch + 1) % save_interval == 0:
        net.save_parameters('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_metric))
    return best_metric

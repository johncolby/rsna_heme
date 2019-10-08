import mxnet as mx
import numpy as np

from mxnet.gluon.data.vision import transforms

def common_transform(data, label):
    data = mx.image.imresize(data, 512, 512)
    return data, label

train_transform = transforms.Compose([transforms.RandomFlipLeftRight(),
                                      transforms.ToTensor()])

val_transform   = transforms.Compose([transforms.ToTensor()])
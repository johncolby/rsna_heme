import mxnet as mx
import numpy as np

from mxnet.gluon.data.vision import transforms

def common_transform(data, label):
    data = mx.image.imresize(data, 512, 512)
    # data = mx.nd.transpose(data, (2,0,1))
    # data = data.astype(np.float32) / 255
    label = label
    return data, label

train_transform = transforms.Compose([transforms.RandomFlipLeftRight(),
                                      transforms.ToTensor()])

val_transform   = transforms.Compose([transforms.ToTensor()])
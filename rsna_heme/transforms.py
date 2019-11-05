import mxnet as mx

from mxnet.gluon.data.vision import transforms

def common_transform(data, label):
    data = mx.image.imresize(data, 512, 512)
    return data, label

train_transform = transforms.Compose([transforms.RandomFlipLeftRight(),
                                      transforms.RandomResizedCrop(size=(512, 512), scale=(0.8, 1), ratio=(0.8, 1.2)),
                                      transforms.ToTensor()])

val_transform   = transforms.Compose([transforms.ToTensor()])

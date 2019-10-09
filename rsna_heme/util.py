import datetime

from matplotlib import pyplot as plt

def get_time():
    return f'{datetime.datetime.now():%Y-%m-%d_%H%M%S}'

def plt_tensor(data):
    plt.imshow(data.transpose([1,2,0]).asnumpy())

import datetime
import subprocess

from matplotlib import pyplot as plt

def get_time():
    return f'{datetime.datetime.now():%Y-%m-%d_%H%M%S}'

def get_git_commit():  
    return subprocess.check_output(["git", "describe", "--always"]).strip().decode('utf-8')

def plt_tensor(data):
    plt.imshow(data.transpose([1,2,0]).asnumpy())

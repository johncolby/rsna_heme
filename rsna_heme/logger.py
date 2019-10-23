import datetime
import logging
import os

from mxboard import SummaryWriter

from . import util

class Logger:
    def __init__(self, log_dir, log_name):
        self._log_dir = log_dir
        self._log_name = log_name
        self._tic = datetime.datetime.now()

        logging.basicConfig()
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self._get_log_fh()
        self._init_messages()

    def _get_log_fh(self):
        self._log_path = os.path.join(self._log_dir, f'{self._log_name}.log')
        if not os.path.exists(self._log_dir):
            os.makedirs(self._log_dir)
        fh = logging.FileHandler(self._log_path)
        self.logger.addHandler(fh)

    def log(self, msg):
        self.logger.info(msg)

    def _init_messages(self):  
        self.log(f'Start time: {self._tic}')
        self.log(f'Log path: {self._log_path}')      
        self.log(f'Git commit hash: {util.get_git_commit()}')
        self.log('')

    def tb_setup(self):
        # Setup mxboard logging
        tb_dir = os.path.join(self._log_dir, f'{self._log_name}_tb')
        if not os.path.exists(tb_dir):
            os.makedirs(tb_dir)
        self.sw = SummaryWriter(logdir=tb_dir, flush_secs=60, verbose=False)

    def close(self):
        if hasattr(self, 'sw'):
            self._tb_close()
        self.log('')
        toc = datetime.datetime.now()
        self.log(f'End time: {toc}')
        self.log(f'Elapsed time: {toc - self._tic}')

    def _tb_close(self):
        self.sw.export_scalars(f'{os.path.splitext(self._log_path)[0]}_scalar_dict.json')
        self.sw.close()

def log_epoch_hooks(logger, epoch, metrics):
    """Epoch logging"""
    logger.log('E %d | Tacc: %.3f, Tloss: %.3f | Vacc: %.3f , Vloss: %.3f' %
             (epoch, metrics[0], metrics[1], metrics[2], metrics[3]))
    return

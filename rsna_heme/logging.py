import logging
from mxboard import SummaryWriter
import numpy as np
import os

def start_logger(args):
    """Start logging utilities for stdout, log files, and mxboard"""
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = args.save_prefix + '_train.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    logger.info(args)
    logger.info('Start training from [Epoch {}]'.format(args.start_epoch))

    # Setup mxboard logging
    tb_dir = args.save_prefix + '_tb'
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)
    sw = SummaryWriter(logdir=tb_dir, flush_secs=60, verbose=False)

    return logger, sw

def log_epoch_hooks(epoch, metrics, logger, sw):
    """Epoch logging"""
    # DSCs = np.array([v['DSC'] for k,v in metrics.items()])
    # DSC_avg = DSCs.mean()
    # logger.info('E %d | loss %.4f | ET %.4f | TC %.4f | WT %.4f | Avg %.4f'%((epoch, train_loss) + tuple(DSCs) + (DSC_avg, )))
    # sw.add_scalar(tag='Dice', value=('Val ET', DSCs[0]), global_step=epoch)
    # sw.add_scalar(tag='Dice', value=('Val TC', DSCs[1]), global_step=epoch)
    # sw.add_scalar(tag='Dice', value=('Val WT', DSCs[2]), global_step=epoch)
    # sw.add_scalar(tag='Dice', value=('Val Avg', DSCs.mean()), global_step=epoch)
    logger.info('Epoch %d | Tacc: %.3f, Tloss: %.6f | Vacc: %.3f , Vloss: %.6f' %
             (epoch, metrics[0], metrics[1], metrics[2], metrics[3]))
    return
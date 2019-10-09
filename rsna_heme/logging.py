import logging
import numpy as np
import os
import subprocess

from mxboard import SummaryWriter

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
    hash = subprocess.check_output(["git", "describe", "--always"]).strip().decode('utf-8')
    logger.info(f'Git commit hash: {hash}')
    logger.info(args)
    logger.info(f'Start training from [Epoch {args.start_epoch}]')

    # Setup mxboard logging
    tb_dir = args.save_prefix + '_tb'
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)
    sw = SummaryWriter(logdir=tb_dir, flush_secs=60, verbose=False)

    return logger, sw

def log_epoch_hooks(epoch, metrics, logger, sw):
    """Epoch logging"""
    logger.info('E %d | Tacc: %.3f, Tloss: %.3f | Vacc: %.3f , Vloss: %.3f' %
             (epoch, metrics[0], metrics[1], metrics[2], metrics[3]))
    return

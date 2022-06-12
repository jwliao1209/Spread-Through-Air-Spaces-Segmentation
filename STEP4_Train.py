import os
import argparse

from Config import (get_cfg_defaults, get_ckpt_cfg)
from src.python import (set_determinism, trainer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='', help='config')
    opt = parser.parse_args()

    if opt.config == '':
        args = get_cfg_defaults()
    else:
        args = get_ckpt_cfg('configs', opt.config)

    set_determinism(args.BASIC.RANDOM_SEED)
    trainer(args)

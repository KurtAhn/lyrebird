#!/usr/bin/env python

if __name__ == '__main__':
    import sys
    import os
    from os import path
    sys.path.insert(0, path.dirname(path.realpath(__file__)))
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'

import numpy as np
from models.kurt import generate_unconditionally
from utils import plot_stroke
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='model', default='conditional')
    parser.add_argument('-e', '--epoch', dest='epoch', type=int, required=True)
    parser.add_argument('-l', '--length', dest='length', type=int, default=1000)
    parser.add_argument('-o', '--output', dest='output', default=None)
    args = parser.parse_args()

    stroke = generate_unconditionally(        
        model=args.model,
        epoch=args.epoch,
        length=args.length
    )
    #print('Stroke length: {}'.format(len(stroke)))

    plot_stroke(stroke, save_name=args.output)
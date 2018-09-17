#!/usr/bin/env python

if __name__ == '__main__':
    import sys
    import os
    from os import path
    sys.path.insert(0, path.dirname(path.realpath(__file__)))
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'

import numpy as np
from models.kurt import generate_conditionally
from utils import plot_stroke
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('text')
    parser.add_argument('-e', '--epoch', dest='epoch', type=int, required=True)
    parser.add_argument('-b', '--sample-bias', dest='sample_bias', type=float, default=0.0)
    parser.add_argument('-f', '--finish-line', dest='finish_line', type=float, default=0.5)
    parser.add_argument('-m', '--model', dest='model', default='conditional')
    args = parser.parse_args()

    stroke = generate_conditionally(
        text=args.text,
        model=args.model,
        epoch=args.epoch,
        sample_bias=args.sample_bias,
        finish_line=args.finish_line
    )

    print('Stroke length: {}'.format(len(stroke)))
    # plot_stroke(np.concatenate([
    #     stroke[:,1:],
    #     np.array([
    #         [0.0, 0.0, 0.0],
    #         [1.0, 0.0, 0.0],
    #     ])
    # ]))
    print(stroke)
    plot_stroke(stroke[:,1:])

#!/usr/bin/env python

import os
from os import path
import sys

import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    sys.path.insert(0, path.dirname(path.dirname(path.realpath(__file__))))
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from data import *
from models.kurt import Unconditional
from utils import Report

from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-m', '--mixture-size',
                        dest='M', type=int, default=20)
    parser.add_argument('-n', '--num-units',
                        dest='N', type=int, default=400)
    parser.add_argument('-l', '--num-layers',
                        dest='D', type=int, default=1)
    parser.add_argument('-r', '--learning-rate', type=float, default=1e-4)
    parser.add_argument('-c', '--clip-threshold', type=float, default=100.0)
    parser.add_argument('-s', '--dont-standardize', dest='standardize', action='store_false')
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=10)
    parser.add_argument('-o', '--output', dest='output', default='unconditional')

    args = parser.parse_args()

    MDLDEF = path.join(path.dirname(path.realpath(__file__)), 'mdldef')
    mdldir = path.join(MDLDEF, args.output)

    with tf.Session().as_default() as session:
        strokes = read_strokes()
        if args.standardize:
            strokes, offset_mean, offset_scale = standardize_strokes(strokes)
        else:
            offset_mean, offset_scale = 0.0, 1.0

        train_size = 5500
        train = unconditional_dataset(
            sort_strokes(strokes[:train_size]),
            batch_size=args.batch_size
        ).shuffle(min(train_size, 10000), seed=1337)
        train_iterator = train.make_initializable_iterator()
        train_example = train_iterator.get_next()

        valid_size = 500
        valid = unconditional_dataset(
            sort_strokes(strokes[-valid_size:]),
            batch_size=args.batch_size
        ).shuffle(min(valid_size, 10000), seed=1337)
        valid_iterator = valid.make_initializable_iterator()
        valid_example = valid_iterator.get_next()

        model = Unconditional(
            mixture_size=args.M,
            num_units=args.N,
            num_layers=args.D,
            offset_mean=offset_mean,
            offset_scale=offset_scale
        )
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=0)
        summarizer = tf.summary.FileWriter(
            path.join(MDLDEF, 'train', args.output),
            graph=session.graph
        )

        epochs = 0
        while True:
            epochs += 1

            session.run(train_iterator.initializer)
            train_report = Report(epochs, mode='t')

            iterations = 0
            total_loss = 0.0
            while True:
                try:
                    stroke, loss, _ = model.predict(
                        session.run(train_example), train=True,
                        learning_rate=args.learning_rate,
                        clip_threshold=args.clip_threshold)
                    total_loss += loss
                    iterations += 1
                    # print(output[0,:], file=sys.stderr)
                    # quit()
                    print('\r\x1b[0;{m};40m'
                           'Epoch: {e}'
                           ' Iteration: {i}'
                           ' Loss: {l:.3e}'
                           ' Avg: {a:.3e}'
                           # ' It./s: {s:.3f}'
                           '\x1b[0m'.format(
                            m=37,
                            e=epochs,
                            i=iterations,
                            l=loss,
                            a=total_loss / iterations
                            ), file=sys.stderr, end='')
                except tf.errors.OutOfRangeError:
                    break
            print('', file=sys.stderr)

            session.run(valid_iterator.initializer)
            iterations = 0
            total_loss = 0.0
            while True:
                try:
                    stroke, loss = model.predict(
                        session.run(valid_example), train=False,
                        learning_rate=args.learning_rate)
                    total_loss += loss
                    iterations += 1
                    # print(w, file=sys.stderr)
                    print('\r\x1b[0;{m};40m'
                           'Epoch: {e}'
                           ' Iteration: {i}'
                           ' Loss: {l:.3e}'
                           ' Avg: {a:.3e}'
                           # ' It./s: {s:.3f}'
                           '\x1b[0m'.format(
                            m=33,
                            e=epochs,
                            i=iterations,
                            l=loss,
                            a=total_loss / iterations
                            ), file=sys.stderr, end='')
                except tf.errors.OutOfRangeError:
                    break
            print('', file=sys.stderr)

            model.save(saver, '../mdldef/unconditional', epochs)

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
from models.kurt import Conditional
from utils import Report

from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-M', '--output-mixture-size',
                        dest='M', type=int, default=20)
    parser.add_argument('-K', '--window-mixture-size',
                        dest='K', type=int, default=10)
    parser.add_argument('-N', '--num-units',
                        dest='N', type=int, default=400)
    parser.add_argument('-D', '--num-layers',
                        dest='D', type=int, default=1)
    parser.add_argument('-r', '--learning-rate', dest='learning_rate', type=float, default=1e-4)
    parser.add_argument('-c', '--clip-threshold', dest='clip_threshold', type=float, default=100.0)
    parser.add_argument('-d', '--keep-prob', dest='keep_prob', type=float, default=1.0)
    parser.add_argument('-s', '--dont-standardize', dest='standardize', action='store_false')
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=10)
    parser.add_argument('-m', '--model', dest='model', required=True)
    parser.add_argument('-e', '--epoch', dest='epoch', type=int, default=0)

    args = parser.parse_args()

    MDLDEF = path.join(path.dirname(path.realpath(__file__)), 'mdldef')
    mdldir = path.join(MDLDEF, args.model)

    with tf.Session().as_default() as session:
        strokes = read_strokes()
        if args.standardize:
            strokes, offset_mean, offset_scale = standardize_strokes(strokes)
        else:
            offset_mean, offset_scale = 0.0, 1.0

        sentences, vocab = read_sentences()

        train_size = 5500
        train = conditional_dataset(
            sentences[:train_size],
            strokes[:train_size],
            batch_size=args.batch_size
        ).shuffle(train_size, seed=1337)
        train_iterator = train.make_initializable_iterator()
        train_example = train_iterator.get_next()

        valid_size = 500
        valid = conditional_dataset(
            sentences[-valid_size:],
            strokes[-valid_size:],
            batch_size=args.batch_size
        )
        valid_iterator = valid.make_initializable_iterator()
        valid_example = valid_iterator.get_next()


        if args.epoch:
            model = Conditional(mdldir=mdldir, epoch=args.epoch)
            session.run(tf.tables_initializer())
            saver = tf.train.Saver(max_to_keep=0)
        else:
            model = Conditional(
                output_mixture_size=args.M,
                window_mixture_size=args.K,
                num_units=args.N,
                num_layers=args.D,
                offset_mean=offset_mean,
                offset_scale=offset_scale,
                character_types=vocab
            )
            session.run(tf.global_variables_initializer())
            session.run(tf.tables_initializer())
            model.save(saver, mdldir, 0, meta=True)

        summarizer = tf.summary.FileWriter(
            path.join(MDLDEF, 'train', args.model),
            graph=session.graph
        )

        epochs = args.epoch
        while True:
            epochs += 1

            session.run(train_iterator.initializer)
            train_report = Report(epochs, mode='t')
            while True:
                try:
                    stroke, loss, sse, sl, ss, p, _ = model.predict(
                        *session.run(train_example),
                        train=True,
                        learning_rate=args.learning_rate,
                        clip_threshold=args.clip_threshold,
                        keep_prob=args.keep_prob
                    )
                    train_report.report(loss, sse)

                    if train_report.iterations % 10 == 0:
                        model.save(saver, mdldir, epochs)
                except tf.errors.OutOfRangeError:
                    break
            print('', file=sys.stderr)

            session.run(valid_iterator.initializer)
            valid_report = Report(epochs, mode='v')
            while True:
                try:
                    stroke, loss, sse, p = model.predict(
                        *session.run(valid_example), train=False
                    )
                    valid_report.report(loss, sse)
                except tf.errors.OutOfRangeError:
                    break
            print('', file=sys.stderr)

            # model.save(saver, '../mdldef/conditional', epochs)

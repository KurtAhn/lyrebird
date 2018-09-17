#!/usr/bin/env python

from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
from kurt import Unconditional, Sampler
import sys
from os import path


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--num_components', type=int, default=20)
    parser.add_argument('-u', '--num_units', type=int, default=400)
    parser.add_argument('-l', '--num_layers', type=int, default=3)
    parser.add_argument('-r', '--learning-rate', type=float, default=1e-4)
    parser.add_argument('-k', '--clip-threshold', type=float, default=100.0)
    args = parser.parse_args()

    with tf.Session().as_default() as session:
        with open('../data/strokes.npy', 'rb') as f:
            strokes = np.load(f, encoding='bytes')

        # Standardize stroke offsets
        flattened = np.concatenate([s.reshape(-1, 3) for s in strokes])
        offset_mean = np.mean(flattened[:,-2:], axis=0)
        offset_scale = np.std(flattened[:,-2:], axis=0)
        # print(offset_mean, offset_scale)
        # print(strokes[0])
        for n in range(len(strokes)):
            strokes[n][:,-2:] = (strokes[n][:,-2:] - offset_mean) / offset_scale
        # print(strokes[0])
        # offset_mean = 0.0
        # offset_scale = 1.0

        train_iterator = tf.data.Dataset.from_generator(
            lambda: strokes[:1],
            tf.as_dtype(strokes[0].dtype)
        ).make_initializable_iterator()
        train_example = train_iterator.get_next()

        valid_iterator = tf.data.Dataset.from_generator(
            lambda: strokes[-10:],
            tf.as_dtype(strokes[0].dtype)
        ).make_initializable_iterator()
        valid_example = valid_iterator.get_next()

        # sampler = Sampler(num_components=args.num_components)
        model = Unconditional(
            num_components=args.num_components,
            num_units=args.num_units,
            num_layers=args.num_layers,
            offset_mean=offset_mean,
            offset_scale=offset_scale,
            sampler=sampler
        )
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=0)
        summarizer = tf.summary.FileWriter('../mdldef/train/unconditional', graph=session.graph)

        epochs = 0
        while True:
            session.run(train_iterator.initializer)
            epochs += 1

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

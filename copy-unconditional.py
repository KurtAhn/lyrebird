#!/usr/bin/env python
"""
For copying parameters of an existing model to a new model.
"""
import tensorflow as tf

if __name__ == '__main__':
    import sys, os
    from os import path
    sys.path.insert(0, path.dirname(path.realpath(__file__)))
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from models.kurt import Unconditional
from data import *


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--model', '-m', required=True)
    parser.add_argument('--epoch', '-e', type=int, required=True)
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('-M', '--output-mixture-size',
                        dest='M', type=int, default=20)
    parser.add_argument('-N', '--num-units',
                        dest='N', type=int, default=400)
    parser.add_argument('-D', '--num-layers',
                        dest='D', type=int, default=1)
    parser.add_argument('-s', '--dont-standardize', dest='standardize', action='store_false')
    args = parser.parse_args()

    MDLDEF = path.join(path.dirname(path.realpath(__file__)), 'mdldef')
    with tf.Graph().as_default() as g1:
        with tf.Session().as_default() as s1:
            old = Unconditional(mdldir=path.join(MDLDEF, args.model), epoch=args.epoch)
            old_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    strokes = read_strokes()
    if args.standardize:
        strokes, offset_mean, offset_scale = standardize_strokes(strokes)
    else:
        offset_mean, offset_scale = 0.0, 1.0

    with tf.Graph().as_default() as g2:
        with tf.Session().as_default() as s2:
            new = Conditional(
                mixture_size=args.M,
                num_units=args.N,
                num_layers=args.D,
                offset_mean=offset_mean,
                offset_scale=offset_scale
            )
            new_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    copy = []
    for x in old_variables:
        xval = x.eval(session=s1)
        with g2.as_default():
            copy.append(tf.assign(
                next(
                    y
                    for y in new_variables
                    if y.name.split('/')[1:] == x.name.split('/')[1:]
                ), xval))
    with g2.as_default():
        s2.run(tf.group(*copy))

    for x in old_variables:
        assert np.array_equal(
            x.eval(session=s1),
            next(
                y
                for y in new_variables
                if y.name.split('/')[1:] == x.name.split('/')[1:]
            ).eval(session=s2)
        )

    with g2.as_default():
        with s2.as_default():
            saver = tf.train.Saver(max_to_keep=0)
            new.save(saver, path.join(MDLDEF, args.output), args.epoch, meta=True)

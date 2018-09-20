#!/usr/bin/env python

import os
from os import path
import sys
if __name__ == '__main__':
    sys.path.insert(0, path.dirname(path.dirname(path.realpath(__file__))))
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from collections import namedtuple

from data import STROKE_DIM


# Used for recording output at each step
Record = namedtuple(
    'Record',
    ['stroke', 'loss', 'sse', 'param']
)


class ModelBase:
    """
    Base class for Tensorflow networks.
    (Adapted from my own project at https://github.com/KurtAhn/SLCV)
    """
    def __init__(self, name, **kwargs):
        """
        Create a new model (no kwargs) or restore an existing model.

        name: Name of the subclass used for scoping
        **kwargs
            mdldir: Directory of an existing model description
            epoch: Subindex of the precise model version representing the
                number of epochs it was trained for
        """
        self._name = name
        if 'mdldir' not in kwargs:
            self._create(**kwargs)
        else:
            try:
                mdldir = kwargs.pop('mdldir')
                epoch = kwargs.pop('epoch', None)
                self._restore(mdldir, epoch)
            except KeyError:
                raise ValueError('mdldir argument not provided')

    def _restore(self, mdldir, epoch):
        """
        Restore an existing model.

        mdldir: Directory with the description of the model to load
        epoch: Version of model to load
        """
        g = tf.get_default_graph()
        meta = tf.train.import_meta_graph(path.join(mdldir, '_.meta'))
        if epoch is not None:
            with open(path.join(mdldir, 'checkpoint'), 'w') as f:
                f.write('model_checkpoint_path: "' +
                        path.join(path.realpath(mdldir), '_-{}'.format(epoch)) + '"')
        meta.restore(tf.get_default_session(),
                     tf.train.latest_checkpoint(mdldir))

    def save(self, saver, mdldir, step, meta=False):
        """
        Save a model to disk.

        saver: tf.Saver object
        mdldir: Directory to save the model in
        step: Version number used to distinguish
            model descriptions under the same directory
        meta: If True, save meta-graph
        """
        if meta:
            tf.train.export_meta_graph(
                filename=path.join(mdldir, '_.meta')
            )
        saver.save(tf.get_default_session(),
                   path.join(mdldir, '_'),
                   global_step=step,
                   write_meta_graph=False)

    @property
    def name(self):
        """
        Model name used for scoping.
        """
        return self._name

    def __getitem__(self, k):
        """
        Convenient method for accessing tensors and operations scoped by
        self.name.
        """
        g = tf.get_default_graph()
        try:
            return g.get_tensor_by_name('{}/{}:0'.format(self.name, k))
        except KeyError:
            try:
                return g.get_operation_by_name('{}/{}'.format(self.name, k))
            except KeyError:
                raise KeyError('Nonexistent name: {}/{}'.format(self.name, k))

    def __getattr__(self, a):
        """
        Convenient method for accessing tensors and operations scoped by
        self.name.
        """
        return self[a]


class Unconditional(ModelBase):
    """
    Handwriting prediction network
    """
    def __init__(self, **kwargs):
        ModelBase.__init__(self, 'Unconditional', **kwargs)

    def _create(self, **kwargs):
        """
        **kwargs
            mixture_size: Number of Gaussian mixture components
            num_units: Number of RNN cell units
            num_layers: Number of cells
            offset_mean: If using standardized offset values, provide the
                dataset mean value
            offset_scale: Dataset standard deviation
        """
        M = self.mixture_size = kwargs.get('mixture_size', 20)
        self.num_units = kwargs.get('num_units', 400)
        self.num_layers = kwargs.get('num_layers', 1)
        with tf.name_scope(self.name) as scope:
            # Number of strokes to output
            tf.placeholder('int32', [None], name='length')
            # Target sequence
            tf.placeholder('float', [None, None, STROKE_DIM], name='target')

            # Operation mode: training or validating
            tf.placeholder('bool', [], name='is_training')
            tf.placeholder('bool', [], name='is_validating')

            # Optimization parameters
            tf.placeholder('float', [], name='learning_rate')
            tf.placeholder('float', [], name='clip_threshold')
            tf.placeholder_with_default(
                tf.constant(1.0, 'float'),
                shape=[],
                name='keep_prob'
            )

            # Standardization
            tf.constant(kwargs.get('offset_mean', 0.0), name='offset_mean')
            tf.constant(kwargs.get('offset_scale', 1.0), name='offset_scale')

            tf.reduce_max(self.length, name='max_length')
            tf.identity(tf.shape(self.length)[0], name='batch_size')

            with tf.variable_scope(self.name) as scope:
                init = tf.contrib.layers.xavier_initializer()
                func = tf.nn.tanh

                rnns = [tf.contrib.rnn.LSTMCell(
                    num_units=self.num_units,
                    initializer=init,
                    activation=func,
                    state_is_tuple=True
                ) for n in range(self.num_layers)]

                def condition(t, x, *_):
                    return t < self.max_length

                def forward(t, x, qc, qh, record):
                    def dropout(v):
                        return tf.nn.dropout(
                            v,
                            tf.cond(
                                self.is_training,
                                lambda: self.keep_prob,
                                lambda: tf.constant(1.0, 'float')
                            )
                        )

                    h, (qc[0], qh[0]) = rnns[0](x, [qc[0], qh[0]])
                    h = dropout(h)

                    for d in range(1, self.num_layers):
                        h, (qc[d], qh[d]) = rnns[d]\
                                            (tf.concat([x, h], axis=-1), [qh[d], qh[d]])
                        h = dropout(h)

                    y_ = tf.reshape(
                        tf.layers.dense(
                            h,
                            self.num_params,
                            kernel_initializer=init
                        ),
                        [-1, self.num_params]
                    )

                    # Mixture weight
                    p = tf.nn.softmax(y_[:,0:M])
                    # Mixture mean
                    m1 = y_[:,M:2*M]
                    m2 = y_[:,2*M:3*M]
                    # Mixture standard deviation
                    s1 = tf.exp(y_[:,3*M:4*M])
                    s2 = tf.exp(y_[:,4*M:5*M])
                    # Mixture correlation
                    r = tf.nn.tanh(y_[:,5*M:6*M])
                    # Probability of lifting the pen tip
                    e = tf.sigmoid(-y_[:,6*M:6*M+1])

                    lift_pdf = tfd.Bernoulli(probs=tf.reshape(e, [self.batch_size]), dtype='float')
                    offset_pdf = tfd.MixtureSameFamily(
                        mixture_distribution=tfd.Categorical(probs=p),
                        components_distribution=tfd.MultivariateNormalFullCovariance(
                            loc=tf.stack([m1, m2], axis=-1),
                            covariance_matrix=tf.reshape(tf.stack(
                                [tf.square(s1),
                                 s1 * s2 * r,
                                 s1 * s2 * r,
                                 tf.square(s2)
                                ],
                                axis=-1
                            ), [self.batch_size, M, 2, 2])
                        )
                    )

                    # Next input -- get from target sequence or
                    # generate one by sampling from distribution
                    x = tf.cond(
                        tf.logical_or(self.is_training, self.is_validating),
                        lambda: target_ta.read(t),
                        lambda: tf.concat(
                            [tf.expand_dims(lift_pdf.sample(), 1),
                             offset_pdf.sample()],
                            axis=-1
                        )
                    )
                    x_ta = record.stroke.write(t, x)

                    # This is for batch training; terminate early for
                    # shorter sequences
                    compute_loss = t < self.length

                    loss = tf.where(
                        compute_loss,
                        -(
                            tf.maximum(offset_pdf.log_prob(x[:,-2:]), -1000) +
                            lift_pdf.log_prob(x[:,0])
                        )  / tf.cast(self.length, 'float'),
                        tf.zeros([self.batch_size])
                    )
                    loss_ta = record.loss.write(t, loss)

                    sse = tf.where(
                        compute_loss,
                        tf.reduce_sum(
                            tf.square(x - tf.concat([e, offset_pdf.mean()], axis=-1)),
                            axis=-1
                        ),
                        tf.zeros([self.batch_size])
                    )
                    sse_ta = record.sse.write(t, sse)

                    param_ta = record.param.write(t, tf.concat([
                        p, m1, m2, s1, s2, r, e
                    ], axis=-1))

                    return t + 1, \
                           tf.reshape(x, [self.batch_size, STROKE_DIM]), \
                           qc, qh, \
                           Record(x_ta, loss_ta, sse_ta, param_ta)

                time = tf.constant(0, dtype='int32')

                target_ta = tf.TensorArray(
                    'float',
                    size=self.max_length
                ).unstack(tf.transpose(self.target, perm=[1,0,2]))

                # Initial c- and h-state for LSTMs
                qch = [rnns[d].zero_state(self.batch_size, dtype='float')
                       for d in range(self.num_layers)]
                record = tf.while_loop(
                    condition,
                    forward,
                    [time,
                     tf.zeros([self.batch_size, STROKE_DIM], dtype='float'),
                     [q[0] for q in qch],
                     [q[1] for q in qch],
                     Record(*[
                        tf.TensorArray('float', size=0, dynamic_size=True)
                        for n in range(4)]
                     )]
                )[-1]

            tf.transpose(record.param.stack(), perm=[1,0,2], name='param')

            stroke = tf.transpose(record.stroke.stack(), perm=[1,0,2])

            tf.concat([
                stroke[:,:,:-2],
                stroke[:,:,-2:] * self.offset_scale + self.offset_mean
            ], axis=-1, name='stroke')

            tf.transpose(record.loss.stack(), name='step_loss')
            tf.reduce_sum(self.step_loss / tf.cast(self.batch_size, 'float'),
                          name='loss')

            tf.transpose(record.sse.stack(), name='step_sse')
            tf.reduce_mean(self.step_sse, name='sse')

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.optimizer.apply_gradients(
                [(tf.clip_by_norm(g, self.clip_threshold), v)
                 for g, v in self.optimizer.compute_gradients(self.loss)],
                name='optimization'
            )

    @property
    def num_params(self):
        return self.mixture_size * 6 + 1

    def predict(self, x, l, train=False, **kwargs):
        """
        Train or validate model on a batch of sequences

        x: Target sequence batch
        l: Target sequence length batch
        train: True to train, False to validate (no weight update)

        **kwargs
            learning_rate: Learning rate (ignored if train == False)
            clip_threshold: Clip gradient to this (ignored if train == False)
            keep_prob: Dropout retain probability (ignored if train == False)

        returns stroke output, log loss, and sum squared error if train == False;
            and all three + output of the optimization operation if train == True
        """
        session = tf.get_default_session()
        learning_rate = kwargs.get('learning_rate', 1e-4)
        clip_threshold = kwargs.get('clip_threshold', 100)
        keep_prob = kwargs.get('keep_prob', 1.0)

        if train:
            return session.run(
                [self.stroke, self.loss, self.sse, self.optimization],
                feed_dict={
                    self.target: x,
                    self.length: l,
                    self.is_training: True,
                    self.is_validating: False,
                    self.learning_rate: learning_rate,
                    self.clip_threshold: clip_threshold,
                    self.keep_prob: keep_prob
                }
            )
        else:
            return session.run(
                [self.stroke, self.loss, self.sse],
                feed_dict={
                    self.target: x,
                    self.length: l,
                    self.is_training: False,
                    self.is_validating: True,
                }
            )

    def synth(self, l, **kwargs):
        """
        Produce handwriting sequence of a given length

        l: Length of the sequence to produce (scalar)

        **kwargs
            (No arguments)
        """
        session = tf.get_default_session()

        return session.run(
            [self.stroke],
            feed_dict={
                self.target: np.zeros([1,1,STROKE_DIM], 'float32'),
                self.length: [l],
                self.is_training: False,
                self.is_validating: False
            }
        )


class Conditional(ModelBase):
    """
    Handwriting synthesis network
    """
    def __init__(self, **kwargs):
        ModelBase.__init__(self, 'Conditional', **kwargs)

    def _create(self, **kwargs):
        """
        **kwargs
            output_mixture_size: Number of mixture components for the
                output layer
            window_mixture_size: Number of mixture components for the
                window layer
            num_units: Number of RNN cell units
            num_layers: Number of cells
            character_types: List of character types (required)
            offset_mean: If using standardized offset values, provide the
                dataset mean value
            offset_scale: Dataset standard deviation
        """
        M = self.output_mixture_size = kwargs.get('output_mixture_size', 20)
        K = self.window_mixture_size = kwargs.get('window_mixture_size', 10)
        self.num_units = kwargs.get('num_units', 400)
        self.num_layers = kwargs.get('num_layers', 1)
        self.num_chars = len(kwargs['character_types']) + 1
        with tf.name_scope(self.name) as scope:
            tf.placeholder('int32', [None], name='text_length')
            tf.placeholder('string', [None, None], name='text')

            tf.placeholder('int32', [None], name='target_length')
            tf.placeholder('float', [None, None, STROKE_DIM], name='target')

            tf.placeholder('bool', [], name='is_training')
            tf.placeholder('bool', [], name='is_validating')

            tf.placeholder('float', [], name='learning_rate')
            tf.placeholder('float', [], name='clip_threshold')
            tf.placeholder_with_default(
                tf.constant(1.0, 'float'),
                shape=[],
                name='keep_prob'
            )

            # For biased sampling
            tf.placeholder_with_default(
                tf.constant(0.0, 'float'),
                shape=[],
                name='sample_bias'
            )

            tf.constant(kwargs.get('offset_mean', 0.0),
                        dtype='float', name='offset_mean')
            tf.constant(kwargs.get('offset_scale', 1.0),
                        dtype='float', name='offset_scale')

            tf.constant(kwargs['character_types'], dtype='string', name='character_types')
            character_table = tf.contrib.lookup.index_table_from_tensor(
                mapping=self.character_types,
                num_oov_buckets=1,
                default_value=-1
            )

            tf.reduce_max(self.target_length, name='max_target_length')
            tf.reduce_max(self.text_length, name='max_text_length')
            tf.identity(tf.shape(self.text)[0], name='batch_size')

            init = tf.contrib.layers.xavier_initializer()
            func = tf.nn.tanh
            with tf.variable_scope(self.name):
                rnns = [tf.contrib.rnn.LSTMCell(
                    num_units=self.num_units,
                    initializer=init,
                    activation=func,
                    state_is_tuple=True
                ) for n in range(self.num_layers)]

                def condition(t, x, *_):
                    return t < self.max_target_length

                def forward(t, x, qc, qh, k, w, record):
                    def dropout(v):
                        return tf.nn.dropout(
                            v,
                            tf.cond(
                                self.is_training,
                                lambda: self.keep_prob,
                                lambda: tf.constant(1.0, 'float')
                            )
                        )

                    h, (qc[0], qh[0]) = rnns[0]\
                                        (tf.concat([x, w], axis=-1), [qc[0], qh[0]])
                    h = dropout(h)

                    # This is to account for the fact that there are about
                    # 21.7 strokes per character
                    # Without this bit, initially kappa would increment by 1
                    # on average, which is too fast
                    bias = np.zeros([3*K], dtype='float32')
                    bias[2*K:] = np.log(1.0/21.7) * np.ones([K], dtype='float32')

                    z_ = tf.reshape(tf.layers.dense(
                        h,
                        3*K,
                        kernel_initializer=init,
                        bias_initializer=tf.constant_initializer(bias)
                    ), [-1, 3*K])

                    a = tf.exp(z_[:,:K])
                    b = tf.exp(z_[:,K:2*K])
                    k += tf.exp(z_[:,2*K:])

                    u_range = tf.range(tf.cast(self.max_text_length, 'float'))
                    u = tf.tile(
                        tf.reshape(
                            u_range,
                            [1, 1, -1]
                        ), [self.batch_size, K, 1])
                    c = tf.one_hot(
                        character_table.lookup(self.text),
                        self.num_chars
                    )

                    # The following bit is for accurately computing the window
                    # function by early terminating shorter text sequences.
                    # Commented out for performance reason, since not using batch training.

                    # mask = tf.tile(
                    #     tf.reshape(
                    #         u_range,
                    #         [1, -1, 1]
                    #     ),
                    #     [self.batch_size, 1, self.num_chars]
                    # )
                    # c = tf.where(
                    #     mask <= tf.expand_dims(
                    #         tf.tile(
                    #             tf.expand_dims(
                    #                 tf.cast(self.text_length, 'float'),
                    #                 -1
                    #             ),
                    #             [1, self.max_text_length]
                    #         ),
                    #         -1
                    #     ),
                    #     c,
                    #     tf.zeros_like(mask)
                    # )
                    phi = tf.reduce_sum(
                        tf.expand_dims(a, -1) * \
                        tf.exp(
                            tf.expand_dims(-b, -1) *
                            tf.square(tf.expand_dims(k, -1) - u)
                        ),
                        axis=1
                    )
                    w = tf.reduce_sum(
                        tf.expand_dims(phi, -1) * c,
                        axis=1
                    )

                    for d in range(1, self.num_layers):
                        h, (qc[d], qh[d]) = rnns[d]\
                                            (tf.concat([x, h, w], axis=-1), [qc[d], qh[d]])
                        h = dropout(h)

                    y_ = tf.reshape(tf.layers.dense(
                        h,
                        self.num_params,
                        kernel_initializer=init
                    ), [-1, self.num_params])

                    p = tf.nn.softmax(y_[:,0:M] * (1.0 + self.sample_bias))
                    m1 = y_[:,M:2*M]
                    m2 = y_[:,2*M:3*M]
                    s1 = tf.exp(y_[:,3*M:4*M] - self.sample_bias)
                    s2 = tf.exp(y_[:,4*M:5*M] - self.sample_bias)
                    r = tf.nn.tanh(y_[:,5*M:6*M])
                    e = tf.sigmoid(-y_[:,6*M:6*M+1])

                    lift_pdf = tfd.Bernoulli(probs=tf.reshape(e, [self.batch_size]), dtype='float')
                    offset_pdf = tfd.MixtureSameFamily(
                        mixture_distribution=tfd.Categorical(probs=p),
                        components_distribution=tfd.MultivariateNormalFullCovariance(
                            loc=tf.stack([m1, m2], axis=-1),
                            covariance_matrix=tf.reshape(tf.stack(
                                [tf.square(s1),
                                 s1 * s2 * r,
                                 s1 * s2 * r,
                                 tf.square(s2)
                                ],
                                axis=-1
                            ), [self.batch_size, M, 2, 2])
                        )
                    )

                    x = tf.cond(
                        tf.logical_or(self.is_training, self.is_validating),
                        lambda: target_ta.read(t),
                        lambda: tf.concat(
                            [
                             # Maybe the pen tip lift value should be definite?
                             # Otherwise, high sampling bias still leads to heavily
                             # different sequences to be produced on each run
                             # tf.where(
                             #    e < 0.039828256 + 0.19555554, # Mean + std of e
                             #    tf.zeros_like(e),
                             #    tf.ones_like(e)
                             # ),
                             tf.expand_dims(lift_pdf.sample(), 1),
                             offset_pdf.sample()],
                            axis=-1
                        )
                    )
                    x_ta = record.stroke.write(t, x)

                    compute_loss = t < self.target_length
                    loss = tf.where(
                        compute_loss,
                        -(
                            tf.maximum(offset_pdf.log_prob(x[:,-2:]), -1000) +
                            lift_pdf.log_prob(x[:,0])
                        )  / tf.cast(self.target_length, 'float'),
                        tf.zeros([self.batch_size])
                    )
                    loss_ta = record.loss.write(t, loss)

                    sse = tf.where(
                        compute_loss,
                        tf.reduce_sum(
                            tf.square(x - tf.concat([e, offset_pdf.mean()], axis=-1)),
                            axis=-1
                        ),
                        tf.zeros([self.batch_size])
                    )
                    sse_ta = record.sse.write(t, sse)

                    param_ta = record.param.write(t, tf.concat([
                        p, m1, m2, s1, s2, r, e, w
                    ], axis=-1))

                    return t + 1, \
                           tf.reshape(x, [self.batch_size, STROKE_DIM]), \
                           qc, qh, k, w, \
                           Record(x_ta, loss_ta, sse_ta, param_ta)

                time = tf.constant(0, dtype='int32')

                target_ta = tf.TensorArray(
                    'float',
                    size=self.max_target_length
                ).unstack(tf.transpose(self.target, perm=[1,0,2]))

                qch = [rnns[d].zero_state(self.batch_size, dtype='float')
                       for d in range(self.num_layers)]
                record = tf.while_loop(
                    condition,
                    forward,
                    [time,
                     tf.zeros([self.batch_size, STROKE_DIM], dtype='float'),
                     [q[0] for q in qch],
                     [q[1] for q in qch],
                     tf.zeros([self.batch_size, K], dtype='float'),
                     tf.zeros([self.batch_size, self.num_chars], dtype='float'),
                     Record(*[
                        tf.TensorArray('float', size=0, dynamic_size=True)
                        for n in range(4)]
                     )]
                )[-1]

            tf.transpose(record.param.stack(), perm=[1, 0, 2], name='param')

            stroke = tf.transpose(record.stroke.stack(), perm=[1, 0, 2])

            tf.concat([
                stroke[:,:,:-2],
                stroke[:,:,-2:] * self.offset_scale + self.offset_mean
            ], axis=-1, name='stroke')

            tf.transpose(record.loss.stack(), name='step_loss')
            tf.reduce_sum(self.step_loss / tf.cast(self.batch_size, 'float'),
                          name='loss')

            tf.transpose(record.sse.stack(), name='step_sse')
            tf.reduce_mean(self.step_sse, name='sse')

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.optimizer.apply_gradients(
                [(tf.clip_by_norm(g, self.clip_threshold), v)
                 for g, v in self.optimizer.compute_gradients(self.loss)],
                name='optimization'
            )

    @property
    def num_params(self):
        return self.output_mixture_size * 6 + 1

    def predict(self, t, tl, x, xl, train=False, **kwargs):
        session = tf.get_default_session()
        learning_rate = kwargs.get('learning_rate', 1e-4)
        clip_threshold = kwargs.get('clip_threshold', 100)
        keep_prob = kwargs.get('keep_prob', 1.0)
        batch_size = len(t)

        if train:
            return session.run(
                [self.stroke, self.loss, self.sse, self.step_loss, self.step_sse, self.param, self.optimization],
                feed_dict={
                    self.text: t,
                    self.text_length: tl,
                    self.target: x,
                    self.target_length: xl,
                    self.is_training: True,
                    self.is_validating: False,
                    self.learning_rate: learning_rate,
                    self.clip_threshold: clip_threshold,
                    self.keep_prob: keep_prob
                }
            )
        else:
            return session.run(
                [self.stroke, self.loss, self.sse, self.param],
                feed_dict={
                    self.text: t,
                    self.text_length: tl,
                    self.target: x,
                    self.target_length: xl,
                    self.is_training: False,
                    self.is_validating: True,
                }
            )

    def synth(self, t, **kwargs):
        session = tf.get_default_session()
        sample_bias = kwargs.get('sample_bias', 0.0)
        max_length = kwargs.get('max_length', 1000)

        return session.run(
            [self.stroke, self.param],
            feed_dict={
                self.text: [t],
                self.text_length: [len(t)],
                self.target: np.zeros([1,1,STROKE_DIM], 'float32'),
                self.target_length: [max_length],
                self.is_training: False,
                self.is_validating: False,
                self.sample_bias: sample_bias,
            }
        )


MDLDEF = path.join(path.dirname(path.dirname(path.realpath(__file__))), 'mdldef')
def generate_unconditionally(random_seed=1, model='u', epoch=4, length=750):
    # Input:
    #   random_seed - integer <-- I didn't implement this!! Not at all straightforward
    #                             to set the seed after model has been created in TF...

    # Output:
    #   stroke - numpy 2D-array (T x 3)

    tf.reset_default_graph()
    with tf.Graph().as_default() as graph:
        with tf.Session().as_default() as session:
            unconditional = Unconditional(mdldir=path.join(MDLDEF, model), epoch=epoch)
            stroke, = unconditional.synth(length)

    return stroke[0]

    # strokes = np.load('../data/strokes.npy', encoding='bytes')
    # stroke = strokes[0]
    #return stroke


def generate_conditionally(text='welcome to lyrebird', random_seed=1,
                           model='c', epoch=4,
                           sample_bias=1.0,
                           stroke_length=None):
    stroke_length = stroke_length or len(text.replace(' ', '')) * 30
    tf.reset_default_graph()
    with tf.Session().as_default() as session:
        conditional = Conditional(mdldir=path.join(MDLDEF, model), epoch=epoch)
        session.run(tf.tables_initializer())
        stroke, param = conditional.synth(np.array([c for c in text]),
                                          sample_bias=sample_bias,
                                          max_length=stroke_length)
    # from matplotlib import pyplot
    #
    # heat = param[0,:,97:].T
    # mean = np.mean(heat)
    # std = np.std(heat)
    # pyplot.imshow(heat, aspect='auto')
    # pyplot.colorbar()
    #
    # pyplot.show()
    return stroke[0]


def recognize_stroke(stroke):
    return 'not implemented'

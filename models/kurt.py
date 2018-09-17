#!/usr/bin/env python

import os
from os import path
import sys
if __name__ == '__main__':
    sys.path.insert(0, path.dirname(path.dirname(path.realpath(__file__))))
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# from utils import plot_stroke

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from collections import namedtuple


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
                suffix = kwargs.pop('suffix', '')
                self._restore(mdldir, suffix)
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
        if epoch:
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
        epoch: Number of epochs the model was trained for (used to distinguish
            model descriptions under the same directory)
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


class Sampler(ModelBase):
    def __init__(self, **kwargs):
        ModelBase.__init__(self, 'Sampler', **kwargs)

    def _create(self, **kwargs):
        self._randomizer = np.random.RandomState(None)
        M = self.num_components = kwargs.get('num_components', None)
        with tf.name_scope(self.name) as scope:
            def sample(e, w, m1, m2, s1, s2, r):
                x0 = self.randomizer.binomial(1, e, 1).astype('float32')[0]
                c = self.randomizer.choice(len(w), 1, p=w)
                x12 = self.randomizer.multivariate_normal(
                    np.stack([m1[c], m2[c]]).reshape([2]),
                    np.stack([
                        s1[c] ** 2, r[c] * s1[c] * s2[c],
                        r[c] * s1[c] * s2[c], s2[c] ** 2
                    ]).reshape([2, 2]),
                    1
                ).astype('float32')
                return np.array([x0, x12[0][0], x12[0][1]])

            if M is None:
                args = [
                    # tf.get_variable(name=name)
                    self[name]
                    for name in 'e w m1 m2 s1 s2 r'.split()
                ]
            else:
                args = [
                    tf.get_variable(initializer=np.zeros([1], dtype='float32'),
                                    trainable=False,
                                    name='e'),
                    tf.get_variable(initializer=np.ones([M], dtype='float32') / M,
                                    trainable=False,
                                    name='w')
                ] + [
                    tf.get_variable(initializer=np.zeros([M], dtype='float32'),
                                    trainable=False,
                                    name=name)
                    for name in 'm1 m2 s1 s2 r'.split()
                ]

            tf.py_func(sample, args, 'float', name='sample')
            print(self.sample)

    @property
    def randomizer(self):
        return self._randomizer

    @randomizer.setter
    def randomizer(self, randomizer):
        self._randomizer = randomizer


class Unconditional(ModelBase):
    def __init__(self, **kwargs):
        ModelBase.__init__(self, 'Unconditional', **kwargs)

    def _create(self, **kwargs):
        M = self.mixture_size = kwargs.get('mixture_size', 20)
        self.num_units = kwargs.get('num_units', 400)
        self.num_layers = kwargs.get('num_layers', 1)
        with tf.name_scope(self.name) as scope:
            # Number of strokes to output
            tf.placeholder('int32', [None], name='length')
            # Target sequence
            tf.placeholder('float', [None, None, 4], name='target')

            # Operation mode: training or validating
            tf.placeholder('bool', [], name='is_training')
            tf.placeholder('bool', [], name='is_validating')

            # Optimization parameters
            tf.placeholder('float', [], name='learning_rate')
            tf.placeholder('float', [], name='clip_threshold')
            tf.placeholder_with_default(
                tf.constant(0.5, 'float'),
                shape=[],
                name='finish_line'
            )

            # Standardization
            tf.constant(kwargs.get('offset_mean', 0.0), name='offset_mean')
            tf.constant(kwargs.get('offset_scale', 1.0), name='offset_scale')

            tf.reduce_max(self.length, name='max_length')
            tf.identity(tf.shape(self.length)[0], name='batch_size')

            with tf.variable_scope(
                'rnn',
                initializer=tf.contrib.layers.xavier_initializer(),
                reuse=tf.AUTO_REUSE
            ):
                rnn = tf.contrib.rnn.GRUCell(
                    num_units=self.num_units,
                    activation=tf.nn.tanh
                )

                def condition(t, x, *_):
                    return tf.cond(
                        tf.logical_or(self.is_training, self.is_validating),
                        lambda: t < self.max_length,
                        lambda: tf.logical_and(
                            tf.cond(
                                tf.equal(self.max_length, 0),
                                lambda: tf.constant(True),
                                lambda: t < self.max_length
                            ),
                            tf.reduce_any(x[:,0] < self.finish_line)
                        )
                    )

                def forward(t, x, q, o_ta):
                    h, q[0] = rnn(x, q[0])
                    for d in range(1, self.num_layers):
                        h, q[d] = rnn([x, h], q[d])
                    y_ = tf.reshape(
                        tf.layers.dense(
                            d,
                            6*M+2,
                            kernel_initializer=init,
                            reuse=tf.AUTO_REUSE
                        )
                    )

                    # Whether to finish writing
                    f = tf.sigmoid(-y_[:,0:1])
                    e = tf.sigmoid(-y_[:,1:2])
                    p = tf.nn.softmax(y_[:,2:M+2])
                    m1 = y_[:,M+2:2*M+2]
                    m2 = y_[:,2*M+2:3*M+2]
                    s1 = tf.exp(y_[:,3*M+2:4*M+2])
                    s2 = tf.exp(y_[:,4*M+2:5*M+2])
                    r = tf.nn.tanh(y_[:,5*M+2:6*M+2])

                    finish_pdf = tfd.Bernoulli(probs=tf.reshape(f, [self.batch_size]), dtype='float')
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

                    compute_loss = x[:,0] < self.finish_line
                    x = tf.cond(
                        tf.logical_or(self.is_training, self.is_validating),
                        lambda: target_ta.read(t),
                        lambda: tf.concat(
                            [tf.where(f < self.finish_line,
                                      tf.zeros([self.batch_size, 1], dtype='float'),
                                      tf.ones([self.batch_size, 1], dtype='float')),
                             tf.expand_dims(lift_pdf.sample(), 1),
                             offset_pdf.sample()],
                            axis=-1
                        )
                    )

                    loss = tf.where(
                        compute_loss,
                        -(
                            tf.maximum(offset_pdf.log_prob(x[:,-2:]), -1e20) +
                            lift_pdf.log_prob(x[:,1]) +
                            finish_pdf.log_prob(x[:,0])
                        )  / tf.cast(self.target_length, 'float'),
                        tf.zeros([self.batch_size])
                    )
                    loss = tf.reshape(loss, [self.batch_size, 1])

                    sse = tf.where(
                        compute_loss,
                        tf.reduce_sum(
                            tf.square(x - tf.concat([f, e, offset_pdf.mean()],
                                                    axis=-1)),
                            axis=-1
                        ),
                        tf.zeros([self.batch_size])
                    )
                    sse = tf.reshape(sse, [self.batch_size, 1])

                    o_ta = o_ta.write(t, tf.concat([
                        x, loss, sse, f, e, p, m1, m2, s1, s2, r
                    ], axis=-1))

                    return t + 1, tf.reshape(x, [self.batch_size, 4]), q, o_ta

                time = tf.constant(0, dtype='int32')

                target_ta = tf.TensorArray(
                    'float',
                    size=self.max_length
                ).unstack(tf.transpose(self.target, perm=[1,0,2]))

                output_ta = tf.while_loop(
                    condition,
                    forward,
                    [time,
                     tf.zeros([self.batch_size, 4], dtype='float'),
                     [rnn.zero_state(self.batch_size, dtype='float')
                      for d in range(self.num_layers)],
                     tf.TensorArray('float', size=0, dynamic_size=True)
                     ]
                )[-1]

            tf.transpose(
                tf.reshape(output_ta.stack(), [-1, self.batch_size, 4+1+1+6*M+2]),
                perm=[1,0,2],
                name='output'
            )
            tf.add(self.output[:,:,2:4] * self.offset_scale, self.offset_mean,
                   name='stroke_offset')
            tf.concat([self.output[:,:,0:2],
                       self.stroke_offset], axis=-1,
                      name='stroke')
            tf.reduce_sum(self.output[:,:,4], name='loss')
            tf.reduce_mean(self.output[:,:,5], name='sse')

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.optimizer.apply_gradients(
                [(tf.clip_by_norm(g, self.clip_threshold), v)
                 for g, v in self.optimizer.compute_gradients(self.loss)],
                name='optimization'
            )

    def predict(self, x, train=False, **kwargs):
        session = tf.get_default_session()
        learning_rate = kwargs.get('learning_rate', 1e-4)
        clip_threshold = kwargs.get('clip_threshold', 100)

        if train:
            return session.run(
                [self.stroke, self.loss, self.sse, self.optimization],
                feed_dict={
                    self.target: x,
                    self.length: len(x),
                    self.is_training: True,
                    self.is_validating: False,
                    self.learning_rate: learning_rate,
                    self.clip_threshold: clip_threshold,
                }
            )
        else:
            if isinstance(x, int):
                return session.run(
                    [self.stroke],
                    feed_dict={
                        self.target: np.zeros([x, 3], 'float'),
                        self.length: x,
                        self.is_training: False,
                        self.is_validating: False,
                    }
                )
            else:
                return session.run(
                    [self.stroke, self.loss],
                    feed_dict={
                        self.target: x,
                        self.length: len(x),
                        self.is_training: False,
                        self.is_validating: True,
                    }
                )


class Conditional(ModelBase):
    def __init__(self, **kwargs):
        ModelBase.__init__(self, 'Conditional', **kwargs)

    def _create(self, **kwargs):
        M = self.output_mixture_size = kwargs.get('output_mixture_size', 20)
        K = self.window_mixture_size = kwargs.get('window_mixture_size', 10)
        self.num_units = kwargs.get('num_units', 400)
        self.num_layers = kwargs.get('num_layers', 1)
        self.num_chars = len(kwargs['character_types']) + 1
        with tf.name_scope(self.name) as scope:
            tf.placeholder('int32', [None], name='text_length')
            tf.placeholder('string', [None, None], name='text')

            tf.placeholder('int32', [None], name='target_length')
            tf.placeholder('float', [None, None, 4], name='target')

            tf.placeholder('bool', [], name='is_training')
            tf.placeholder('bool', [], name='is_validating')

            tf.placeholder('float', [], name='learning_rate')
            tf.placeholder('float', [], name='clip_threshold')
            tf.placeholder_with_default(
                tf.constant(1.0, 'float'),
                shape=[],
                name='keep_prob'
            )

            # For sampling
            tf.placeholder_with_default(
                tf.constant(0.0, 'float'),
                shape=[],
                name='sample_bias'
            )
            tf.placeholder_with_default(
                tf.constant(0.5, 'float'),
                shape=[],
                name='finish_line'
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
            with tf.variable_scope(
                'rnn',
                initializer=init
            ):
                rnn1 = tf.contrib.rnn.GRUCell(
                    num_units=self.num_units,
                    activation=tf.nn.tanh
                )
                rnn2 = tf.contrib.rnn.GRUCell(
                    num_units=self.num_units,
                    activation=tf.nn.tanh
                )

                def condition(t, x, *_):
                    return tf.cond(
                        tf.logical_or(self.is_training, self.is_validating),
                        lambda: t < self.max_target_length,
                        lambda: tf.logical_and(
                            tf.cond(
                                tf.equal(self.max_target_length, 0),
                                lambda: tf.constant(True),
                                lambda: t < self.max_target_length
                            ),
                            tf.reduce_any(x[:,0] < self.finish_line)
                        )
                    )

                def forward(t, x, q1, q2, k, w, record):
                    def dropout(v):
                        return tf.nn.dropout(
                            v,
                            tf.cond(
                                self.is_training,
                                lambda: self.keep_prob,
                                lambda: tf.constant(1.0, 'float')
                            )
                        )

                    h, q1 = rnn1(tf.concat([x, w], axis=-1), q1)
                    h = dropout(h)

                    z_ = tf.reshape(tf.layers.dense(
                        h,
                        3*K,
                        kernel_initializer=init
                    ), [-1, 3*K])
                    z_ = dropout(z_)

                    a = tf.exp(z_[:,:K])
                    b = tf.exp(z_[:,K:2*K])
                    k += tf.exp(z_[:,2*K:])

                    u = tf.tile(
                        tf.reshape(
                            tf.range(1, tf.cast(self.max_text_length, 'float') + 1),
                            [1, 1, -1]
                        ), [self.batch_size, K, 1])
                    c = tf.one_hot(
                        character_table.lookup(self.text),
                        self.num_chars
                    )
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

                    for d in range(self.num_layers):
                        h, q2[d] = rnn2(tf.concat([x, h, w], axis=-1),
                                        tf.reshape(q2[d], [-1, self.num_units]))
                        h = dropout(h)

                    y_ = tf.reshape(tf.layers.dense(
                        h,
                        6*M+2,
                        kernel_initializer=init
                    ), [-1, 6*M+2])
                    y_ = dropout(y_)

                    # Whether to finish writing
                    f = tf.sigmoid(-y_[:,0:1])
                    e = tf.sigmoid(-y_[:,1:2])
                    p = tf.nn.softmax(y_[:,2:M+2] * (1.0 + self.sample_bias))
                    m1 = y_[:,M+2:2*M+2]
                    m2 = y_[:,2*M+2:3*M+2]
                    s1 = tf.exp(y_[:,3*M+2:4*M+2] - self.sample_bias)
                    s2 = tf.exp(y_[:,4*M+2:5*M+2] - self.sample_bias)
                    r = tf.nn.tanh(y_[:,5*M+2:6*M+2])

                    finish_pdf = tfd.Bernoulli(probs=tf.reshape(f, [self.batch_size]), dtype='float')
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

                    compute_loss = x[:,0] < self.finish_line
                    x = tf.cond(
                        tf.logical_or(self.is_training, self.is_validating),
                        lambda: target_ta.read(t),
                        lambda: tf.concat(
                            [tf.where(f < self.finish_line,
                                      tf.zeros([self.batch_size, 1], dtype='float'),
                                      tf.ones([self.batch_size, 1], dtype='float')),
                             tf.expand_dims(lift_pdf.sample(), 1),
                             offset_pdf.sample()],
                            axis=-1
                        )
                    )
                    x_ta = record.stroke.write(t, x)

                    loss = tf.where(
                        compute_loss,
                        -(
                            tf.maximum(offset_pdf.log_prob(x[:,-2:]), -1e20) +
                            lift_pdf.log_prob(x[:,1]) +
                            finish_pdf.log_prob(x[:,0])
                        )  / tf.cast(self.target_length, 'float'),
                        tf.zeros([self.batch_size])
                    )
                    loss_ta = record.loss.write(t, loss)

                    sse = tf.where(
                        compute_loss,
                        tf.reduce_sum(
                            tf.square(x - tf.concat([f, e, offset_pdf.mean()],
                                                    axis=-1)),
                            axis=-1
                        ),
                        tf.zeros([self.batch_size])
                    )
                    sse_ta = record.sse.write(t, sse)

                    param_ta = record.param.write(t, tf.concat([
                        f, e, p, m1, m2, s1, s2, r
                    ], axis=-1))

                    return t + 1, \
                           tf.reshape(x, [self.batch_size, 4]), \
                           q1, q2, \
                           k, w, \
                           Record(x_ta, loss_ta, sse_ta, param_ta)

                time = tf.constant(0, dtype='int32')

                target_ta = tf.TensorArray(
                    'float',
                    # size=self.batch_size
                    size=self.max_target_length
                ).unstack(tf.transpose(self.target, perm=[1,0,2]))

                Record = namedtuple(
                    'Record',
                    ['stroke', 'loss', 'sse', 'param']
                )

                record = tf.while_loop(
                    condition,
                    forward,
                    [time,
                     tf.zeros([self.batch_size, 4], dtype='float'),
                     rnn1.zero_state(self.batch_size, dtype='float'),
                     [rnn2.zero_state(self.batch_size, dtype='float')
                      for d in range(self.num_layers)],
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

    def predict(self, t, tl, x, xl, train=False, **kwargs):
        session = tf.get_default_session()
        learning_rate = kwargs.get('learning_rate', 1e-4)
        clip_threshold = kwargs.get('clip_threshold', 100)
        keep_prob = kwargs.get('keep_prob', 1.0)
        batch_size = len(t)

        if train:
            return session.run(
                [self.stroke, self.loss, self.sse, self.optimization],
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
                [self.stroke, self.loss, self.sse],
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
        finish_line = kwargs.get('finish_line', 0.5)
        max_length = kwargs.get('max_length', 1000)

        return session.run(
            [self.stroke],
            feed_dict={
                self.text: [t],
                self.text_length: [len(t)],
                self.target: np.zeros([1,1,4], 'float32'),
                self.target_length: [max_length],
                self.is_training: False,
                self.is_validating: False,
                self.sample_bias: sample_bias,
                self.finish_line: finish_line
            }
        )

MDLDEF = path.join(path.dirname(path.dirname(path.realpath(__file__))), 'mdldef')
def generate_unconditionally(random_seed=1, epoch=1):
    # Input:
    #   random_seed - integer

    # Output:
    #   stroke - numpy 2D-array (T x 3)

    with tf.Session().as_default() as session:
        model = Unconditional(mdldir='../mdldef/unconditional', epoch=epoch)
        model.sampler.randomizer = np.random.RandomState(random_seed)
        stroke, = model.predict(750)

    # strokes = np.load('../data/strokes.npy', encoding='bytes')
    # stroke = strokes[0]
    return stroke


def generate_conditionally(text='welcome to lyrebird', random_seed=1,
                           model='conditional', epoch=0,
                           sample_bias=0.0, finish_line=0.5):
    with tf.Session().as_default() as session:
        conditional = Conditional(mdldir=path.join(MDLDEF, model), epoch=epoch)
        session.run(tf.tables_initializer())
        stroke, = conditional.synth(np.array([c for c in text]),
                                    sample_bias=sample_bias,
                                    finish_line=finish_line)
    return stroke[0]


# if __name__ == '__main__':
#     # stroke = generate_unconditionally(epoch=int(sys.argv[1]))
#     # print(np.count_nonzero(stroke[:,0]), len(stroke[:,0]))
#     # # print(stroke[:20])
#     # plot_stroke(stroke)
#
#     stroke = generate_conditionally(epoch=int(sys.argv[1]))
#     print(stroke)
#     plot_stroke(stroke)

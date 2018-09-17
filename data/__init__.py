import numpy as np
import tensorflow as tf
from os import path


DATA_PATH = path.dirname(path.realpath(__file__))

def read_strokes(sort=True):
    with open(path.join(DATA_PATH, 'strokes.npy'), 'rb') as f:
        strokes = np.load(f, encoding='bytes')
    for n in range(len(strokes)):
        finish = np.zeros([len(strokes[n]), 1])
        finish[-1,0] = 1.0
        strokes[n] = np.concatenate([finish, strokes[n]], axis=-1)
    return strokes


def sort_strokes(strokes):
    return strokes[np.argsort([len(s) for s in strokes])]


def standardize_strokes(strokes):
    flattened = np.concatenate([s.reshape(-1, 4) for s in strokes])
    offset_mean = np.mean(flattened[:,-2:], axis=0)
    offset_scale = np.std(flattened[:,-2:], axis=0)

    return np.array([
        np.array([
            np.concatenate([p[:-2], (p[-2:] - offset_mean) / offset_scale])
            for p in s
        ])
        for s in strokes
    ]), offset_mean, offset_scale


def read_sentences():
    with open(path.join(DATA_PATH, 'sentences.txt')) as f:
        sentences = []
        flattened = []
        for l in f:
            characters = [c for c in l.rstrip()]
            sentences.append(np.array(characters))
            flattened.extend(characters)
    return np.array(sentences), np.unique(np.array(flattened))


def unconditional_dataset(strokes, batch_size=1):
    s_ds = tf.data.Dataset.from_generator(
        lambda: strokes,
        'float'
    )
    sl_ds = tf.data.Dataset.from_generator(
        lambda: [len(s) for s in strokes],
        'int32'
    )
    return tf.data.Dataset.zip(
        (s_ds, sl_ds)
    ).padded_batch(
        batch_size,
        ([None, 4], []),
        (1.0, 0)
    )


def conditional_dataset(text, strokes, batch_size=1):
    t_ds = tf.data.Dataset.from_generator(
        lambda: text,
        'string'
    )
    tl_ds = tf.data.Dataset.from_generator(
        lambda: [len(t) for t in text],
        'int32'
    )
    s_ds = tf.data.Dataset.from_generator(
        lambda: strokes,
        'float'
    )
    sl_ds = tf.data.Dataset.from_generator(
        lambda: [len(s) for s in strokes],
        'int32'
    )
    return tf.data.Dataset.zip(
        (t_ds, tl_ds, s_ds, sl_ds)
    ).padded_batch(
        batch_size,
        ([None], [], [None, 4], []),
        ('unk', 0, 1.0, 0)
    )

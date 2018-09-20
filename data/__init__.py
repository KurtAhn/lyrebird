import numpy as np
import tensorflow as tf
from os import path


DATA_PATH = path.dirname(path.realpath(__file__))
STROKE_DIM = 3

def read_strokes():
    """
    Read in stroke.npy
    """
    with open(path.join(DATA_PATH, 'strokes.npy'), 'rb') as f:
        strokes = np.load(f, encoding='bytes')
    return strokes


def standardize_strokes(strokes):
    """
    Standardize stroke offsets. Pen tip lift probability is not standardized.
    (This was an oversight.)
    """
    flattened = np.concatenate([s.reshape(-1, STROKE_DIM) for s in strokes])
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
    """
    Read in sentence.txt and flatten it into individual characters.
    Return the character sequences along with an array of all character types
    appearing in the file
    """
    with open(path.join(DATA_PATH, 'sentences.txt')) as f:
        sentences = []
        flattened = []
        for l in f:
            characters = [c for c in l.rstrip()]
            sentences.append(np.array(characters))
            flattened.extend(characters)
    return np.array(sentences), np.unique(np.array(flattened))


def unconditional_dataset(strokes, batch_size=1):
    """
    Create dataset for training the prediction network
    Organized as (Stroke point sequence (dim: length x 3), Sequence length (scalar))
    """
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
        ([None, STROKE_DIM], [])
    )


def conditional_dataset(text, strokes, batch_size=1):
    """
    Create dataset for training the synthesis network
    Organized as (Character sequence (dim: length),
                  Character sequence length (scalar),
                  Stroke point sequence (dim: length x 3),
                  Stroke length (scalar))
    """
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
        ([None], [], [None, STROKE_DIM], []),
    )

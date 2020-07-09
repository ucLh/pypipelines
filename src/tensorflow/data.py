"""Classes for managing datasets in tensorflow
"""
from __future__ import absolute_import, division, print_function
from functools import partial

import tensorflow as tf


class SeverstalData:

    def __init__(self,
                 image_paths,
                 session,
                 batch_size=10,
                 num_channels=3,
                 resize=False,
                 use_flip=False,
                 size=256,
                 drop_remainder=True,
                 num_threads=16,
                 shuffle=False,
                 buffer_size=2048,
                 repeat=1):
        self._sess = session
        self.image_paths = image_paths
        self._batch_size = batch_size
        self._num_channels = num_channels
        self._resize = resize
        self._use_flip = use_flip
        self._size = size
        self._drop_remainder = drop_remainder
        self._num_threads = num_threads
        self._shuffle = shuffle
        self._buffer_size = buffer_size
        self._repeat = repeat
        self._img_batch = tf.compat.v1.data.make_one_shot_iterator(self._image_batch()).get_next()
        self._img_num = len(image_paths)

    def __len__(self):
        return self._img_num

    def batch(self):
        return self._sess.run(self._img_batch)

    @staticmethod
    def _normalize(img):
        img = img / 255
        mean = tf.constant((0.485, 0.456, 0.406), dtype=tf.float32)
        std = tf.constant((0.229, 0.224, 0.225), dtype=tf.float32)
        img = img - mean
        img = img / std
        return img

    def _parse_func(self, path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=self._num_channels, dct_method='INTEGER_ACCURATE')
        if self._use_flip:
            img = tf.image.flip_left_right(img)
        if self._resize:
            img = tf.image.resize(img, [self._size, self._size])
        img = self._normalize(img)
        img = tf.transpose(img, perm=[2, 0, 1])
        return img

    def _map_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.image_paths)
        return dataset.map(self._parse_func, num_parallel_calls=self._num_threads)

    def _image_batch(self):
        dataset = self._map_dataset()

        if self._shuffle:
            dataset = dataset.shuffle(self._buffer_size)

        dataset = dataset.batch(self._batch_size, drop_remainder=self._drop_remainder)
        dataset = dataset.repeat(self._repeat).prefetch(2)

        return dataset



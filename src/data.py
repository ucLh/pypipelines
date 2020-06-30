"""Classes for managing datasets in tensorflow
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf

class ImageDataRaw:

    def __init__(self,
                 image_paths,
                 session,
                 batch_size=10,
                 num_channels=3,
                 resize=False,
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
        self._size = size
        self._drop_remainder = drop_remainder
        self._num_threads = num_threads
        self._shuffle = shuffle
        self._buffer_size = buffer_size
        self._repeat = repeat
        self._img_batch = self._image_batch().make_one_shot_iterator().get_next()
        self._img_num = len(image_paths)

    def __len__(self):
        return self._img_num

    def batch(self):
        return self._sess.run(self._img_batch)

    def _parse_func(self, path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=self._num_channels, dct_method='INTEGER_ACCURATE')
        if self._resize:
            img = tf.image.resize(img, [self._size, self._size])
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



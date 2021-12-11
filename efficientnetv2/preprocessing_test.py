# Copyright 2021 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for preprocessing."""
from absl import logging
from absl.testing import parameterized
import tensorflow as tf
from tensorflow.python.data import AUTOTUNE

import preprocessing


class PreprocessingTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters('effnetv1_autoaug', 'effnetv1_randaug', None)
  def test_preprocessing_legacy(self, augname):
    image = tf.zeros((300, 300, 3), dtype=tf.float32)
    try:
      preprocessing.preprocess_image(image, 224, False, None, augname)
    except tf.errors.InvalidArgumentError as e:
      if 'ExtractJpegShape' not in str(e):
        raise e

  @parameterized.parameters('autoaug', 'randaug', 'ft', 'ft_autoaug', None)
  def test_preprocessing(self, augname):
    image = tf.zeros((300, 300, 3), dtype=tf.float32)
    preprocessing.preprocess_image(image, 224, True, None, augname)

  def test_wrap(self):
    import numpy as np
    print(np.__version__)
    dataset = self.get_ds()
    for img in dataset.take(1).map(self.wrap, num_parallel_calls=AUTOTUNE, deterministic=True):
      image = img
    tf.print(image[0].shape)

  def get_ds(self)-> tf.data.Dataset:
      import tensorflow_datasets as tfds
      return tfds.load(name="hyperkvasir_li/no_aug",split="split_0",as_supervised=True)

  @staticmethod
  def wrap(image, label):
    """Returns 'image' with an extra channel set to all 1s."""
    shape = tf.shape(image)
    extended_channel = tf.ones([shape[0], shape[1], 1], image.dtype)
    extended = tf.concat([image, extended_channel], 2)
    return extended, label


if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  tf.test.main()

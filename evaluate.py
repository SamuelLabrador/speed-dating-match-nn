from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow import keras
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import KFold

import pandas as pd
import numpy as np

model = tf.keras.models.load_model('saved_models/dating-model.h5')
model.summary()

data_file = 'data/cleaned_data.csv'
source_dataset = pd.read_csv(data_file)

dataset = source_dataset.copy()

labels = dataset.pop('match')

model.evaluate(dataset, labels)
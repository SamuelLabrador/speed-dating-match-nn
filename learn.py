from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow import keras
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import KFold

import pandas as pd
import numpy as np

data_file = 'data/cleaned_data.csv'
source_dataset = pd.read_csv(data_file)

dataset = source_dataset.copy()

# Partition Datasets
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Extract labels
train_labels = train_dataset.pop('match')
test_labels = test_dataset.pop('match')

print(len(train_dataset.keys()))
model = keras.Sequential([
		keras.layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
		keras.layers.Dense(16, activation='relu'),
		keras.layers.Dense(1, activation='sigmoid'),
	])

model.compile(
	optimizer='adam',
	loss='binary_crossentropy',
	metrics=['accuracy'],
	)

history = model.fit(train_dataset, 
					train_labels,
					epochs=15,
					verbose=1,
					)

entry = model.evaluate(test_dataset, test_labels)

model.save('saved_models/dating-model.h5')

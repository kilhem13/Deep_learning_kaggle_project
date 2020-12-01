import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import feature_column


from keras.models import Sequential
from keras.layers import Dense, Input
from pydoc import locate


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
table_features = []
table_labels = []
def show_batch(dataset):
  for batch, label in dataset.take(1):
    for key, value in batch.items():
        table_features.append(value)
        print("{:20s}: {}".format(key,value.numpy()))
    for lab in label:
        table_labels.append(label)
        print(lab)

def df_to_dataset(dataframe, labels, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  #labels = dataframe.pop('target')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds


def _preprocess_line(features, targets):
    # Pack the result into a dictionary
    features = dict(zip(visu_data.columns, features))
    #print(features)
    features.pop('sig_id')
    features.pop('cp_time')
    #targets.pop('sig_id')
    targets = tf.stack(targets[1:])
    return features, targets





for dirname, _, filenames in os.walk('/Data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


visu_data = pd.read_csv('./Data/train_features.csv')
visu_label = pd.read_csv('./Data/train_targets_scored.csv')
visu_data.head()

X_train = pd.DataFrame(pd.read_csv('./Data/train_features.csv', nrows=5))
X_train.dtypes
types = [str(), str(), str(), str()]
for i in range(4, len(X_train.dtypes)):
    if X_train.dtypes[i].kind == 'f':
        types.append(float())


features = tf.data.experimental.CsvDataset('./Data/train_features.csv', record_defaults=types, header=True)


y = pd.DataFrame(pd.read_csv('./Data/train_targets_scored.csv', nrows=5))
types = ['']
for i in range(1, len(y.dtypes)):
    types.append(float())

targets = tf.data.experimental.CsvDataset('./Data/train_targets_scored.csv', record_defaults=types, header=True)

dataset = tf.data.Dataset.zip((features, targets))

dataset_size = dataset.reduce(np.int64(0), lambda x, _:x+1).numpy()
train_dataset = dataset.take(0.7*dataset_size)
val_dataset = dataset.skip(0.7*dataset_size)
val_dataset = dataset.take(dataset_size - 0.7*dataset_size)

train_dataset = train_dataset.map(_preprocess_line).batch(32)
val_dataset = val_dataset.map(_preprocess_line).batch(32)


all_columns = list(list(train_dataset.element_spec)[0].keys())
categorical_columns = [all_columns[0], all_columns[1]]
numerical_columns = all_columns[2:]
#numerical_columns.append(all_columns[1])

feature_columns = []

for col in categorical_columns:
    categorical_column = feature_column.categorical_column_with_vocabulary_list(col, visu_data[col].unique())
    indicator_column = feature_column.indicator_column(categorical_column)
    feature_columns.append(indicator_column)

for col in numerical_columns:
    feature_columns.append(feature_column.numeric_column(col))

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

model = tf.keras.Sequential([
    feature_layer,
    layers.Dense(2048, activation='relu'),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(.1),
    layers.Dense(206)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_dataset,
          validation_data=val_dataset,
          epochs=10)
show_batch(train_dataset)
import cv2
import csv
from enum import Enum
import keras
import numpy as np
import os
from pathlib import Path
import random
import sys
import tensorflow
import time

try:
  sys.path.append(str(Path('.').resolve().parent))
  from misc.progress_bar import printProgressBar
  import datasets.split_images as split_images
except Exception as e:
  print(e)


class SherdType(Enum):
  Kanaa = 0
  Black_Mesa = 1
  Sosi = 2
  Dogoszhi = 3
  Flagstaff = 4
  Tusayan = 5
  Kayenta = 6


def create_training_test_sets(image_dir, training_split):
  """
  Since the training data we have for each class is disproportionate, we need to
  ensure that the testing set contains the same ratio of classes as found in the
  full image set.

  @params:
    image_dir: Full path to the image directory.
    training_split: floating point of percentage of data to use for training
  """
  image_count = [0] * 7
  training_image_count = [0] * 7
  data_dict = split_images.consolidate_image_data(image_dir)
  training_set_filename = image_dir / 'training_list.csv'
  test_set_filename = image_dir / 'test_list.csv'

  # Get the number of sherds per type 
  for val in data_dict.values():
    image_count[SherdType[val].value] += 1
  
  # Shuffle dictionary keys to create shuffled training and testing sets
  print(f'Shuffling dictionary keys')
  keys = list(data_dict.keys())
  random.shuffle(keys)
  # Open csv files for writing
  try:
    with open(training_set_filename, 'w') as training_csv, open(test_set_filename, 'w') as test_csv:
      training_writer = csv.writer(training_csv)
      test_writer = csv.writer(test_csv)
      rows_written = 0
      # Loop through dictionary using shuffled keys
      for key in keys:
        label = data_dict[key]
        idx = SherdType[label].value
        rows_written += 1
        # If current ratio less than training split add image to training set
        if training_image_count[idx] / image_count[idx] < training_split:
          training_writer.writerow([key, label])
          training_image_count[idx] += 1
        else:
          test_writer.writerow([key, label])
        if rows_written % 70 == 0:
          printProgressBar(rows_written, len(keys), 'Progress', 'Complete', length=50)
          time.sleep(0.1)
      printProgressBar(len(keys), len(keys), 'Progress', 'Complete', length=50)
          
  except IOError as e:
    sys.stderr.write(f"Error opening file for writing\n {e}")
    return False
  
  split_images.categorize_images(image_dir, 'tusayan_whiteware')
  return True


def load_data(image_dimension, training_split, batch_size=32, regenerate=False):
  """
  @params:
    image_dimension: a tuple containing the size to resize images (height, width)
    training_split: a tuple (training, validation, testing) which takes in
    doubles in the range 0.0-1.0. Must add up to 1.0
    batch_size: (int) size of batches of data, default 32
  @returns:
    A tuple of datasets
  """
  train_split = training_split[0]
  val_split = training_split[1]
  test_split = training_split[2]

  imagelist_path = Path('.').resolve().parents[1] / 'image_data' / 'image_list.csv'
  tusayan_ww_path = imagelist_path.parent / 'tusayan_whiteware'

  if not tusayan_ww_path.exists() or regenerate:
    # Create both sets
    print(f'Creating training and test sets')
    create_training_test_sets(imagelist_path.parent, train_split + val_split)
  # Load dataset from directory
  if val_split != 0.0:
    print(f'Loading Training and Validation datasets')
    train_ds, val_ds = keras.utils.image_dataset_from_directory(
      directory=str(tusayan_ww_path / 'Training'),
      labels='inferred',
      label_mode='categorical',
      class_names=[label.name for label in SherdType],
      batch_size=batch_size,
      image_size=image_dimension,
      seed=random.randint(1, 999999),
      validation_split=val_split,
      subset='both',
      interpolation='area'
    )
  else:
    print(f'Loading Training dataset')
    train_ds = keras.utils.image_dataset_from_directory(
      directory=str(tusayan_ww_path / 'Training'),
      labels='inferred',
      label_mode='categorical',
      class_names=[label.name for label in SherdType],
      batch_size=batch_size,
      image_size=image_dimension,
      seed=random.randint(1, 999999),
      validation_split=None,
      subset=None,
      interpolation='area'
    )
  print(f'Loading Test dataset')
  test_ds = keras.utils.image_dataset_from_directory(
    directory=str(tusayan_ww_path / 'Testing'),
    labels='inferred',
    label_mode='categorical',
    class_names=[label.name for label in SherdType],
    batch_size=batch_size,
    image_size=image_dimension,
    shuffle=True,
    seed=random.randint(1, 999999),
    validation_split=None,
    subset=None,
    interpolation='area'
  )
  if val_split != 0.0:
    return (train_ds, val_ds, test_ds)
  elif test_split != 0.0:
    return (train_ds, test_ds)
  else:
    return train_ds
import csv
import cv2
import os
from pathlib import Path
import re
import shutil
import sys
import time

try:
  sys.path.append(str(Path('.').resolve().parent))
  from misc.progress_bar import printProgressBar
except Exception as e:
  print(e)


def categorize_images(image_dir, dir_name):
  """
  @params
    image_dir: Full path to the image directory.
    dir_name: String of the desired name to give to parent directory where 
    images are stored.
  """
  print(f'Categorizing images')
  images_dir = image_dir / 'images'
  categorized_dir = image_dir / dir_name
  training_dir = categorized_dir / 'Training'
  test_dir = categorized_dir / 'Testing'
  if categorized_dir.exists():
    shutil.rmtree(categorized_dir)
    print(f'Removing old {categorized_dir}')
  print(f'Creating {categorized_dir}')
  categorized_dir.mkdir()
  if training_dir.exists():
    shutil.rmtree(training_dir)
    print(f'Removing old {training_dir}')
  print(f'Creating {training_dir}')
  training_dir.mkdir()
  if test_dir.exists():
    shutil.rmtree(test_dir)
    print(f'Removing old {test_dir}')
  print(f'Creating {test_dir}')
  test_dir.mkdir()
  
  # Find csv files for training and testing dynamically
  files = os.listdir(image_dir)
  print(f'Searching for training and test csv files')
  # Copy all training images into training dir
  for file in files:
    training_match = re.search('training', file)
    test_match = re.search('test', file)
    if training_match:
      print(f'{training_match.string} found')
      training_list = image_dir / training_match.string
      copy_images(training_list, images_dir, training_dir)
    # Copy all test images into test dir
    if test_match:
      print(f'{test_match.string} found')
      test_list = image_dir / test_match.string
      copy_images(test_list, images_dir, test_dir)



def consolidate_image_data(image_dir): 
  """
  Image data was originally split into set directories for cross-fold 
  validation. Labels were stored in csv files for each set along with the file
  name of the image. This function goes through every set directory and copies
  the image file names and labels to a main csv file that contains all data.

  @params:
    image_dir: Full path to the image directory.  
  @returns:
    A dictionary containing every image file name and corresponding label
  """
  # Search all subdirectories for directories named Set_
  set_pattern = "Set_"
  filetype_pattern = ".csv"
  data_dict = {}

  print('Consolidating image data')
  for (dir, subdirs, files) in os.walk(image_dir):
    match = re.search(set_pattern, dir)
    if match:
      print(f'Directory {match.string} found, searching for csv files')
      for filename in files:
        match = re.search(filetype_pattern, filename)
        if match:
          print(f'File {filename} found')
          file_path = Path(dir) / filename
          read_csv(file_path, data_dict)
  print(f'Searching for csv files completed')

  full_imagelist_path = os.path.join(image_dir, "image_list.csv")
  print(f'Writing dictionary to {full_imagelist_path}')
  try:
    with open(full_imagelist_path, 'w') as csv_file:
      csv_writer = csv.writer(csv_file)
      rows_read = 0
      total_rows = len(data_dict.items())
      for row in data_dict.items():
        rows_read += 1
        csv_writer.writerow(row)
        if rows_read % 70 == 0:
          printProgressBar(rows_read, total_rows, 'Progress', 'Complete', length=50)
          time.sleep(0.1)
      printProgressBar(total_rows, total_rows, 'Progress', 'Complete', length=50)
  except IOError as e:
    sys.stderr.write(f"Error opening {filename} for writing\n {e}")
    return False
  return data_dict


def copy_images(image_list, source, destination, gray_scale=True):
  images_copied = 0
  data_dict = {}
  read_csv(image_list, data_dict)
  print(f'Copying images from {source} to {destination}')
  if gray_scale:
    print(f'Augmenting images through grayscale')
  for (image_name, label) in data_dict.items():
    categorized_image_dir = destination / label
    if not categorized_image_dir.exists():
      categorized_image_dir.mkdir()
    if gray_scale:
      image = cv2.imread(str(source / image_name))
      image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
      cv2.imwrite(str(categorized_image_dir / image_name), image)
    else:
      shutil.copy(source / image_name, categorized_image_dir / image_name)
    images_copied += 1
    if images_copied % 70 == 0:
      printProgressBar(images_copied, len(data_dict.keys()), 'Progress', 'Complete', length=50)
      time.sleep(0.1)
  printProgressBar(len(data_dict.keys()), len(data_dict.keys()), 'Progress', 'Complete', length=50)
  

def read_csv(file_path, data_dict):
  """
  Read in csv file data into dictionary.
  @params:
    file_path: Full path to csv file.
    data_dict: Python dictionary to hold data.
  """
  read_bytes = 0
  idx = 0
  total_bytes = os.path.getsize(file_path)
  
  try:
    with open(file_path, 'r') as csv_file:
      print(f'Reading {file_path} into dictionary')
      for row in csv.reader(csv_file):
        if idx % 150 == 0:
          printProgressBar(read_bytes, total_bytes, 'Progress', 'Complete', length=50)
          time.sleep(0.1)
        data_dict[row[0]] = row[1]
        read_bytes += (len(row[0]) + len(row[1]) + 2)
        idx += 1
      printProgressBar(total_bytes, total_bytes, 'Progress', 'Complete', length=50)
    return True
  except IOError as e:
    sys.stderr.write(f"Error opening {file_path} for reading\n {e}")
    return False

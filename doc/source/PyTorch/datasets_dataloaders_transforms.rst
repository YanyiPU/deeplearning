.. _header-n0:

构造自定义的 Datasets, Dataloaders, Transforms
==============================================

.. _header-n3:

依赖库
------

.. code:: python

   from __future__ import print_function, division
   import os
   import torch
   import pandas as pd 
   from skimage import io, transform
   import numpy as np 
   import matplotlib.pyplot as plt 
   from torch.utils.data import Dataset, DataLoader
   from torchvision import transform, utils

   # Ignore warnings
   import warnings
   warnings.filterwarnings("ignore")

   plt.ion() # interactive mode

.. _header-n5:

读取数据
--------

.. code:: python

   landmarks_frame = pd.read_csv("../data/faces/face_landmarks.csv")
   n = 65
   img_name = landmarks_frame.iloc[n, 0]
   landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
   landmarks = landmarks.astype("float").reshape(-1, 2)


   print("Image name: {}".format(img_name))
   print("Landmarks shape: {}".format(landmarks.shape))
   print("First 4 Landmarks: {}".format(landmarks[:4]))

.. code:: python

   def show_landmarks(image, landmarks):
   	"""show image with landmarks"""
   	plt.imshow(image)
   	plt.scatter(landmarks[:, 0], landmarks[:, 1], s = 10, marker = ".", c = "r")
   	plt.pause(0.001)

   plt.figure()
   show_landmarks(io.imread(os.path.join("../data/faces/", img_name)), landmarks)

.. _header-n9:

Dataset class
-------------

.. code:: python

   class FaceLandmarksDataset(Dataset):
   	"""Face Landmarks dataset."""
   	def __init__(self, csv_file, root_dir, transform = None):
   		pass

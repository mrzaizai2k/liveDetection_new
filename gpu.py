import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

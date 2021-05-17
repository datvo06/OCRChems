import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel.data_parallel import data_parallel

from torch.nn.utils.rnn import *


def seed_torch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
#     return seed

import os
from datetime import datetime

import math
import numpy as np
import random
import PIL
import cv2
import matplotlib

# std libs
import collections
from collections import defaultdict
import copy
import numbers
import inspect
import shutil
from timeit import default_timer as timer
import itertools
from collections import OrderedDict
from multiprocessing import Pool
import multiprocessing as mp

import json
import zipfile
from shutil import copyfile

import csv
import pandas as pd
import pickle
import glob
import sys
from distutils.dir_util import copy_tree
import time
import itertools as it

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


PI  = np.pi
INF = np.inf
EPS = 1e-12

def seed_py(seed):
    random.seed(seed)
    np.random.seed(seed)
#     return seed

#Do this at the start of every runtime
!pip install PyWavelets
!pip install pyentrp
!pip install eeglib

import math
from scipy.stats import skew, kurtosis
import numpy as np
import pywt
import csv
from pyentrp import entropy as ent
import statsmodels.api as sm
import eeglib
import scipy.signal as signal

import scipy.io as spio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict
import pandas as pd
import json
import torch.nn.functional as F


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm

from collections import defaultdict
import pandas as pd
import json

from google.colab import drive
drive.mount('/content/drive')
! cp -r /content/drive/MyDrive/NinaproDB1 /content
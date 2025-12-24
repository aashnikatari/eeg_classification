
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from mne.io import read_raw_brainvision
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Deep Learning (LSTM)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Configuration
TARGET_SFREQ = 256  # Target sampling frequency (to match Mendeley dataset)
EPOCH_DURATION = 10  # Seconds per epoch
FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha1': (8, 10),
    'alpha2': (10, 13),
    'beta1': (13, 20),
    'beta2': (20, 30)
}

# Common channels between TDBRAIN (10-10) and Mendeley (10-20)
COMMON_CHANNELS = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
                   'T7', 'C3', 'Cz', 'C4', 'T8',
                   'P7', 'P3', 'Pz', 'P4', 'P8',
                   'O1', 'O2']

# print("config.py recreated successfully!")
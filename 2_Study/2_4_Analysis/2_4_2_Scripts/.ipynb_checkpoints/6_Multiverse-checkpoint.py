# Multiverse

# single subject

# single trial 

from ctypes.wintypes import SIZE
from distutils.log import set_verbosity
from turtle import st
import kabuki
import os
import hddm
import matplotlib.pyplot as plt
import pickle
from functools import partial
from p_tqdm import p_map
from glob import glob
from kabuki.analyze import gelman_rubin
import arviz
import numpy as np
from patsy import dmatrix 
from sklearn.preprocessing import StandardScaler


# joint model - single trial


## behavior model
m1: hddm

m2: hddm with sz st sv

m3: hddmregressor v~coherency

m4: hddmregressor v~coherency z~attention

m5: hddmregressor v~coherency t~attention


## joint model

m6: hddmregressor v~coherency slope z~attention

m7: hddmregressor v~coherency amplitudev z~attention

m8: hddmregressor v~coherency peak z~attention


# joint model - single subject
data = hddm.load_csv('/home/mw/input/cpp7572/data.csv')

data['rt'] = data['rt'] + 0.2 # stim time

data['response'] = data['_responnse']

data['cpp_peak'] = StandardScaler().fit_transform(data['cpp_peak'].to_numpy())

data['cpp_amplitude'] = StandardScaler().fit_transform(data['cpp_amplitude'].to_numpy())

data['cpp_slope'] = StandardScaler().fit_transform(data['cpp_slope'].to_numpy())

data.groupby('subj_idx').cpp_peak = data.groupby('subj_idx').cpp_peak.mean()

data.groupby('subj_idx').cpp_amplitude = data.groupby('subj_idx').cpp_amplitude.mean()

data.groupby('subj_idx').cpp_slope = data.groupby('subj_idx').cpp_slope.mean()

m6

m7

m8

# two-step -single subject each condition

## best model v & cpp_peak

## best model v & cpp_amplitude

## best model v & cpp_slope

# two-step -single subject condition difference

## best model v & cpp_peak

## best model v & cpp_amplitude

## best model v & cpp_slope












# -*- coding: utf-8 -*-
import sys,os
import getpass
from pprint import pprint
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


import neo
import quantities as pq


from matplotlib import pyplot as plt
# import seaborn as sns
import spikeinterface.full as si
import platform


print('Who is there? -->',  platform.system(), getpass.getuser())
if getpass.getuser() == 'maximejuventin':

    workdir = '/Volumes/m-GCarleton/GCarleton/JUVENTIN_Maxime/project/Neuropixel_pipeline/'
    datadir = workdir+'data/'
    sortingresultsdir = workdir + 'sorting/'

elif platform.system() == 'Linux' and getpass.getuser() == 'juventin':
    workdir = '/home/users/j/juventin/Neuropixel_test/'
    datadir = workdir + '/data/'
    sortingdir = workdir + 'sorting/'
    precomputedir = workdir + 'precompute/'
    figuredir = workdir + 'generated_figures/'
    slurmdir = workdir + 'slurm/'
    sortingresultsdir = workdir + 'sorting_results/'


elif platform.system() == 'Windows' and getpass.getuser() == 'juventin' :
    workdir = 'N:/GCarleton/JUVENTIN_Maxime/project/Neuropixel_pipeline/'
    datadir = workdir + 'data/'
    sortingdir = workdir + 'sorting'
    precomputedir = workdir + 'precompute/'
    figuredir = workdir + 'generated_figures/'
    # slurmdir = workdir + 'slurm/'
    sortingresultsdir = workdir + 'sorting_results/'

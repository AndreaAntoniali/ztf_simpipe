#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 10:17:10 2023

@author: cosmostage
"""


from astropy.table import QTable, Table, Column, join, vstack
from astropy.io.misc.hdf5 import write_table_hdf5, read_table_hdf5


import numpy as np 
import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from astropy.cosmology import w0waCDM
from scipy.stats import norm
from scipy import interpolate
from astropy import units as u
import matplotlib.pyplot as plt

from optparse import OptionParser

parser = OptionParser()

import matplotlib as mpl
from cycler import cycler
import time


parser.add_option('--config_file', type=str, default='config.csv',
                  help='the file with all the configuration [%default]')

parser.add_option('--config', type=str, default='conf1',
                  help='configuration chosen in the config.csv file [%default]')

parser.add_option('--z_min', type=float, default=0.01,
                  help='minimum z value for producing simulations [%default]')

parser.add_option('--z_max', type=float, default=0.20,
                  help='maximum z value for producing simulations [%default]')

parser.add_option('--n_transient', type=int, default=100,
                  help='number of transient to produce [%default]')

parser.add_option('--delta_z', type=float, default=0.01,
                  help='the bins interval of the z[%default]')

parser.add_option('--stretch_mean', type=float, default=-2,
                  help='The stretch factor x1 mean value [%default]')

parser.add_option('--color_mean', type=float, default=0.2,
                  help='the color factor c mean value [%default]')

parser.add_option('--stretch_sigma', type=float, default=0, #1
                  help='The stretch factor sigma value[%default]')

parser.add_option('--color_sigma', type=float, default=0,#0.1
                  help='the color factor sigma value [%default]')




opts, args = parser.parse_args()
conf = opts.config
configuration = conf.split(',')
config_file = opts.config_file

zmin = opts.z_min
zmax = opts.z_max
ntransient = opts.n_transient
delta_z = opts.delta_z
stretch_mean = opts.stretch_mean
stretch_sigma= opts.stretch_sigma
color_mean = opts.color_mean
color_sigma = opts.color_sigma

print("------------------------------------------------------\n \
      Simulation parameters are : \n configuration file : {}\n Range of z : [{}, {}] \n".format(config_file,zmin, zmax))
r = pd.read_csv(config_file)

config_file = config_file.split('.')[0]
print(config_file)

    
    
def exec_fake(configuration):
    '''
    Generates the fake observations using FakeObs.py in ztf_simpipe/run_scripts/cadence

    Parameters
    ----------
    configuration : str
        refers to the configuration stored in config.csv

    Returns
    -------
    None.

    '''
    #First, we do the fake observations 
    
    cmd_fakeObs = 'python run_scripts/cadence/FakeObs.py'
    #On définit toutes les conditions d'observations 
    
    for b in 'gri':
        cmd_fakeObs += ' --cad_{}={}'.format(b, data_config['cad_{}'.format(b)].values[0])

        cmd_fakeObs += ' --N_{}={}'.format(b, data_config['N_{}'.format(b)].values[0])
        
    #On définit les différents répertoires 
    cmd_fakeObs +=' --output_dir=../fake_obs/{}/{}'.format(config_file, configuration)
    cmd_fakeObs += ' --filename=fake_data_obs_{}.hdf5'.format(configuration)
    os.system(cmd_fakeObs)
    
    #Then, we do the metric of those fake observations
    cmd_metric = 'python run_scripts/cadence/metric.py'
    cmd_metric += ' --fileName=fake_data_obs_{}.hdf5'.format(configuration)
    cmd_metric += ' --input_dir=../fake_obs/{}/{}'.format(config_file, configuration)
    cmd_metric += ' --output_dir=../fake_obs/{}/{}'.format(config_file, configuration)
    cmd_metric += ' --outName=cadenceMetric_{}.hdf5'.format(configuration)
    cmd_metric += ' --nproc=1'
    os.system(cmd_metric)
    
def exec_prod(configuration = configuration, zmin = zmin, zmax = zmax,\
              ntransient = ntransient, delta_z=delta_z, stretch_mean = stretch_mean,
              color_mean = color_mean, stretch_sigma=stretch_sigma, color_sigma=color_sigma):
    
    cmd_prod_sn = 'python run_scripts/ztfIII_cadence/prod_sn_write_dev.py'
    cmd_prod_sn += ' --config={}'.format(configuration)
    cmd_prod_sn += ' --config_file={}'.format(config_file)
    cmd_prod_sn += ' --z_min={}'.format(zmin)
    cmd_prod_sn += ' --z_max={}'.format(zmax)
    cmd_prod_sn += ' --n_transient={}'.format(ntransient)
    cmd_prod_sn += ' --delta_z={}'.format(delta_z)
    cmd_prod_sn += ' --stretch_mean={}'.format(stretch_mean)
    cmd_prod_sn += ' --color_mean={}'.format(color_mean)
    cmd_prod_sn += ' --stretch_sigma={}'.format(stretch_sigma)
    cmd_prod_sn += ' --color_sigma={}'.format(color_sigma)
    os.system(cmd_prod_sn)
    
    
start = time.time()
    
if (configuration ==['all']):
    
    configs_liste = r['config'].tolist()
    
    for i in configs_liste:
        data_config = r[r['config'] == i]
        exec_fake(i)
        exec_prod(i)
else:
    for i in configuration:
        data_config = r[r['config'] == i]
        exec_fake(i)
        exec_prod(i)
        
        
end = time.time()
print('Time of computation : \n {}s'.format(np.round(end-start, decimals=2)))




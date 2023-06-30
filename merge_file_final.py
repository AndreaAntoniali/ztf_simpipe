#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 11:37:36 2023

@author: cosmostage
"""

import warnings
warnings.filterwarnings("ignore")

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


import matplotlib as mpl
from cycler import cycler
import time

#I define the global parameters for my plots
mpl.rcParams['lines.linewidth'] = 2

mpl.rcParams['axes.prop_cycle'] = cycler(color=['k', 'r', 'g', 'y', 'b'])
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['axes.grid'] = True
mpl.rcParams['axes.grid.which'] = 'major'
mpl.rcParams['grid.linewidth'] = 1.2
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['figure.figsize'] = [10, 6]
mpl.rcParams['font.size']= 16

mpl.rcParams['legend.fontsize'] = 16

mpl.rcParams['legend.title_fontsize'] = 16
mpl.rcParams['legend.shadow'] = False
mpl.rcParams["legend.fancybox"] = False
mpl.rcParams["legend.edgecolor"] = 'white'


mpl.rcParams['xtick.minor.visible'] = False
mpl.rcParams['xtick.minor.size'] = 5
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['xtick.top'] = False

mpl.rcParams['ytick.minor.visible'] = False
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['ytick.right'] = False
mpl.rcParams['ytick.labelright'] = False

#################################################################################

def browse_directory(directory_name, extension_name = '.hdf5'):
    '''
    Function to browse the hdf5 files containing the various SNIa and 
    merge them in a single dataframe.

    Parameters
    ----------
    directory_name : string
        The directory where all the files are. 
    extension_name : str
        the name of the extension to consider to read the files. The default is '.hdf5'.

    Returns
    -------
    df : astropy table
        An astropy table of containing all the informations of the files. 

    '''
    directory = directory_name
    df = Table()
    for filename in os.listdir(directory):
        if filename.endswith(extension_name):
             path = os.path.join(directory, filename)
             df_file = read_table_hdf5(path)
             df = vstack([df, df_file])
    return df

def save_plot(path_plot, file_name):
    '''
    Saves the plot of the current figure in a given directory at the path_plot
    and with the name of file.
    If the directory indicated doesn't exist, it will create one.
    
    Parameters
    ----------
    path_plot : str
        Where to save the file
    file_name : str
        the name of the plot

    '''
    if not os.path.exists(path_plot):
        os.makedirs(path_plot)
    plt.savefig('{}/{}'.format(path_plot, file_name))





from optparse import OptionParser

parser = OptionParser()

parser.add_option('--config_file', type=str, default='config.csv',
                  help='the file with all the configuration [%default]')

parser.add_option('--config', type=str, default='conf1',
                  help='configuration chosen in the config.csv file [%default]')

parser.add_option('--plot_sig_c', type=str, default='save',
                  help='if we want to plot sigma_c on z [%default]')

parser.add_option('--plot_config_compare', type=str, default='save',
                  help='if we want to plot the comparison between the configurations given [%default]')

parser.add_option('--plot_distribution_z', type=str, default='save',
                  help='if we want to plot the distribution of z [%default]')

parser.add_option('--plot_mu_z', type=str, default='save',
                  help='if we want to plot the distance modulus in function of z [%default]')

parser.add_option('--plot_mu_distrib', type=int, default=0,
                  help='Number of supernovae to take from distribution [%default]')


opts, args = parser.parse_args()

#We get the informations from the parser. 
conf = opts.config
plot_sig_c_state = opts.plot_sig_c
plot_config_compare_state = opts.plot_config_compare
plot_distribution_z_state = opts.plot_distribution_z
plot_mu_z_state = opts.plot_mu_z
plot_mu_distrib_state = opts.plot_mu_distrib
config_file = opts.config_file
confs = conf.split(',')

print("------------------------------- \n Configuration file : \n {}".format(config_file))

#We read the file. 

r = pd.read_csv(config_file)
if confs == ['all']:
    confs= r['config'].tolist()

#We erase the .csv from the configuration file name. 
config_file= config_file.split('.')[0]

def selec(data):
    '''
    Makes a selection on the given dataframe

    Parameters
    ----------
    data : dataframe or astropy table
        The dataframe to make a selection on

    Returns
    -------
    data : panda dataframe

    '''
    data = data[data['c_err']>0]
    data = data[data['c_err']<0.3]
    data = data[data['sel']==1]
    data = data[data['fitstatus']=='fitok']
    data = data.to_pandas()
    return data




def ratio_fit(data):
    '''
    Calculates the binomial estimate and uncertainty for a given number of supernovae
    in a binned array of z. 

    Parameters
    ----------
    data : panda dataframe

    Returns
    -------
    expectation_value : float(in %)
        The expected percent of SNe Ia kept againt the others. 
    uncertainty_value : float(in %)
        The uncertainty percent of SNe Ia kept againt the others. 

    '''
    df = data
    bins = np.arange(np.min(df['z']), np.max(df['z']), 0.01)
    df_fitok= df[df['fitstatus'] == b'fitok']
    df_err= df[df['fitstatus'] != b'fitok']
    
    array_ok = df_fitok.groupby(pd.cut(df_fitok['z'], bins))['fitstatus'].count()
    array_err = df_err.groupby(pd.cut(df_err['z'], bins))['fitstatus'].count()
    
    expectation_value = array_ok/(array_ok + array_err)
    N = array_ok + array_err
    #We calculate the uncertainty with the binomial law
    uncertainty = np.sqrt(N * expectation_value*(1-expectation_value))/N
    
    
    return expectation_value *100, uncertainty *100




def plot_compare(confs, config_file, save=False):
    '''
    Calculates the redshift completness and its uncertainty for the given 
    configuration and plot them in a comparison plot. 
    

    Parameters
    ----------
    confs : list of string
        list of the different name of configuration
    config_file : str
        name of the file where are written the configuration

    '''
    print('------------------------------- \n Running : plot_compare \n\
------------------------------- \n')
    starter = time.time()
    
    redshift_dic = {}
    fig, ax = plt.subplots(figsize = (14, 5))   

    redshift_low_dic = {}
    redshift_high_dic = {}
    print('Processing : \n')
    for i in confs:
        print("configuration : {}".format(i))
        
        
        name = '../dataSN/{}/{}'.format(config_file,i)
        df = browse_directory(name)
        
        #We apply a selection on the SNIa. 
        data =selec(df)
        
        z = data['z']
        bins = np.arange(np.min(z), np.max(z), 0.01)
        
        plot_centers = (bins[:-1] + bins[1:])/2
        
        y = data.groupby(pd.cut(z, bins)).mean()
        err_y = data.groupby(pd.cut(z, bins)).std()
        
        #We estimate the redshift completness and its lower and upper bound. 
        redshift_com = interpolate.interp1d(y['c_err'], plot_centers, fill_value = 0, bounds_error=False)(0.04)
        redshift_com_lower = interpolate.interp1d(y['c_err']-err_y['c_err']\
                                                           , plot_centers, fill_value = 0, bounds_error=False)(0.04)
        redshift_com_higher = interpolate.interp1d(y['c_err']+err_y['c_err'],\
                                                            plot_centers, fill_value = 0, bounds_error=False)(0.04)
            
        
        redshift_dic[i] = float(redshift_com)   
        redshift_low_dic[i] = float(np.abs(redshift_com_lower -redshift_com))
        redshift_high_dic[i] = float(np.abs(redshift_com_higher -redshift_com))
        
    redshift_low = list(redshift_low_dic.values())
    redshift_high = list(redshift_high_dic.values())
    err_redshift = [redshift_high,redshift_low]
    end = time.time()
    
    print("------------------------------- \n Processing time of plot_compare : {} s".format(np.round(end-starter, 2)))
    
    ax.errorbar(redshift_dic.keys(), redshift_dic.values(), yerr = err_redshift,marker = 'o', linestyle='', color='r')
    ax.set_xlabel('Configurations')
    ax.set_ylabel(r'$z_{lim}$')
    
    plt.setp(ax.get_xticklabels(), rotation='vertical')
    
    ax.grid(True, which = 'major', axis = 'y')
    ax.set_title('Redshift completness for given configurations')
    plt.tight_layout()
    
    if save=='save':
        path_plot = r'../Plots/{}'.format(config_file)
        save_plot(path_plot, 'compare_{}'.format(config_file))
    else:
        plt.show()
    plt.close()


def plot_z_c(confs, config_file, save=False):
    '''
    
    Plot the uncertainty on the color parameter c againt the redshift 
    and makes an estimation of the redshift completeness. 
 

     Parameters
     ----------
     confs : list of string
     list of the different name of configuration
     config_file : str
     name of the file where are written the configuration

    '''
    
    print('------------------------------- \n Running : plot_z_c \n\
------------------------------- \n')

    path_plot = r'../Plots/{}'.format(config_file)
    for i in confs:
        name = '../dataSN/{}/{}'.format(config_file,i)
        df = browse_directory(name)
        
        data = selec(df)
        
        z = data['z']
        
        bins = np.arange(np.min(z), np.max(z), 0.01)
        
        plot_centers = (bins[:-1] + bins[1:])/2
        
        y = data.groupby(pd.cut(z, bins)).mean()
        err_y = data.groupby(pd.cut(z, bins)).std()
        
        redshift_com = interpolate.interp1d(y['c_err'], plot_centers, fill_value = 0, bounds_error=False)(0.04)
        redshift_com_lower = interpolate.interp1d(y['c_err']-err_y['c_err']\
                                                           , plot_centers, fill_value = 0, bounds_error=False)(0.04)
        redshift_com_higher = interpolate.interp1d(y['c_err']+err_y['c_err'],\
                                                            plot_centers, fill_value = 0, bounds_error=False)(0.04)
        
        data_r = r[r['config'] == i]
        
        conf_text = 'Configuration {} \n  '.format(i)
        
        for b in 'gri':
            conf_text += 'cad_{}={} '.format(b, data_r['cad_{}'.format(b)].values[0])

            conf_text += 'N_{}={} '.format(b, data_r['N_{}'.format(b)].values[0])
            
        fig, ax= plt.subplots()
        
        
        ax.errorbar(plot_centers, y['c_err'], err_y['c_err'],\
                     marker = 'o',color='k', linestyle='-', label = 'binned data points')
 
        ax.plot(plot_centers, y['c_err'] + err_y['c_err'], marker = '', color='red')
        ax.plot(plot_centers, y['c_err'] - err_y['c_err'], marker = '', color='red')
        
        ax.set_xlabel("z")
        ax.set_ylabel(r"$\sigma(c)$")
        ax.set_ylim(0-np.max(err_y['c_err']), np.max(y['c_err']) + 2* np.max(err_y['c_err']))
        
        
        ax.set_xlim(np.min(plot_centers -0.01), np.max(plot_centers +0.01))
        #Line for the zlim
        ax.hlines(0.04, 0.01, redshift_com_lower,  ls="--", color = 'red' )

        ax.vlines(redshift_com, 0-np.max(err_y['c_err']), 0.04, ls="-.", color = 'blue')
        #Line zlim+1sigma
        ax.vlines(redshift_com_lower, 0-np.max(err_y['c_err']), 0.04,  ls="--", color = 'green' )
        #Line for zlim-1sigma
        ax.vlines(redshift_com_higher, 0-np.max(err_y['c_err']), 0.04,  ls="--", color = 'green' )
        
        ax.legend(title = 'Redshift completness : {}'.format(np.round(redshift_com, 3)))
        ax.set_title(conf_text, fontsize='medium')
        if save=='save':
            
            save_plot(path_plot, 'z_c_{}'.format(i))
        else:
            plt.show()
        plt.close()

     
def mu_exp(data, alpha=0.16, beta=3, M =-19.5):
    '''
    This function takes as input a Table containing informations on supernovae
    and gives back the calculated value of distance modulus and its uncertainty
    according to the formula : mu = mb - M + alpha * x1 - beta * c
    with mb = -2.5 log10(x0) + 10.63

    Parameters
    ----------
    data : panda dataframe or Astropy table
        The dataframe containing the informations of the SNe Ia. 
    alpha : float, optional
         alpha nuisance parameter (from the SALT2 model).  The default is 0.16.
    beta :float, optional
        beta nuisance parameter (from the SALT2 model).  The default is 3.
    M : float, optional
        absolute magnitude, it is a nuisance parameter. The default is -19.5.

    Returns
    -------
    mu : panda serie containing floats
         panda serie with the distance modulus
    sig_mu :  panda serie containing floats
         panda serie with the distance modulus uncertainties

    '''
    data['mb'] = -2.5 *np.log10(data['x0_fit'])+ 10.63
    5
    #Here, we calculate the uncertainty of mb
    data['mb_err'] = (2.5/(np.log(10)*data['x0_fit']))*data['x0_err']
    
    
    mu = data['mb'] + alpha * data['x1_fit'] - beta * data['c_fit'] - M
    #First, we calculate the uncertainty on the observation linked to the
    #SALT2 model
    #we add the diagonals
    var_mu = data['mb_err']**2 + alpha**2 * data['x1_err'] + beta**2 * data['c_err']**2
    
    #We calculate the covariance between mb and x1 and c 
    
    data['mb_c_cov'] = -2.5/(data['x0_fit'] * np.log(10)) * data['x0_c_cov']
    data['mb_x1_cov'] = -2.5/(data['x0_fit'] * np.log(10)) * data['x0_x1_cov']
    
    #We add the covariance
    var_mu += -2*beta*alpha* data['x1_c_cov']#then x1, mb
    var_mu += -2*beta * data['mb_c_cov']#mb, c
    var_mu += +2*alpha* data['mb_x1_cov']#mb, x1
    
    #There is also an intrinsic uncertainty of 0.15. To simulate that, 
    #the mu value will be smeared by a normal law centered on the mu 
    #and with a deviation of 0.15.
    sig_int = 0.15
    mu = np.random.normal(mu, sig_int)
    #We must add this deviation to the sig_mu
    var_mu = var_mu + sig_int**2
    sig_mu = np.sqrt(var_mu)
    return mu, sig_mu

def plot_distribution_z(save=False):
    '''
    Plot the distribution of the SNe Ia of a given file. 
    They are displayed with no selection, then with only the one where 
    the light curve has been fitted (and so intrinsic parameters have been
                                     extracted)
    and finally the ones with a sigma_c < 0.04

    '''
    print('------------------------------- \n Running : plot_distribution_z \n\
------------------------------- \n')
    for i in confs:
        name = '../dataSN/{}/{}'.format(config_file,i)
        df = browse_directory(name)
        #First, we have the data without any selection
        data = df.to_pandas()
        #Then, we select only the supernovae with a fit on their intrinsic 
        #parameters
        data_fitok = data[data['fitstatus']==b'fitok']
        #Then, we select only the supernovae with a error on the color parameter c inferior 
        #to 0.04 corresponding to the z lim
        data_sel = data_fitok[data_fitok['c_err'] < 0.04]
        
        ratio_data, err_data = ratio_fit(data)
        
        bins = np.arange(np.min(data['z']), np.max(data['z']), 0.01)
        
        
        
        fig = plt.figure(figsize=(12, 6))
        gs = plt.GridSpec(4,2, hspace=0.3, wspace=0) #we define a grid 
        ax_main = fig.add_subplot(gs[0:3,0:2]) #ax_joint will be our main plot
        ax_ratio = fig.add_subplot(gs[3:4,0:2]) #ax_joint_x will be our residual
        
        
        ax_main.hist(data['z'],bins, histtype='step' ,  label = 'No selection',\
                  linewidth=2, alpha = 0.8)
            
        ax_main.hist(data_fitok['z'], bins,histtype='step',\
                label ='Fitted SNe Ia', linewidth = 2, color='b', alpha = 0.8)
            
        ax_main.hist(data_sel['z'], bins,histtype='step',\
                label = r'$\sigma_{c} < 0.04$', linewidth = 2, \
                    color='g', alpha = 0.8)
            
        
        ax_main.set_ylabel(r"$N_{SNIa}$")
        ax_main.legend(loc='best')
        
        ax_ratio.errorbar(bins[1:], ratio_data, yerr = err_data , color='b', marker = 'o')
        ax_ratio.set_xlabel('z')
        ax_ratio.set_ylabel('Ratio fit/all (%)')
        
        if save=='save':   
            path_plot = r'../Plots/{}'.format(config_file)
            save_plot(path_plot, 'distrib_z_{}'.format(i))
        else:
            plt.show()
        plt.close()
        
def mu_plot_z(save=False):
    '''
    Plot the distance modulus and its uncertainty relative to the redshift and shows the standard deviation 
    of the sample. 

    Parameters
    ----------
    save : str, optional
        Indicates if the plot must be saved. The default is False.

    Returns
    -------
    None.

    '''
    print('------------------------------- \n Running : mu_plot_z  \n\
------------------------------- \n')
    for i in confs:
        name = '../dataSN/{}/{}'.format(config_file, i)
        df = browse_directory(name)
        data = selec(df)
        
        data['mu_fit'], data['mu_err'] = mu_exp(data)
        
        data = data[data['mu_fit']<100]
        data = data[data['mu_fit']>0]

        bins = np.arange(np.min(data['z']), np.max(data['z']), 0.01)
        
        z_mean = data.groupby(pd.cut(data['z'], bins)).mean()
        z_std = data.groupby(pd.cut(data['z'], bins)).std()
        
        fig, (ax_mu, ax_sig) = plt.subplots(1, 2, figsize=(13, 8))
        
        ax_mu.errorbar(z_mean['z'], z_mean['mu_fit'], yerr=z_std['mu_fit'], xerr=z_std['z'], linestyle='-', marker = '.')
        ax_mu.set_xlabel('z')
        ax_mu.set_ylabel(r'$\mu$')
        
        ax_sig.errorbar(z_mean['z'], z_mean['mu_err'], yerr=z_std['mu_err'], xerr=z_std['z'], linestyle='-', marker = '.')
        ax_sig.set_xlabel('z')
        ax_sig.set_ylabel(r'$\sigma_{\mu}$')
        
        if save=='save':
            path_plot = r'../Plots/{}'.format(config_file)
            save_plot(path_plot, 'mu_z_{}'.format(i))
        else:
            plt.show()
        plt.close()
        
    

if plot_config_compare_state =='True' or plot_config_compare_state =='save':
    plot_compare(confs, config_file, plot_config_compare_state)

if plot_sig_c_state =='True' or plot_sig_c_state=='save':
    plot_z_c(confs, config_file, plot_sig_c_state)
if plot_distribution_z_state =='True' or plot_distribution_z_state=='save':
    plot_distribution_z(plot_distribution_z_state)
    
if plot_mu_z_state == 'True' or plot_mu_z_state=='save':
    mu_plot_z(plot_mu_z_state)

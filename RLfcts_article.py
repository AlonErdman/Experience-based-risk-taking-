import sqlite3
import pandas as pd
import numpy as np
from numpy import matlib as mb
import scipy.stats
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objects import Layout
from plotly.subplots import make_subplots
import sys, os
import time
from sklearn.linear_model import LinearRegression
#import seaborn as sns
import re # regex
import h5py
import itertools
import statsmodels.api as sm
from scipy.special import expit, logit
import multiprocessing as mp
import pickle as pk
import sklearn.decomposition as dc
from llik_building_blocks import *
from llik_adv import *
from llik_simple import *
import plotly.express as px



## Some helper functions
def logsumexp_dim0(A):
    """
    Computes the logsumexp of the numpy array 'A' along axis 0, while avoiding numerical underflow.
    
    Arguments:
     - A, (d,N) numpy array
     
    Returns:
    - (N) numpy array, the logsumexp
    """
    
    Amax = A.max(axis = 0) # (N)
    return np.log(np.sum(np.exp(A - Amax[None,:]), axis = 0)) + Amax

def logsumexp_1D(A):
    """
    Computes the logsumexp of the numpy array 'A', while avoiding numerical underflow.
    
    Arguments:
     - A, (d) numpy array
     
    Returns:
    - float, the logsumexp
    """
    Amax = A.max() # float
    return np.log(np.sum(np.exp(A - Amax))) + Amax

def multinomial_vect(p):
    """
    Generates multinomial random variables, with 1 draw per column of p. Returns the index of the p-entry that won in the end. 
    It's vectorized along the N-axis.
    
    Arguments:
    - p, (d, N) numpy array: the probabilities of the different outcomes.
    
    Returns:
    - (N) numpy array, with the index of the p-entry chosen per column of p.
    """
    N = p.shape[1]
    rvs = np.random.uniform(size = (N)) 
    p_cum = p.cumsum(axis = 0)
    idx = np.zeros((N))
    for i in range(p.shape[0]):
        idx += (rvs > p_cum[i,:])
    return idx.astype('int64')
   
def BIC_comparison_summary(BIC, plot_results = True, model_names = None):
    """
    Given BIC scores for various models fit on data, computes:
    1. The frequency the first model came out as best, and its stddev.
    2. The average (BIC_other_model - BIC_first_model) and its stddev.
    4. The average BIC per model and its stddev.
    
    Arguments:
    - BIC: (num_models, num_experiments) numpy array with in position (m, l) the BIC of model m in experiment l.
    """
    
    L = BIC.shape[1]
    M = BIC.shape[0]
    
    if model_names is None:
        model_names = ["Reference"] + [str(int(m)) for m in np.arange(1,M)]
    
    # Compute the frequency of the first model winning:
    freq = np.mean(np.argmin(BIC, axis = 0) == 0)
    std_freq = np.sqrt(freq * (1. - freq) / L)
    
    # Compute the mean and std of the BIC differences with the other models:
    diff = BIC[1:,:] - BIC[0,:].reshape((1,L)) 
    avg_diff = np.mean(diff, axis = 1)
    std_diff = np.std(diff, axis = 1)
    
    if plot_results:
        df_freq = pd.DataFrame({'freq': [freq], 'error': [1.96 * std_freq], 'name': model_names[0]})
        # Frequency plot:
        fig = px.bar(df_freq, x = 'name', y = 'freq', error_y = 'error', title = "Recovery Frequency")
        fig.update_layout(
            xaxis={
                'title':""
            },
            yaxis={
                'title':'Frequency',
                'range': [0., 1.]  
            })
        fig.show()
        
        # Difference plot:
        fig = px.bar(x = model_names[1:], y = avg_diff, error_y = 1.96 * std_diff, title = "BIC Difference")
        fig.update_layout(
            xaxis={
                'title':'Model'
            },
            yaxis={
                'title':'BIC[model] - BIC[Reference]',
            })
        fig.show()
        
        # Absolute BIC plot:
        fig = px.bar(x = model_names, y = np.mean(BIC, axis = 1), error_y = 1.96 * np.std(BIC, axis = 1), title = "Average BIC")
        fig.update_layout(
            xaxis={
                'title':'Model'
            },
            yaxis={
                'title':'BIC[model]',
            })
        fig.show()  
    
    return (freq, std_freq, avg_diff, std_diff, np.mean(BIC, axis = 1), np.std(BIC, axis = 1))
    
# def generate_PM_datasets(m0, s_dat, sim_data_folder, num_datasets = 25):
    # """
    # Create generated datasets and save them, according to the Posterior Mean and styles in 
    # the model m0. Saves them in sim_data_folder.    
    # """
   
    # if not os.path.exists(sim_data_folder):
        # os.mkdir(sim_data_folder)
        
    # P, P_aux, _, _ = m0.extract_param_values()
    # style = m0.get_style()
    
    # s_dat = subjects_EMA().copy_subject_states(s_dat)

    # for k in range(num_datasets):
        # filename = os.path.join(sim_data_folder, f'Simulated-PM-{k}.h5')
        # if not os.path.exists(filename):
            # print(f'\nGenerating dataset {k:2d}:\n----------------------')
            # s_dat.simulate_data(style, P, P_aux = P_aux)
            # with h5py.File(filename, 'w') as hf:
                # hf.create_dataset('o', data = s_dat.o)
                # hf.create_dataset('C', data = s_dat.C)
                # hf.create_dataset('C_st', data = s_dat.C_st)
                # hf.create_dataset('acc', data = s_dat.acc)
                # gp = hf.create_group('P')
                # for par_name in P.columns:
                    # gp.create_dataset(par_name, data = P[par_name].values)
                # if not P_aux is None:
                    # gp = hf.create_group('P_aux')
                    # for par_name in P_aux.columns:
                        # gp.create_dataset(par_name, data = P_aux[par_name].values) 
                        
def generate_PM_datasets(m0, s_dat, sim_data_folder, num_datasets = 20, copy_function = None):
    """
    Create generated datasets and save them, according to the Posterior Mean and styles in 
    the model m0. Saves them in sim_data_folder.    
    """
   
    if not os.path.exists(sim_data_folder):
        os.mkdir(sim_data_folder)
        
    P, P_aux, _, _ = m0.extract_param_values()
    
    if copy_function is None:
        copy_s_dat = False
        if subjects_EMA in locals():
            if type(s_dat) == subjects_EMA:
                copy_s_dat = True
        if copy_s_dat:
            s_dat = subjects_EMA().copy_subject_states(s_dat)
    else:
        s_dat = copy_function(s_dat)

    for k in range(num_datasets):
        filename = os.path.join(sim_data_folder, f'Simulated-PM-{k}.h5')
        if not os.path.exists(filename):
            print(f'\nGenerating dataset {k:2d}:\n----------------------')
            tm = time.time()
            for n in range(s_dat.N):
                
                P_n = P.iloc[[n],:]
                if P_aux is None:
                    P_aux_n = None
                else:
                    P_aux_n = P_aux.iloc[[n],:]
                    
                if m0.llik_style == 'advanced_template':
                    llik_adv(P_n, P_aux_n, n, s_dat, m0.style, simulate_data = True)
                elif m0.llik_style == 'simple_template':
                    llik_simple(P_n, P_aux_n, n, s_dat, m0.style, simulate_data = True)
                elif m0.llik_style == 'custom':
                    m0.llik(P_n, P_aux_n, n, s_dat, simulate_data = True)
                else:
                    sys.exit("generate_PM_datasets > Not a valid llik_style.")
                    
            s_dat.P_true = P
            s_dat.P_aux_true = P_aux
                      
            s_dat.save_choices(filename)
            print(f"Time to simulate {s_dat.N} subjects over {s_dat.T} timesteps:\n==> {time.time() - tm:.3f}s.")  

def generate_PM_datasets_two_models(m0, m1 , s_dat, sim_data_folder, num_datasets = 20, copy_function = None):
    """
    Create generated datasets and save them, according to the Posterior Mean and styles in 
    the model m0. Saves them in sim_data_folder.    
    """
   
    if not os.path.exists(sim_data_folder):
        os.mkdir(sim_data_folder)
        
    P, P_aux, _, _ = m0.extract_param_values()
    P_sec, P_aux_sec, _, _ = m1.extract_param_values()
    
    if copy_function is None:
        copy_s_dat = False
        if subjects_EMA in locals():
            if type(s_dat) == subjects_EMA:
                copy_s_dat = True
        if copy_s_dat:
            s_dat = subjects_EMA().copy_subject_states(s_dat)
    else:
        s_dat = copy_function(s_dat)

    for k in range(num_datasets):
        filename = os.path.join(sim_data_folder, f'Simulated-PM-{k}.h5')
        if not os.path.exists(filename):
            print(f'\nGenerating dataset {k:2d}:\n----------------------')
            tm = time.time()
            for n in range(s_dat.N):
                
                P_n = P.iloc[[n],:]
                P_n_sec = P_sec.iloc[[n],:]
                if P_aux is None:
                    P_aux_n = None
                else:
                    P_aux_n = P_aux.iloc[[n],:]
                    P_aux_n_sec = P_aux_sec.iloc[[n],:]

                    
                if m0.llik_style == 'advanced_template':
                    llik_adv(P_n, P_aux_n, n, s_dat, m0.style, simulate_data = True)
                elif m0.llik_style == 'simple_template':
                    llik_simple(P_n, P_aux_n, n, s_dat, m0.style, simulate_data = True)
                elif m0.llik_style == 'custom':
                    m0.llik(P_n, P_aux_n, P_n_sec, P_aux_n_sec ,n, s_dat, simulate_data = True)
                else:
                    sys.exit("generate_PM_datasets > Not a valid llik_style.")
                    
            s_dat.P_true = P
            s_dat.P_aux_true = P_aux
                      
            s_dat.save_choices(filename)
            print(f"Time to simulate {s_dat.N} subjects over {s_dat.T} timesteps:\n==> {time.time() - tm:.3f}s.")             

def generate_smoothed_PM_datasets_specific(m0, s_dat, sim_data_folder, num_datasets = 20, copy_function = None):
    """
    Create generated datasets and save them, according to the Posterior Mean and styles in 
    the model m0. Saves them in sim_data_folder.    
    """
   
    if not os.path.exists(sim_data_folder):
        os.mkdir(sim_data_folder)
        
    P, P_aux, _, _ = m0.extract_param_values()
    
    for param_name in ['logR_r', 'logR_pun']:
        for d in range(27): # TODO: change adaptively to other number of days
            P_aux[f'{param_name}_{d}'] = np.mean(P_aux[[f'{param_name}_{d}', f'{param_name}_{d+1}']] , axis = 1)
            
    for d in range(28):
        P_aux[f'R_r_{d}'] = np.exp(P_aux[f'logR_r_{d}'])           
        P_aux[f'R_pun_{d}'] = -np.exp(P_aux[f'logR_pun_{d}'])
    
        P_aux[f'R_mean_{d}'] = np.sqrt(P_aux[f'R_r_{d}'] * P_aux[f'R_pun_{d}'].abs())
        P_aux[f'R_asym_{d}'] = P_aux[f'R_r_{d}'] / P_aux[f'R_pun_{d}'].abs()    
        
        for param_name in ['R_mean','R_asym']:
            P_aux[f'log{param_name}_{d}'] = np.log(P_aux[f'{param_name}_{d}'])
    
    if copy_function is None:
        copy_s_dat = False
        if subjects_EMA in locals():
            if type(s_dat) == subjects_EMA:
                copy_s_dat = True
        if copy_s_dat:
            s_dat = subjects_EMA().copy_subject_states(s_dat)
    else:
        s_dat = copy_function(s_dat)

    for k in range(num_datasets):
        filename = os.path.join(sim_data_folder, f'Simulated-PM-{k}.h5')
        if not os.path.exists(filename):
            print(f'\nGenerating dataset {k:2d}:\n----------------------')
            tm = time.time()
            for n in range(s_dat.N):
                
                P_n = P.iloc[[n],:]
                if P_aux is None:
                    P_aux_n = None
                else:
                    P_aux_n = P_aux.iloc[[n],:]
                    
                if m0.llik_style == 'advanced_template':
                    llik_adv(P_n, P_aux_n, n, s_dat, m0.style, simulate_data = True)
                elif m0.llik_style == 'simple_template':
                    llik_simple(P_n, P_aux_n, n, s_dat, m0.style, simulate_data = True)
                elif m0.llik_style == 'custom':
                    m0.llik(P_n, P_aux_n, n, s_dat, simulate_data = True)
                else:
                    sys.exit("generate_PM_datasets > Not a valid llik_style.")
                    
            s_dat.P_true = P
            s_dat.P_aux_true = P_aux
                      
            s_dat.save_choices(filename)
            print(f"Time to simulate {s_dat.N} subjects over {s_dat.T} timesteps:\n==> {time.time() - tm:.3f}s.")                     
            

     
def load_PM_dataset(data_folder, s_dat, dataset_idx = 0, base_name = 'Simulated-PM', P_true = None, P_aux_true = None, copy_function = None):
    k = dataset_idx
    
    filename = os.path.join(data_folder, f'{base_name}-{k}.h5')
    
    if copy_function is None:
        copy_s_dat = False
        if subjects_EMA in locals():
            if type(s_dat) == subjects_EMA:
                copy_s_dat = True
        if copy_s_dat:
            s_dat = subjects_EMA().copy_subject_states(s_dat)
    else:
        s_dat = copy_function(s_dat)
    
    s_dat = s_dat.load_choices(filename)
    
    return s_dat
    
def check_and_load_repdat(s_dat_sim_l, dataset_idx = 0, num_if_new = 2, model_folder = '', spec = None, style_adv = None, style_simple = None, llik_custom = None):

    # save folder and model basename:
    save_folder = os.path.join(model_folder, f'dataset-{dataset_idx}')
    model_base_name = f'model'
    
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # Loading, or creation:
    all_models = []
    k = 0
    for l in range(num_if_new):
        filename = f'model-{l}.h5'
        if filename in os.listdir(save_folder):
            m = RL_model(empty = True, verbose = False)
            m.load(os.path.join(save_folder, filename), s_dat_sim_l, llik_custom = llik_custom, verbose = False)
            all_models.append(m)
            k += 1
        else:
            m = RL_model(subjects_data = s_dat_sim_l, name = f'{model_base_name}-{l}', spec = spec, style_adv = style_adv, style_simple = style_simple, llik_custom = llik_custom)
            all_models.append(m)
    print(f"Found {k:2d} existing models. Created {num_if_new - k} more.")
         
    return all_models, save_folder
    
def check_and_load_one_model(m_name, s_dat, spec, save_folder, style_adv = None, style_simple = None, llik_custom = None):
    
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
        
    if f'{m_name}.h5' in os.listdir(save_folder):
       print(f"- Loaded {m_name}.")
       
       m = RL_model(empty = True, verbose = False)
       m.load(os.path.join(save_folder, f"{m_name}.h5"), s_dat, llik_custom = llik_custom, verbose = False)
       
    else:
       print(f"- Created {m_name}.")
       m = RL_model(subjects_data = s_dat, name = m_name, spec = spec, style_adv = style_adv, style_simple = style_simple, llik_custom = llik_custom) # Model
    
    return m
            
    
def check_and_load_extended(model_base_name, s_dat, num_if_new = 5, model_folder = 'models_greatcomparison_sim', style_adv = None, style_simple = None, llik_custom = None, spec = None, verbose = False):

    # save folder and model basename:
    save_folder = f'{model_folder}'
    
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # Loading, or creation:
    all_models = []
    k = 0
    for l in range(num_if_new):
        filename = f'{model_base_name}-{l}.h5'
        if filename in os.listdir(save_folder):
            if verbose:
                print(f'{filename} - Found')
            m = RL_model(empty = True, verbose = False)
            m.load(os.path.join(save_folder, filename), s_dat, llik_custom = llik_custom, verbose = False)
            all_models.append(m)
            k += 1
        else:
            if verbose:
                print(f'{filename} - Creating new.')
            m = RL_model(subjects_data = s_dat, name = f'{model_base_name}-{l}', spec = spec, style_adv = style_adv, style_simple = style_simple, llik_custom = llik_custom) # Model
            all_models.append(m)
    print(f"Found {k:2d} existing models. Created {num_if_new - k} more.")
         
    return all_models, save_folder, model_base_name

def load_models(folder, inference_name, s_dat, llik_custom = None):
    
    save_folder = f'{folder}\\{inference_name}'
    all_models = []
    
    k = 0
    for filename in os.listdir(save_folder):        
        if filename[-3:] == '.h5':
            print(f"{k:2d})")
            m = RL_model(empty = True)
            m.load(os.path.join(save_folder, s_dat, filename), llik_custom = llik_custom)
            all_models.append(m)
            k += 1
                
    return all_models, save_folder

def load_and_duplicate_model(inference_name, filepath, s_dat, num = 5, AIS_folder = 'models_greatcomparison_sim', llik_custom = None):
    
    all_models = []
    for l in range(num):
        m = RL_model(empty = True)
        m.load(filepath, s_dat, llik_custom = llik_custom)
        model_base_name = m.name + f'-{inference_name}'
        m.name = m.name + f'-{inference_name}-{l}'
        all_models.append(m)
         
    # save folder and model basename:
    save_folder = f'{AIS_folder}\\{inference_name}'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    
    return all_models, save_folder, model_base_name
    
def model_comparison_summary(model_folders, s_dat_m, llik_custom = None, llik_type = None):
    models = []; k = 0
    spec = None
    
    if type(model_folders) != list:
        model_folders = [model_folders]
        
    if type(llik_custom) != list:
        llik_custom = [llik_custom for folder in model_folders]
        
    if type(llik_type) != list:
        llik_type = [llik_type for folder in model_folders]
        
    if type(s_dat_m) != list:
        s_dat_m = [s_dat_m for folder in model_folders]
        
    if len(llik_custom) != len(model_folders):
        sys.exit("`llik_custom` should be None, one function, or a list the length of the model_folder list.")
        
    if len(llik_type) != len(model_folders):
        sys.exit("`llik_type` should be None, one type, or a list the length of the model_folder list.")
        
    if len(s_dat_m) != len(model_folders):
        sys.exit("`subjects data` should be one Subjects() object, or a list the length of the model_folder list.")
        
    all_llik_types = []
    
    for l, model_folder in enumerate(model_folders):
        for filename in os.listdir(model_folder):
            if filename[-3:] == '.h5':
                print(f"{k:2d})")
                models.append(check_and_load_one_model(filename[:-3], s_dat_m[l], spec, model_folder, llik_custom = llik_custom[l]))
                all_llik_types.append(llik_type[l])
                k += 1

    df_results = []

    for k, m in enumerate(models):
        delta_ev = m.ev_jump 
        
        if m.llik_style in ['advanced_template', 'simple_template'] and not m.q_fitted_EM_it[-1]:
            ev       = m.evidence[-1]
            ev_trial = np.exp(m.evidence[-1] / np.sum(m.num_datapoints))
            ci_ev   = 1.96 * np.std([m.fit[n]['evidence'][-1] for n in range(m.N)]) / np.sqrt(m.N)
        
            ev_f       = m.evidence_f[-1]
            ev_f_trial = np.exp(m.evidence_f[-1] / np.sum(m.num_nopractice_f))
            ci_ev_f    = 1.96 * np.std([m.fit[n]['evidence_f'][-1] for n in range(m.N)]) / np.sqrt(m.N)
            
            ev_nf       = m.evidence_nf[-1]
            ev_nf_trial = np.exp(m.evidence_nf[-1] / np.sum(m.num_nopractice_nf))
            ci_ev_nf    = 1.96 * np.std([m.fit[n]['evidence_nf'][-1] for n in range(m.N)]) / np.sqrt(m.N)
            
            df_results.append(pd.DataFrame({'iBIC': m.bic[-1] , 'delta_iBIC': 2. * delta_ev, 'ESS_mean (last)': m.ESS_mean[-1], 'ev': ev, 'ci_ev': ci_ev, 'avg prob': ev_trial, 'ev_f': ev_f, 'ci_ev_f': ci_ev_f, 'avg f prob': ev_f_trial, 'ev_nf': ev_nf, 'ci_ev_nf': ci_ev_nf, 'avg nf prob': ev_nf_trial, 'index' : k}, index = [m.name]))
        else:
            ev       = m.evidence[-1]
            if all_llik_types[k] is None or all_llik_types[k] == 'All':
                ev_trial = np.exp(m.evidence[-1] / np.sum(m.num_datapoints))
            elif all_llik_types[k] == 'NF':
                ev_trial = np.exp(m.evidence[-1] / np.sum(m.num_nopractice_nf))
            elif all_llik_types[k] == 'F':
                ev_trial = np.exp(m.evidence[-1] / np.sum(m.num_nopractice_f))
            ci_ev   = 1.96 * np.std([m.fit[n]['evidence'][-1] for n in range(m.N)]) / np.sqrt(m.N)

            df_results.append(pd.DataFrame({'iBIC': m.bic[-1] , 'delta_iBIC': 2. * delta_ev, 'ESS_mean (last)': m.ESS_mean[-1], 'ev': ev, 'ci_ev': ci_ev, 'avg prob': ev_trial, 'index' : k}, index = [m.name]))

    display(pd.concat(df_results).sort_values('iBIC',ascending=False).round(2))
    
    return models    
    

### Samplers for base parameters: ###
#
# The following function instantiates samplers for the distributions as specified by the spec of the RL_models

def sampleP(spec, S):
    """
    Samples parameter values according to the parameter specifications in 'spec'.

    Arguments:
    - S, int: the number of parameters to be sampled.

    Creates a new (S, .) dataframe 'self.P', with S parameter values.
    """
    P = {}
    for param_name in spec.keys():
        param_type = spec[param_name]['type'] # Distribution of the hyperparameter
        param_val = spec[param_name]['val']   # Values of that distribution
        
        # 1) Normal distribution:
        if param_type == 'normal':
            P[param_name] = np.random.normal(loc = param_val[0], scale = param_val[1], size = (S))
        # 2) Beta distribution:
        elif param_type == 'beta':
            P[param_name] = np.random.beta(param_val[0], param_val[1], size = (S))
        # 3) Gamma distribution:
        elif param_type == 'gamma':
            P[param_name] = np.random.gamma(param_val[0], param_val[1], size = (S))
        # 4) Minus the Gamma dsitribution:
        elif param_type == 'min_gamma':
            P[param_name] = -np.random.gamma(param_val[0], param_val[1], size = (S))
        # 5) Binomial distribution:
        elif param_type == 'binom':
            P[param_name] = np.random.binomial(1, param_val[0], size = (S))
        # 6) Lognormal distribution:
        elif param_type == 'lognormal':
            P[param_name] = np.exp(np.random.normal(loc = param_val[0], scale = param_val[1], size = (S)))
        # 7) Minus the lognormal distribution:
        elif param_type == 'min_lognormal':
            P[param_name] = -np.exp(np.random.normal(loc = param_val[0], scale = param_val[1], size = (S)))
        elif param_type == 'logitnormal':
            P[param_name] = expit(np.random.normal(loc = param_val[0], scale = param_val[1], size = (S)))
        # 8) A fixed constant that does not get updated ('fixed_val' gets used as opposed to 'val'):
        elif param_type == 'fixed':
            P[param_name] = np.full((S), spec[param_name]['fixed_val'])
        # 9) Student t distribution
        elif param_type == 't':
            P[param_name] = param_val[0] + param_val[1] * np.random.standard_t(param_val[2], size = (S))
        # 10) log Student t distribution
        elif param_type == 'logt':
            P[param_name] = np.exp(param_val[0] + param_val[1] * np.random.standard_t(param_val[2], size = (S)))
        # 11) Minus the log Student t distribution
        elif param_type == 'min_logt':
            P[param_name] = - np.exp(param_val[0] + param_val[1] * np.random.standard_t(param_val[2], size = (S)))
        # 12) logit Student t distribution
        elif param_type == 'logitt':
            P[param_name] = expit(param_val[0] + param_val[1] * np.random.standard_t(param_val[2], size = (S)))
        elif param_type == 'logBB_stat' or param_type == 'min_logBB_stat':
            P[f'{param_name}_0'] = np.exp(np.random.normal(loc = param_val[0], scale = param_val[1], size = (S))) 
            if param_type == 'min_logBB_stat':
                P[f'{param_name}_0'] = - P[f'{param_name}_0']
            P[f'{param_name}_sigma'] = np.exp(np.random.normal(loc = param_val[2], scale = param_val[3], size = (S)))
        elif param_type == 'BB_stat':
            P[f'{param_name}_0'] = np.random.normal(loc = param_val[0], scale = param_val[1], size = (S))
            P[f'{param_name}_sigma'] = np.exp(np.random.normal(loc = param_val[2], scale = param_val[3], size = (S)))
        elif param_type == 'logitBB_stat':
            P[f'{param_name}_0'] = expit(np.random.normal(loc = param_val[0], scale = param_val[1], size = (S)))
            P[f'{param_name}_sigma'] = np.exp(np.random.normal(loc = param_val[2], scale = param_val[3], size = (S)))
        # elif param_type in ['OU2', 'logOU2', 'min_logOU2', 'logitOU2']:
            # ## TODO still: integrated OU
            # P[f'tf_{param_name}_0']  = np.random.normal(loc = param_val[0], scale = param_val[1], size = (S)) 
            # P[f'{param_name}_mu']    = np.random.normal(loc = param_val[2], scale = param_val[3], size = (S))               
            # P[f'{param_name}_rate']  = np.exp(np.random.normal(loc = param_val[4], scale = param_val[5], size = (S)))
            # P[f'{param_name}_sigma'] = np.exp(np.random.normal(loc = param_val[6], scale = param_val[7], size = (S)))
        elif param_type in ['OU', 'logOU', 'min_logOU', 'logitOU']:
            P[f'{param_name}_mu']       = np.random.normal(loc = param_val[0], scale = param_val[1], size = (S))             
            P[f'{param_name}_rate']     = np.exp(np.random.normal(loc = param_val[2], scale = param_val[3], size = (S)))
            P[f'{param_name}_stat_std'] = np.exp(np.random.normal(loc = param_val[4], scale = param_val[5], size = (S)))  
        elif param_type in ['OU_fixed_mean', 'logOU_fixed_mean', 'min_logOU_fixed_mean', 'logitOU_fixed_mean']:
            P[f'{param_name}_mu']       = np.full((S), spec[param_name]['fixed_val'])             
            P[f'{param_name}_rate']     = np.exp(np.random.normal(loc = param_val[0], scale = param_val[1], size = (S)))
            P[f'{param_name}_stat_std'] = np.exp(np.random.normal(loc = param_val[2], scale = param_val[3], size = (S)))
        else:
            sys.exit(f"{param_name} does not have a valid 'spec['{param_name}']['type']' ('{param_type}' given)."+ \
         "\nTry one of the following:\n- 'normal'/'lognormal'/'min_lognormal' with val (loc, scale)\n- 'beta' with val (a, b)" + \
         "\n- 'gamma'/'min_gamma' with val (loc, scale)\n- 'binom' with val (p)\n- 't'/'logt'/'min_logt'/'logitt' with val (loc, scale, df)" + \
         "\n- 'fixed' with fixed_val (value)")

    return pd.DataFrame(P)

### Compute the log-pdf of the parameters in P, given their marginal distributions       ###
#   This assumes the parameters are independent. Use the other lpdf functions otherwise. ###
def compute_lpdf(spec, P):
    """
    Computes the log-pdf values of the samples in P, given the distribution specifications of 'spec'.
    Assumes independence across dimensions.
    """
    
    lpdf = np.zeros((P.shape[0]))
    
    for param_name in spec.keys():
        param_type = spec[param_name]['type'] # Distribution of the hyperparameter
        param_val = spec[param_name]['val']   # Values of that distribution
                
        if param_type == 'normal':
            lpdf += scipy.stats.norm.logpdf(P[param_name].values, loc = param_val[0], scale = param_val[1])
        elif param_type == 'beta':
            lpdf += scipy.stats.beta.logpdf(P[param_name].values, param_val[0], param_val[1])
        elif param_type == 'gamma':
            lpdf += scipy.stats.gamma.logpdf(P[param_name].values, param_val[0], scale = param_val[1])
        elif param_type == 'min_gamma':
            lpdf += scipy.stats.gamma.logpdf(-P[param_name].values, param_val[0], scale = param_val[1])
        elif param_type == 'binom':
            lpdf += scipy.stats.binom.logpdf(P[param_name].values, 1, param_val[0])
        elif param_type == 'lognormal':
            # pos_idx = P[param_name].values > 1e-5 
            lpdf += scipy.stats.norm.logpdf(np.log(P[param_name].values), loc = param_val[0], scale = param_val[1]) - np.log(P[param_name].values)
        elif param_type == 'min_lognormal':
            #neg_idx = P[param_name].values < -1e-5 
            lpdf += scipy.stats.norm.logpdf(np.log(-P[param_name].values), loc = param_val[0], scale = param_val[1]) - np.log(-P[param_name].values)
            #lpdf[~neg_idx] = np.nan
        elif param_type == 'logitnormal':
            lpdf += scipy.stats.norm.logpdf(logit(P[param_name].values), loc = param_val[0], scale = param_val[1]) - np.log(P[param_name].values) - np.log(1. - P[param_name].values)
        elif param_type == 'fixed':
            lpdf += 0.
        elif param_type == 't':
            lpdf += scipy.stats.t.logpdf(P[param_name].values, param_val[2], loc = param_val[0], scale = param_val[1])
        elif param_type == 'logt':
            lpdf += scipy.stats.t.logpdf(np.log(P[param_name].values), param_val[2], loc = param_val[0], scale = param_val[1]) - np.log(P[param_name].values)
        elif param_type == 'logitt':
            x = P[param_name].values
            t = np.log(x) - np.log(1. - x)
            lpdf += scipy.stats.t.logpdf(t, param_val[2], loc = param_val[0], scale = param_val[1]) - np.log(x) - np.log(1. - x)
        elif param_type == 'min_logt':
            lpdf += scipy.stats.t.logpdf(np.log(-P[param_name].values), param_val[2], loc = param_val[0], scale = param_val[1]) - np.log(-P[param_name].values)
        else:
            sys.exit(f"{param_name} does not have a valid 'spec['{param_name}']['type']' ('{param_type}' given)."+ \
         "\nTry one of the following:\n- 'normal'/'lognormal'/'min_lognormal' with val (loc, scale)\n- 'beta' with val (a, b)" + \
         "\n- 'gamma'/'min_gamma' with val (loc, scale)\n- 'binom' with val (p)\n- 't'/'logt'/'min_logt'/'logitt' with val (loc, scale, df)" + \
         "\n- 'fixed' with fixed_val (value)")
         
        
    return lpdf

### Sampler for the Auxiliary parameters, esp. the daily-varying parameters. ###
# 
# Auxiliary parameters are parameters that do not directly come from the distributions specified by the hyperprior that gets tuned by EM/EM-AIS/...,
# but they're parameters whose distributions that are specified by those parameters. For example, the BB-subjective-value hyperprior's first level
# is on the knot points and the standard deviation of the random walks. From these, we can sample daily-varying subjective values with 'sampleP_aux'.
#                            
def sampleP_aux_new(s_dat, P, spec, n):

    S = P.shape[0]
    
    P_aux = {}

    for param_name in spec.keys():
        param_type = spec[param_name]['type'] # Distribution of the hyperparameter
        param_val = spec[param_name]['val']
        
        if param_type == 'logBB_stat' or param_type == 'min_logBB_stat':   
            day_range = int((np.max(s_dat.block[:s_dat.T[n],n]) - np.min(s_dat.block[:s_dat.T[n],n])) / 2) + 1
            num_anchor_points = int((day_range - 2) / 7) + 1         
            P_aux[f'{param_name}_0'] = P[f'{param_name}_0'].values
            
            eps = np.random.normal(np.zeros((num_anchor_points*7, S)), P[f'{param_name}_sigma'].values)
            
            if param_type in ['logBB_stat', 'min_logBB_stat']:
                for s in range(num_anchor_points):
                    P[f'{param_name}_step_{s+1}'] = np.random.normal(loc = 0., scale = param_val[4], size = (S))
            
            d = 0
            while d < day_range - 1:
                d_min = int(d/7)*7
                P_aux[f'{param_name}_{d+1}'] = P_aux[f'{param_name}_{d}'] * np.exp(eps[d,:] - (np.sum(eps[d_min:(d_min+7),:], axis = 0))/7. + P[f'{param_name}_step_{int(d/7) + 1}'].values / 7.)
                d+=1
            
                
        elif param_type == 'BB_stat':
            day_range = int((np.max(s_dat.block[:s_dat.T[n],n]) - np.min(s_dat.block[:s_dat.T[n],n])) / 2) + 1
            num_anchor_points = int((day_range - 2) / 7) + 1         
            
            P_aux[f'{param_name}_0'] = P[f'{param_name}_0'].values
            
            eps = np.random.normal(np.zeros((num_anchor_points*7, S)), P[f'{param_name}_sigma'].values)
            
            for s in range(num_anchor_points):
                P[f'{param_name}_step_{s+1}'] = np.random.normal(loc = 0., scale = param_val[4], size = (S))
            
            d = 0
            while d < day_range - 1:
                d_min = int(d/7)*7
                P_aux[f'{param_name}_{d+1}'] = P_aux[f'{param_name}_{d}'] + eps[d,:] - (np.sum(eps[d_min:(d_min+7),:], axis = 0))/7. + P[f'{param_name}_step_{int(d/7) + 1}'].values / 7.
                d+=1
                
        elif param_type == 'logitBB_stat':
            day_range = int((np.max(s_dat.block[:s_dat.T[n],n]) - np.min(s_dat.block[:s_dat.T[n],n])) / 2) + 1
            num_anchor_points = int((day_range - 2) / 7) + 1         
            
            P_aux[f'{param_name}_0'] = P[f'{param_name}_0'].values
            P_aux[f'logit{param_name}_0'] = logit(P_aux[f'{param_name}_0'])
            
            eps = np.random.normal(np.zeros((num_anchor_points*7, S)), P[f'{param_name}_sigma'].values)
            
            for s in range(num_anchor_points):
                P[f'{param_name}_step_{s+1}'] = np.random.normal(loc = 0., scale = param_val[4], size = (S))
            
            d = 0
            while d < day_range - 1:
                d_min = int(d/7)*7
                P_aux[f'logit{param_name}_{d+1}'] = P_aux[f'logit{param_name}_{d}'] + eps[d,:] - (np.sum(eps[d_min:(d_min+7),:], axis = 0))/7. + P[f'{param_name}_step_{int(d/7) + 1}'].values / 7.
                P_aux[f'{param_name}_{d+1}'] = expit(P_aux[f'logit{param_name}_{d+1}'])
                d+=1
                    
        elif param_type in ['OU', 'logOU', 'min_logOU', 'logitOU']:
            session_times = s_dat.session_times[n]
            time_diffs = session_times[1:] - session_times[:-1]
            day_range = np.max(s_dat.session[:s_dat.T[n],n]) + 1
            
            # The stationary variance:
            stat_std = P[f'{param_name}_stat_std'].values
            
            # The time decay:
            ratetime_decay = np.exp(- time_diffs[:, None] * P[f'{param_name}_rate'].values[None, :])
            
            eps = np.random.normal(size = (session_times.shape[0]-1, S))
            
            P_aux[f'{param_name}_0'] = np.random.normal(loc = P[f'{param_name}_mu'], scale = stat_std)
            
            for d in range(session_times.shape[0] - 1):
                P_aux[f'{param_name}_{d+1}'] = ratetime_decay[d,:] * P_aux[f'{param_name}_{d}'] + \
                                               (1. - ratetime_decay[d,:]) * P[f'{param_name}_mu'] + \
                                               stat_std * np.sqrt(1. - np.square(ratetime_decay[d,:])) * eps[d,:]
            
            if param_type == 'logOU':
                for d in range(session_times.shape[0]):
                    P_aux[f'{param_name}_{d}'] = np.exp(P_aux[f'{param_name}_{d}'])
            elif param_type == 'min_logOU':
                for d in range(session_times.shape[0]):
                    P_aux[f'{param_name}_{d}'] = -np.exp(P_aux[f'{param_name}_{d}'])
            elif param_type == 'logitOU':
                for d in range(session_times.shape[0]):
                    P_aux[f'{param_name}_{d}'] = expit(P_aux[f'{param_name}_{d}'])
                    
        elif param_type in ['OU_fixed_mean', 'logOU_fixed_mean', 'min_logOU_fixed_mean', 'logitOU_fixed_mean']: # The same as non-fixed_mean
            session_times = s_dat.session_times[n]
            time_diffs = session_times[1:] - session_times[:-1]
            day_range = np.max(s_dat.session[:s_dat.T[n],n]) + 1
            
            # The stationary variance:
            stat_std = P[f'{param_name}_stat_std'].values
            
            # The time decay:
            ratetime_decay = np.exp(- time_diffs[:, None] * P[f'{param_name}_rate'].values[None, :])
            
            eps = np.random.normal(size = (session_times.shape[0]-1, S))
            
            P_aux[f'{param_name}_0'] = np.random.normal(loc = P[f'{param_name}_mu'], scale = stat_std)
            
            for d in range(session_times.shape[0] - 1):
                P_aux[f'{param_name}_{d+1}'] = ratetime_decay[d,:] * P_aux[f'{param_name}_{d}'] + \
                                               (1. - ratetime_decay[d,:]) * P[f'{param_name}_mu'] + \
                                               stat_std * np.sqrt(1. - np.square(ratetime_decay[d,:])) * eps[d,:]
            
            if param_type == 'logOU_fixed_mean':
                for d in range(session_times.shape[0]):
                    P_aux[f'{param_name}_{d}'] = np.exp(P_aux[f'{param_name}_{d}'])
            elif param_type == 'min_logOU_fixed_mean':
                for d in range(session_times.shape[0]):
                    P_aux[f'{param_name}_{d}'] = -np.exp(P_aux[f'{param_name}_{d}'])
            elif param_type == 'logitOU_fixed_mean':
                for d in range(session_times.shape[0]):
                    P_aux[f'{param_name}_{d}'] = expit(P_aux[f'{param_name}_{d}'])
                    
        # elif param_type in ['OU2', 'logOU2', 'min_logOU2', 'logitOU2']:
            # session_times = s_dat.session_times[n]
            # time_diffs = session_times[1:] - session_times[:-1]
            # day_range = np.max(s_dat.session[:s_dat.T[n],n]) + 1
            
            # # The stationary variance:
            # stat_std = P[f'{param_name}_sigma'].values / np.sqrt(2.*P[f'{param_name}_rate'].values)
            
            # # The time decay:
            # ratetime_decay = np.exp(- time_diffs[:, None] * P[f'{param_name}_rate'].values[None, :])
            
            # eps = np.random.normal(size = (session_times.shape[0]-1, S))
            
            # P_aux[f'd{param_name}_0'] = np.random.normal(loc = P[f'{param_name}_mu'], scale = stat_std)
            # P_aux[f'{param_name}_0']  = P[[f'tf_{param_name}_0'].values
            
            # for d in range(session_times.shape[0] - 1):
                # P_aux[f'{param_name}_{d+1}'] = P_aux[f'{param_name}_{d}'] + time_diffs[d]
                # P_aux[f'd{param_name}_{d+1}'] = ratetime_decay[d,:] * P_aux[f'{param_name}_{d}'] + \
                                               # (1. - ratetime_decay[d,:]) * P[f'{param_name}_mu'] + \
                                               # stat_std * np.sqrt(1. - np.square(ratetime_decay[d,:])) * eps[d,:]
                                               
           
            
            # if param_type == 'logOU2':
                # for d in range(session_times.shape[0]):
                    # P_aux[f'{param_name}_{d}'] = np.exp(P_aux[f'{param_name}_{d}'])
            # elif param_type == 'min_logOU2':
                # for d in range(session_times.shape[0]):
                    # P_aux[f'{param_name}_{d}'] = -np.exp(P_aux[f'{param_name}_{d}'])
            # elif param_type == 'logitOU2':
                # for d in range(session_times.shape[0]):
                    # P_aux[f'{param_name}_{d}'] = expit(P_aux[f'{param_name}_{d}'])
            
                                                                   
    if len(P_aux.keys()) == 0:
        P_aux = None
    else:
        P_aux = pd.DataFrame(P_aux)
        
        # Creates or transforms means and asymmetries:
        for param_name in spec.keys():
            param_type = spec[param_name]['type']
            
            if 'BB' in param_type:
                day_range = np.max(s_dat.session[:s_dat.T[n],n]) + 1
            elif 'OU' in param_type:
                day_range = np.max(s_dat.session[:s_dat.T[n],n]) + 1
                            
            if param_type[:5] == 'logBB' or param_type[:9] == 'min_logBB' or param_type[:5] == 'logOU' or param_type[:9] == 'min_logOU' or param_type[:7] == 'logitOU' or param_type[:11] == 'min_logitOU' or param_type[:7] == 'logitBB' or param_type[:11] == 'min_logitBB':
                if param_name[-5:] == '_asym':
                    base_name = param_name[:-5]
                    for d in range(day_range):
                        P_aux[f'{base_name}_r_{d}']   =  np.sqrt(np.square(P_aux[f'{base_name}_mean_{d}'].values) * P_aux[f'{base_name}_asym_{d}'].values)
                        P_aux[f'{base_name}_pun_{d}'] = -np.sqrt(np.square(P_aux[f'{base_name}_mean_{d}'].values) / P_aux[f'{base_name}_asym_{d}'].values)
                        P_aux[f'log{base_name}_r_{d}'] = np.log(np.abs(P_aux[f'{base_name}_r_{d}']))
                        P_aux[f'log{base_name}_pun_{d}'] = np.log(np.abs(P_aux[f'{base_name}_pun_{d}']))
                elif param_name[-9:] == '_asym_inv':
                    base_name = param_name[:-9]
                    for d in range(day_range):
                        P_aux[f'{base_name}_r_{d}']   =  np.sqrt(np.square(P_aux[f'{base_name}_mean_{d}'].values) / P_aux[f'{base_name}_asym_inv_{d}'].values)
                        P_aux[f'{base_name}_pun_{d}'] = -np.sqrt(np.square(P_aux[f'{base_name}_mean_{d}'].values) * P_aux[f'{base_name}_asym_inv_{d}'].values)
                        P_aux[f'log{base_name}_r_{d}'] = np.log(np.abs(P_aux[f'{base_name}_r_{d}']))
                        P_aux[f'log{base_name}_pun_{d}'] = np.log(np.abs(P_aux[f'{base_name}_pun_{d}']))
                elif param_name[-2:] == '_r':
                    base_name = param_name[:-2]
                    for d in range(day_range):
                        P_aux[f'{base_name}_mean_{d}'] = np.sqrt(P_aux[f'{base_name}_r_{d}'].values * np.abs(P_aux[f'{base_name}_pun_{d}'].values))
                        P_aux[f'{base_name}_asym_{d}'] = P_aux[f'{base_name}_r_{d}'].values / np.abs(P_aux[f'{base_name}_pun_{d}'].values)
                        P_aux[f'log{base_name}_mean_{d}'] = np.log(P_aux[f'{base_name}_mean_{d}'])
                        P_aux[f'log{base_name}_asym_{d}'] = np.log(P_aux[f'{base_name}_asym_{d}'])
                elif param_name[-6:] == '_pasym':
                    base_name = param_name[:-6]
                    for d in range(day_range):
                        P_aux[f'{base_name}_plus_{d}']  =  np.sqrt(np.square(P_aux[f'{base_name}_pmean_{d}'].values) * P_aux[f'{base_name}_pasym_{d}'].values)
                        P_aux[f'{base_name}_min_{d}']   = np.sqrt(np.square(P_aux[f'{base_name}_pmean_{d}'].values) / P_aux[f'{base_name}_pasym_{d}'].values)
                        P_aux[f'log{base_name}_plus_{d}'] = np.log(np.abs(P_aux[f'{base_name}_plus_{d}']))
                        P_aux[f'log{base_name}_min_{d}'] = np.log(np.abs(P_aux[f'{base_name}_min_{d}']))
                elif param_name[-5:] == '_plus':
                    base_name = param_name[:-5]
                    for d in range(day_range):
                        P_aux[f'{base_name}_pmean_{d}'] = np.sqrt(P_aux[f'{base_name}_plus_{d}'].values * P_aux[f'{base_name}_min_{d}'].values)
                        P_aux[f'{base_name}_pasym_{d}'] = P_aux[f'{base_name}_plus_{d}'].values / P_aux[f'{base_name}_min_{d}'].values
                        P_aux[f'log{base_name}_pmean_{d}'] = np.log(P_aux[f'{base_name}_pmean_{d}'])
                        P_aux[f'log{base_name}_pasym_{d}'] = np.log(P_aux[f'{base_name}_pasym_{d}'])
            
            if param_type[:5] == 'logBB' or param_type[:9] == 'min_logBB' or param_type[:5] == 'logOU' or param_type[:9] == 'min_logOU':
                # Create logabs transforms:
                for d in range(day_range):
                    P_aux[f'log{param_name}_{d}'] = np.log(np.abs(P_aux[f'{param_name}_{d}']))
                    
            elif param_type[:7] == 'logitBB' or param_type[:7] == 'logitOU' or param_type[:11] == 'min_logitBB' or param_type[:11] == 'min_logitOU':
                for d in range(day_range):
                    P_aux[f'logit{param_name}_{d}'] = logit(np.abs(P_aux[f'{param_name}_{d}']))
                
            
    return P_aux
            
    
### Functions for AIS in the following functions: ###
    
def lstats_to_stats(lmean, lsd):
    """
    Converts the mean and sd of log(X) to the mean and std of X, when log(X) is normally distributed.
    """
    
    mean = np.exp(lmean + .5 * lsd**2)
    sd = np.sqrt(np.exp(lmean) * (np.exp(2. * lsd**2) - np.exp(lsd**2)))
    return mean, sd

    
def construct_initial_sampler_EIS_mixture_new(m, n, q_style = None, q_type = None, random_init = False, num_mixtures = 1):
    q_spec = [{} for mx in range(num_mixtures)]
    
    m_spec = m.spec
    s_dat = m.subjects_data
    
    for mx in range(num_mixtures):
        for par_name in m.spec:
            par_type = m.spec[par_name]['type']
            
            if par_type == 'lognormal':
                vals = m_spec[par_name]['val'].copy()
                vals[0] += np.random.normal() * vals[1]  * random_init
                if q_type == 'standard':                    
                    q_spec[mx].update({par_name: {'type': 'lognormal', 'val': vals}})
                elif q_type == 'robust':
                    q_spec[mx].update({par_name: {'type': 'logt', 'val': vals + [5]}})
            elif par_type == 'min_lognormal':
                vals = m_spec[par_name]['val'].copy()
                vals[0] += np.random.normal() * vals[1]  * random_init
                if q_type == 'standard':                    
                    q_spec[mx].update({par_name: {'type': 'min_lognormal', 'val': vals}})
                elif q_type == 'robust':
                    q_spec[mx].update({par_name: {'type': 'min_logt', 'val': vals + [5]}})
            elif par_type == 'normal':
                vals = m_spec[par_name]['val'].copy()
                vals[0] += np.random.normal() * vals[1]  * random_init
                if q_type == 'standard':                    
                    q_spec[mx].update({par_name: {'type': 'normal', 'val': vals}})
                elif q_type == 'robust':
                    q_spec[mx].update({par_name: {'type': 't', 'val': vals + [5]}})
            elif par_type == 'logitnormal':
                vals = m_spec[par_name]['val'].copy()
                vals[0] += np.random.normal() * vals[1]  * random_init
                if q_type == 'standard':                    
                    q_spec[mx].update({par_name: {'type': 'logitnormal', 'val': vals}})
                elif q_type == 'robust':
                    q_spec[mx].update({par_name: {'type': 'logitt', 'val': vals + [5]}})
            elif par_type == 'fixed':
                q_spec[mx].update({par_name:{'type':'fixed', 'val':[], 'fixed_val': m_spec[par_name]['fixed_val']}})
            
            elif par_type == 'logBB_stat':
                day_range = int((np.max(s_dat.block[:s_dat.T[n],n]) - np.min(s_dat.block[:s_dat.T[n],n])) / 2) + 1
                num_anchor_points = int((day_range - 2) / 7) + 1
                d_EOW = num_anchor_points*7
                
                vals = m_spec[par_name]['val'].copy()
                
                # par_name_0:
                vals[0] += np.random.normal() * vals[1]  * random_init
                if q_type == 'standard':                    
                    q_spec[mx].update({f'{par_name}_0': {'type': 'lognormal', 'val': [vals[0],vals[1]]}})
                elif q_type == 'robust':                    
                    q_spec[mx].update({f'{par_name}_0': {'type': 'logt', 'val': [vals[0],vals[1],5]}})
                    
                # par_name_sigma:
                if q_type == 'standard':
                    q_spec[mx].update({f'{par_name}_sigma': {'type': 'lognormal', 'val': [vals[2],vals[3],5]}})
                    
                elif q_type == 'robust':
                    q_spec[mx].update({f'{par_name}_sigma': {'type': 'logt', 'val': [vals[2],vals[3],5]}})
                    
                # par_name_*
                for d in range(day_range-1):
                    w = int((d)/7)
                    mean_lmu = q_spec[mx][f'{par_name}_{w*7}']['val'][0]
                    mean_lstd = np.sqrt(np.square(q_spec[mx][f'{par_name}_{w*7}']['val'][1]) + np.square(vals[4]))

                    if q_type == 'standard':
                        q_spec[mx][f'{par_name}_{d+1}'] = {'type': 'lognormal', 
                                                   'val': [mean_lmu, mean_lstd]}
                    elif q_type == 'robust':
                        q_spec[mx][f'{par_name}_{d+1}'] = {'type': 'logt', 
                                                   'val': [mean_lmu, mean_lstd, 5]}
                # d_EOW:
                if d_EOW >= day_range:
                    mean_lmu = q_spec[mx][f'{par_name}_{d_EOW-7}']['val'][0]
                    mean_lstd = np.sqrt(np.square(q_spec[mx][f'{par_name}_{d_EOW-7}']['val'][1]) + np.square(vals[4]))
                    
                    if q_type == 'standard':
                        q_spec[mx][f'{par_name}_{d_EOW}'] = {'type': 'lognormal', 
                                                   'val': [mean_lmu, mean_lstd]}
                    elif q_type == 'robust':
                        q_spec[mx][f'{par_name}_{d_EOW}'] = {'type': 'logt', 
                                                   'val': [mean_lmu, mean_lstd, 5]}
            elif par_type == 'min_logBB_stat':
                day_range = int((np.max(s_dat.block[:s_dat.T[n],n]) - np.min(s_dat.block[:s_dat.T[n],n])) / 2) + 1
                num_anchor_points = int((day_range - 2) / 7) + 1
                d_EOW = num_anchor_points*7
                
                vals = m_spec[par_name]['val'].copy()
                
                # par_name_0:
                vals[0] += np.random.normal() * vals[1]  * random_init
                if q_type == 'standard':                    
                    q_spec[mx].update({f'{par_name}_0': {'type': 'min_lognormal', 'val': [vals[0],vals[1]]}})
                elif q_type == 'robust':                    
                    q_spec[mx].update({f'{par_name}_0': {'type': 'min_logt', 'val': [vals[0],vals[1],5]}})
                    
                # par_name_sigma:
                if q_type == 'standard':
                    q_spec[mx].update({f'{par_name}_sigma': {'type': 'lognormal', 'val': [vals[2],vals[3],5]}})                    
                elif q_type == 'robust':
                    q_spec[mx].update({f'{par_name}_sigma': {'type': 'logt', 'val': [vals[2],vals[3],5]}})
                
                # par_name_*
                for d in range(day_range-1):
                    w = int((d)/7)
                    mean_lmu = q_spec[mx][f'{par_name}_{w*7}']['val'][0]
                    mean_lstd = np.sqrt(np.square(q_spec[mx][f'{par_name}_{w*7}']['val'][1]) + np.square(vals[4]))

                    if q_type == 'standard':
                        q_spec[mx][f'{par_name}_{d+1}'] = {'type': 'min_lognormal', 
                                                   'val': [mean_lmu, mean_lstd]}
                    elif q_type == 'robust':
                        q_spec[mx][f'{par_name}_{d+1}'] = {'type': 'min_logt', 
                                                   'val': [mean_lmu, mean_lstd, 5]}
                                                   
                # d_EOW:
                if d_EOW >= day_range:
                    mean_lmu = q_spec[mx][f'{par_name}_{d_EOW-7}']['val'][0]
                    mean_lstd = np.sqrt(np.square(q_spec[mx][f'{par_name}_{d_EOW-7}']['val'][1]) + np.square(vals[4]))
                    
                    if q_type == 'standard':
                        q_spec[mx][f'{par_name}_{d_EOW}'] = {'type': 'min_lognormal', 
                                                   'val': [mean_lmu, mean_lstd]}
                    elif q_type == 'robust':
                        q_spec[mx][f'{par_name}_{d_EOW}'] = {'type': 'min_logt', 
                                                   'val': [mean_lmu, mean_lstd, 5]}
            elif par_type == 'BB_stat':
                day_range = int((np.max(s_dat.block[:s_dat.T[n],n]) - np.min(s_dat.block[:s_dat.T[n],n])) / 2) + 1
                num_anchor_points = int((day_range - 2) / 7) + 1
                d_EOW = num_anchor_points*7
                
                vals = m_spec[par_name]['val'].copy()
                
                # par_name_0:
                vals[0] += np.random.normal() * vals[1]  * random_init
                if q_type == 'standard':                    
                    q_spec[mx].update({f'{par_name}_0': {'type': 'normal', 'val': [vals[0],vals[1]]}})
                elif q_type == 'robust':                    
                    q_spec[mx].update({f'{par_name}_0': {'type': 't', 'val': [vals[0],vals[1],5]}})
                    
                # par_name_sigma:
                if q_type == 'standard':
                    q_spec[mx].update({f'{par_name}_sigma': {'type': 'lognormal', 'val': [vals[2],vals[3],5]}})                    
                elif q_type == 'robust':
                    q_spec[mx].update({f'{par_name}_sigma': {'type': 'logt', 'val': [vals[2],vals[3],5]}})
                                    
                # par_name_*
                for d in range(day_range-1):
                    w = int((d)/7)
                    mean_lmu = q_spec[mx][f'{par_name}_{w*7}']['val'][0]
                    mean_lstd = np.sqrt(np.square(q_spec[mx][f'{par_name}_{w*7}']['val'][1]) + np.square(vals[4]))

                    if q_type == 'standard':
                        q_spec[mx][f'{par_name}_{d+1}'] = {'type': 'normal', 
                                                   'val': [mean_lmu, mean_lstd]}
                    elif q_type == 'robust':
                        q_spec[mx][f'{par_name}_{d+1}'] = {'type': 't', 
                                                   'val': [mean_lmu, mean_lstd, 5]}
                # d_EOW:
                if d_EOW >= day_range:
                    mean_lmu = q_spec[mx][f'{par_name}_{d_EOW-7}']['val'][0]
                    mean_lstd = np.sqrt(np.square(q_spec[mx][f'{par_name}_{d_EOW-7}']['val'][1]) + np.square(vals[4]))
                    
                    if q_type == 'standard':
                        q_spec[mx][f'{par_name}_{d_EOW}'] = {'type': 'normal', 
                                                   'val': [mean_lmu, mean_lstd]}
                    elif q_type == 'robust':
                        q_spec[mx][f'{par_name}_{d_EOW}'] = {'type': 't', 
                                                   'val': [mean_lmu, mean_lstd, 5]}
            elif par_type == 'logitBB_stat':
                day_range = int((np.max(s_dat.block[:s_dat.T[n],n]) - np.min(s_dat.block[:s_dat.T[n],n])) / 2) + 1
                num_anchor_points = int((day_range - 2) / 7) + 1
                d_EOW = num_anchor_points*7
                
                vals = m_spec[par_name]['val'].copy()
                
                # par_name_0:
                vals[0] += np.random.normal() * vals[1]  * random_init
                if q_type == 'standard':                    
                    sys.exit('AIS_fit > no "q_type == standard" for "logitBB_stat" as a prior. Only "robust".')
                elif q_type == 'robust':                    
                    q_spec[mx].update({f'{par_name}_0': {'type': 'logitt', 'val': [vals[0],vals[1],5]}})
                    
                # par_name_sigma:
                if q_type == 'standard':
                    q_spec[mx].update({f'{par_name}_sigma': {'type': 'lognormal', 'val': [vals[2],vals[3],5]}})                    
                elif q_type == 'robust':
                    q_spec[mx].update({f'{par_name}_sigma': {'type': 'logt', 'val': [vals[2],vals[3],5]}})
                    
                # par_name_*
                for d in range(day_range-1):
                    w = int((d)/7)
                    mean_lmu = q_spec[mx][f'{par_name}_{w*7}']['val'][0]
                    mean_lstd = np.sqrt(np.square(q_spec[mx][f'{par_name}_{w*7}']['val'][1]) + np.square(vals[4]))

                    q_spec[mx][f'{par_name}_{d+1}'] = {'type': 'logitt', 
                                               'val': [mean_lmu, mean_lstd, 5]}
                # d_EOW:
                if d_EOW >= day_range:
                    mean_lmu = q_spec[mx][f'{par_name}_{d_EOW-7}']['val'][0]
                    mean_lstd = np.sqrt(np.square(q_spec[mx][f'{par_name}_{d_EOW-7}']['val'][1]) + np.square(vals[4]))
                    
                    q_spec[mx][f'{par_name}_{d_EOW}'] = {'type': 'logitt', 
                                               'val': [mean_lmu, mean_lstd, 5]}

            elif par_type in ['OU', 'logOU', 'min_logOU', 'logitOU']:
                day_range = np.max(s_dat.session[:s_dat.T[n], n]) + 1
                
                vals = m_spec[par_name]['val'].copy()
                
                # mu, rate, sigma:
                if q_type == 'standard':
                    q_spec[mx][f'{par_name}_mu'] = {'type': 'normal', 
                                               'val': [vals[0], vals[1]]}
                    q_spec[mx][f'{par_name}_rate'] = {'type': 'lognormal', 
                                               'val': [vals[2], vals[3]]}
                    q_spec[mx][f'{par_name}_stat_std'] = {'type': 'lognormal', 
                                               'val': [vals[4], vals[5]]}
                elif q_type == 'robust':
                    q_spec[mx][f'{par_name}_mu'] = {'type': 't', 
                                               'val': [vals[0], vals[1], 5]}
                    q_spec[mx][f'{par_name}_rate'] = {'type': 'logt', 
                                               'val': [vals[2], vals[3], 5]}
                    q_spec[mx][f'{par_name}_stat_std'] = {'type': 'logt', 
                                               'val': [vals[4], vals[5], 5]}
                
                stat_var_approx = np.exp(vals[4] + vals[5]**2 / 2)
                
                perturbed_mean = vals[0] #+ np.random.normal() * vals[1]  * random_init            
                
                for d in range(day_range):
                    if q_type == 'standard':
                        q_spec[mx][f'tf_{par_name}_{d}'] = {'type': 'normal', 
                                                   'val': [perturbed_mean, stat_var_approx]}
                    elif q_type == 'robust':
                        q_spec[mx][f'tf_{par_name}_{d}'] = {'type': 't', 
                                                   'val': [perturbed_mean, stat_var_approx, 5]}    
            elif par_type in ['OU_fixed_mean', 'logOU_fixed_mean', 'min_logOU_fixed_mean', 'logitOU_fixed_mean']:
                day_range = np.max(s_dat.session[:s_dat.T[n], n]) + 1
                
                vals = m_spec[par_name]['val'].copy()
                mean = m_spec[par_name]['fixed_val']
                
                # mu, rate, sigma:
                if q_type == 'standard':
                    q_spec[mx][f'{par_name}_mu'] = {'type': 'fixed', 
                                               'val': [],
                                               'fixed_val': mean}
                    q_spec[mx][f'{par_name}_rate'] = {'type': 'lognormal', 
                                               'val': [vals[0], vals[1]]}
                    q_spec[mx][f'{par_name}_stat_std'] = {'type': 'lognormal', 
                                               'val': [vals[2], vals[3]]}
                elif q_type == 'robust':
                    q_spec[mx][f'{par_name}_mu'] = {'type': 'fixed', 
                                               'val': [],
                                               'fixed_val': mean}
                    q_spec[mx][f'{par_name}_rate'] = {'type': 'logt', 
                                               'val': [vals[0], vals[1], 5]}
                    q_spec[mx][f'{par_name}_stat_std'] = {'type': 'logt', 
                                               'val': [vals[2], vals[3], 5]}
                
                stat_var_approx = np.exp(vals[2] + vals[3]**2 / 2)
                
                perturbed_mean = vals[0] #+ np.random.normal() * vals[1]  * random_init            
                
                for d in range(day_range):
                    if q_type == 'standard':
                        q_spec[mx][f'tf_{par_name}_{d}'] = {'type': 'normal', 
                                                   'val': [perturbed_mean, stat_var_approx]}
                    elif q_type == 'robust':
                        q_spec[mx][f'tf_{par_name}_{d}'] = {'type': 't', 
                                                   'val': [perturbed_mean, stat_var_approx, 5]}                                           
                                            
            else:
                sys.exit(f"AIS > the distribution {par_type} in spec does not have an implementation yet (parameter: {par_name}.")
                                               
    return q_spec

    
def P_q_to_m_new(P, m, n, max_day = 1000):
    P_m = {}
    P_aux = {}
    
    s_dat = m.subjects_data
    
    for par_name in m.spec:
        par_type = m.spec[par_name]['type']
        
        if par_type not in ['logBB_stat', 'min_logBB_stat', 'BB_stat', 'logitBB_stat', 'OU', 'logOU', 'min_logOU', 'logitOU', 'OU_fixed_mean', 'logOU_fixed_mean', 'min_logOU_fixed_mean', 'logitOU_fixed_mean']:
            P_m[par_name] = P[par_name].values
        elif par_type in ['logBB_stat', 'min_logBB_stat', 'BB_stat', 'logitBB_stat']:
            day_range = min(int((np.max(s_dat.block[:s_dat.T[n],n]) - np.min(s_dat.block[:s_dat.T[n],n])) / 2) + 1, max_day)
            num_anchor_points = int((day_range - 2) / 7) + 1
        
            # par_name_0, par_name_sigma:
            P_m[f'{par_name}_0']     = P[f'{par_name}_0'].values
            P_m[f'{par_name}_sigma'] = P[f'{par_name}_sigma'].values
            
            # steps:
            if par_type in ['logBB_stat']:
                w = 0; flag = True
                while flag:
                    if f'{par_name}_{(w+1)*7}' in P.columns:
                        P_m[f'{par_name}_step_{w+1}'] = (np.log(P[f'{par_name}_{(w+1)*7}']) - np.log(P[f'{par_name}_{w*7}'])).values
                        w+=1
                    else:
                        flag = False
            elif par_type in ['min_logBB_stat']:
                
                w = 0; flag = True
                while flag:
                    if f'{par_name}_{(w+1)*7}' in P.columns:
                        P_m[f'{par_name}_step_{w+1}'] = (np.log(-P[f'{par_name}_{(w+1)*7}']) - np.log(-P[f'{par_name}_{w*7}'])).values
                        w+=1
                    else:
                        flag = False
            elif par_type == 'BB_stat':
                w = 0; flag = True
                while flag:
                    if f'{par_name}_{(w+1)*7}' in P.columns:
                        P_m[f'{par_name}_step_{w+1}'] = (P[f'{par_name}_{(w+1)*7}'] - P[f'{par_name}_{w*7}']).values
                        w+=1
                    else:
                        flag = False
            elif par_type == 'logitBB_stat':
                w = 0; flag = True
                while flag:
                    if f'{par_name}_{(w+1)*7}' in P.columns:
                        P_m[f'{par_name}_step_{w+1}'] = logit(P[f'{par_name}_{(w+1)*7}']) - logit(P[f'{par_name}_{w*7}'])
                        w+=1
                    else:
                        flag = False
            
            # day parameters:
            for d in range(day_range):
                P_aux[f'{par_name}_{d}'] = P[f'{par_name}_{d}'].values
                
        elif par_type in ['OU', 'logOU', 'min_logOU', 'logitOU']:
            day_range = np.max(s_dat.session[:s_dat.T[n],n]) + 1
        
            P_m[f'{par_name}_mu']    = P[f'{par_name}_mu'].values
            P_m[f'{par_name}_rate']  = P[f'{par_name}_rate'].values
            P_m[f'{par_name}_stat_std'] = P[f'{par_name}_stat_std'].values
            
            # day parameters:
            if par_type == 'OU':
                for d in range(day_range):
                    P_aux[f'{par_name}_{d}'] = P[f'tf_{par_name}_{d}'].values
            elif par_type == 'min_logOU':
                for d in range(day_range):
                    P_aux[f'{par_name}_{d}'] = -np.exp(P[f'tf_{par_name}_{d}'].values)
            elif par_type == 'logOU':
                for d in range(day_range):
                    P_aux[f'{par_name}_{d}'] = np.exp(P[f'tf_{par_name}_{d}'].values)
            elif par_type == 'logitOU':
                for d in range(day_range):
                    P_aux[f'{par_name}_{d}'] = expit(P[f'tf_{par_name}_{d}'].values)
                    
        elif par_type in ['OU_fixed_mean', 'logOU_fixed_mean', 'min_logOU_fixed_mean', 'logitOU_fixed_mean']:
            day_range = np.max(s_dat.session[:s_dat.T[n],n]) + 1
        
            P_m[f'{par_name}_mu']    = P[f'{par_name}_mu'].values
            P_m[f'{par_name}_rate']  = P[f'{par_name}_rate'].values
            P_m[f'{par_name}_stat_std'] = P[f'{par_name}_stat_std'].values
            
            # day parameters:
            if par_type == 'OU_fixed_mean':
                for d in range(day_range):
                    P_aux[f'{par_name}_{d}'] = P[f'tf_{par_name}_{d}'].values
            elif par_type == 'min_logOU_fixed_mean':
                for d in range(day_range):
                    P_aux[f'{par_name}_{d}'] = -np.exp(P[f'tf_{par_name}_{d}'].values)
            elif par_type == 'logOU_fixed_mean':
                for d in range(day_range):
                    P_aux[f'{par_name}_{d}'] = np.exp(P[f'tf_{par_name}_{d}'].values)
            elif par_type == 'logitOU_fixed_mean':
                for d in range(day_range):
                    P_aux[f'{par_name}_{d}'] = expit(P[f'tf_{par_name}_{d}'].values)
        
    if len(P_aux.keys()) == 0:
        P_aux = None
    else:
        P_aux = pd.DataFrame(P_aux)
        
    
    # Creates or transforms means and asymmetries:
    for param_name in m.spec.keys():
        param_type = m.spec[param_name]['type']
        
        if 'BB' in param_type:
            day_range = np.max(s_dat.session[:s_dat.T[n],n]) + 1
        elif 'OU' in param_type:
            day_range = np.max(s_dat.session[:s_dat.T[n],n]) + 1
        
        if param_type[:5] == 'logBB' or param_type[:9] == 'min_logBB' or param_type[:5] == 'logOU' or param_type[:9] == 'min_logOU' or param_type[:7] == 'logitOU' or param_type[:11] == 'min_logitOU' or param_type[:7] == 'logitBB' or param_type[:11] == 'min_logitBB':
            if param_name[-5:] == '_asym':
                base_name = param_name[:-5]
                for d in range(day_range):
                    P_aux[f'{base_name}_r_{d}']   =  np.sqrt(np.square(P_aux[f'{base_name}_mean_{d}'].values) * P_aux[f'{base_name}_asym_{d}'].values)
                    P_aux[f'{base_name}_pun_{d}'] = -np.sqrt(np.square(P_aux[f'{base_name}_mean_{d}'].values) / P_aux[f'{base_name}_asym_{d}'].values)
                    P_aux[f'log{base_name}_r_{d}'] = np.log(np.abs(P_aux[f'{base_name}_r_{d}']))
                    P_aux[f'log{base_name}_pun_{d}'] = np.log(np.abs(P_aux[f'{base_name}_pun_{d}']))
            elif param_name[-9:] == '_asym_inv':
                base_name = param_name[:-9]
                for d in range(day_range):
                    P_aux[f'{base_name}_r_{d}']   =  np.sqrt(np.square(P_aux[f'{base_name}_mean_{d}'].values) / P_aux[f'{base_name}_asym_inv_{d}'].values)
                    P_aux[f'{base_name}_pun_{d}'] = -np.sqrt(np.square(P_aux[f'{base_name}_mean_{d}'].values) * P_aux[f'{base_name}_asym_inv_{d}'].values)
                    P_aux[f'log{base_name}_r_{d}'] = np.log(np.abs(P_aux[f'{base_name}_r_{d}']))
                    P_aux[f'log{base_name}_pun_{d}'] = np.log(np.abs(P_aux[f'{base_name}_pun_{d}']))
            elif param_name[-2:] == '_r':
                base_name = param_name[:-2]
                for d in range(day_range):
                    P_aux[f'{base_name}_mean_{d}'] = np.sqrt(P_aux[f'{base_name}_r_{d}'].values * np.abs(P_aux[f'{base_name}_pun_{d}'].values))
                    P_aux[f'{base_name}_asym_{d}'] = P_aux[f'{base_name}_r_{d}'].values / np.abs(P_aux[f'{base_name}_pun_{d}'].values)
                    P_aux[f'log{base_name}_mean_{d}'] = np.log(P_aux[f'{base_name}_mean_{d}'])
                    P_aux[f'log{base_name}_asym_{d}'] = np.log(P_aux[f'{base_name}_asym_{d}'])
            elif param_name[-6:] == '_pasym':
                base_name = param_name[:-6]
                for d in range(day_range):
                    P_aux[f'{base_name}_plus_{d}']  =  np.sqrt(np.square(P_aux[f'{base_name}_pmean_{d}'].values) * P_aux[f'{base_name}_pasym_{d}'].values)
                    P_aux[f'{base_name}_min_{d}']   = np.sqrt(np.square(P_aux[f'{base_name}_pmean_{d}'].values) / P_aux[f'{base_name}_pasym_{d}'].values)
                    P_aux[f'log{base_name}_plus_{d}'] = np.log(np.abs(P_aux[f'{base_name}_plus_{d}']))
                    P_aux[f'log{base_name}_min_{d}'] = np.log(np.abs(P_aux[f'{base_name}_min_{d}']))
            elif param_name[-5:] == '_plus':
                base_name = param_name[:-5]
                for d in range(day_range):
                    P_aux[f'{base_name}_pmean_{d}'] = np.sqrt(P_aux[f'{base_name}_plus_{d}'].values * P_aux[f'{base_name}_min_{d}'].values)
                    P_aux[f'{base_name}_pasym_{d}'] = P_aux[f'{base_name}_plus_{d}'].values / P_aux[f'{base_name}_min_{d}'].values
                    P_aux[f'log{base_name}_pmean_{d}'] = np.log(P_aux[f'{base_name}_pmean_{d}'])
                    P_aux[f'log{base_name}_pasym_{d}'] = np.log(P_aux[f'{base_name}_pasym_{d}'])
        
            if param_type[:5] == 'logBB' or param_type[:9] == 'min_logBB' or param_type[:5] == 'logOU' or param_type[:9] == 'min_logOU':
                # Create logabs transforms:
                for d in range(day_range):
                    P_aux[f'log{param_name}_{d}'] = np.log(np.abs(P_aux[f'{param_name}_{d}']))
                    
            elif param_type[:7] == 'logitBB' or param_type[:7] == 'logitOU' or param_type[:11] == 'min_logitBB' or param_type[:11] == 'min_logitOU':
                for d in range(day_range):
                    P_aux[f'logit{param_name}_{d}'] = logit(np.abs(P_aux[f'{param_name}_{d}']))
                                
    return pd.DataFrame(P_m), P_aux

    
# def P_m_to_q_new(P, P_aux, m):
    # P_q = {}
    
    # s_dat = m.subjects_data
    # day_range = int((np.max(s_dat.block) - np.min(s_dat.block)) / 2) + 1
    # num_anchor_points = int((day_range - 2) / 7) + 1
        
    # for par_name in m.spec:
        # par_type = m.spec['type']
        # if par_type not in ['logBB_stat', 'min_logBB_stat']:
            # P_q[par_name] = P[par_name].values
        # elif par_type in ['logBB_stat', 'min_logBB_stat']:
            # # par_name_0, par_name_sigma:
            # P_q[f'{par_name}_0']     = P[f'{par_name}_0'].values
            # P_q[f'{par_name}_sigma'] = P[f'{par_name}_sigma'].values
            
            # if par_type in ['logBB_stat', 'min_logBB_stat']:
                # for d in range(day_range + 1):
                    # P_q[f'{par_name}_{d}'] = P_aux[f'{par_name}_{d}'].values
                    # #P_q[f'{par_name}_{d}'] = P_aux[f'{par_name}_{d}'].values
            # elif par_type in ['logBB_stat', 'min_logBB_stat']:
                # d = 0; flag = True
                # while flag:
                    # if f'{par_name}_{d}' in P_aux.columns:
                        # P_q[f'{par_name}_{d}'] = P_aux[f'{par_name}_{d}'].values
                        # d += 1
                    # else:
                        # flag = False
                
    # return pd.DataFrame(P_q)

# def P_q_to_samples(m,P, max_day = 29, day_range = 29):
    # # May be deprecated
    
    # # end of week day:
    # d_EOW = max(int((max_day - 2) / 7) * 7 + 7, 7) + 1
    
    # P_new_ext = P.copy()
    
    # if m.R_style == 'subjective-n0-day-BB-V3-2':
        # clms_to_drop1 = [f'R_mean_{d}' for d in range(d_EOW , day_range)] \
                       # + [f'R_asym_{d}' for d in range(d_EOW , day_range)]
        # P_new_ext.drop(columns = clms_to_drop1, inplace = True)
        
        # clms_to_drop2 = [f'R_mean_{d}' for d in range(max_day, d_EOW)] \
                       # + [f'R_asym_{d}' for d in range(max_day, d_EOW)]
                       
        
    # for par_name in P_new_ext.columns:
        # if par_name[:6] == 'R_mean' or par_name[:6] == 'R_asym':
            # P_new_ext[par_name] = np.log(P_new_ext[par_name])
        # elif m.spec[par_name]['type'] == 'lognormal':
            # P_new_ext[par_name] = np.log(P_new_ext[par_name])
        # elif m.spec[par_name]['type'] == 'min_lognormal':
            # P_new_ext[par_name] = np.log(-P_new_ext[par_name])
            
    # if m.R_style == 'subjective-n0-day-BB-V3-2':
        # P_new = P_new_ext.drop(columns = clms_to_drop2)
        
    # return P_new

# def samples_to_P_q(m, samples, columns):
    # # may be deprecated
    
    # P_q = pd.DataFrame(samples.copy(), columns = columns)
    
    # for par_name in columns:
        # if par_name[:6] == 'R_mean' or par_name[:6] == 'R_asym':
            # P_q[par_name] = np.exp(P_q[par_name])
        # elif m.spec[par_name]['type'] == 'lognormal':
            # P_q[par_name] = np.exp(P_q[par_name])
        # elif m.spec[par_name]['type'] == 'min_lognormal':
            # P_q[par_name] = -np.exp(P_q[par_name])
            
    # return P_q
            
def compute_lpdf_prior_new(P_m, P_aux, spec, s_dat, m, n, max_day = 10000):
    """
    max_day are inclusive. max_day is deprecated.
    """
    
    lpdf = np.zeros((P_m.shape[0]))
    
    for par_name in spec:
        par_type = spec[par_name]['type']
        
        if par_type == 'fixed':
            1+1
        elif par_type not in ['logBB_stat', 'min_logBB_stat','BB_stat', 'logitBB_stat', 'OU', 'logOU', 'min_logOU', 'logitOU', 'OU_fixed_mean', 'logOU_fixed_mean', 'min_logOU_fixed_mean', 'logitOU_fixed_mean']:
            lpdf += compute_lpdf({par_name: spec[par_name]}, P_m)
        
        elif par_type == 'logBB_stat':
            day_range = min(np.max(s_dat.session[:s_dat.T[n],n]) + 1, max_day)
            num_anchor_points = int((day_range - 2) / 7) + 1
            d_EOW = num_anchor_points*7
            vals = spec[par_name]['val']
            P_style = {f'{par_name}_0': {'type': 'lognormal',
                                         'val' : [vals[0], vals[1]]},
                       f'{par_name}_sigma': {'type': 'lognormal',
                                         'val' : [vals[2], vals[3]]}}
            for w in range(num_anchor_points):
                P_style[f'{par_name}_step_{w+1}'] = {'type': 'normal',
                                                     'val' : [0., vals[4]]}

            lpdf += compute_lpdf(P_style, P_m)
                
            # Brownian bridge pdf
            V_BB_tf = np.array([[(7. - max(i,j)) * min(i,j) / 7. for j in (1. + np.arange(6))] for i in (1. + np.arange(6))])
                    
            for w in range(num_anchor_points - 1): # Number of weeks-1.
                week_params_demeaned = np.log(P_aux[[f'{par_name}_{d}' for d in w * 7 + np.arange(1,7).astype('int64')]].values) - np.log(P_aux[f'{par_name}_{w*7}'].values[:,None]) - np.arange(1,7)[None,:] / 7. * (P_m[f'{par_name}_step_{(w+1)}'].values[:,None])
                week_params_std = week_params_demeaned / P_m[f'{par_name}_sigma'].values[:,None]

                lpdf += scipy.stats.multivariate_normal.logpdf(week_params_std, cov = V_BB_tf) - 6. * np.log(P_m[f'{par_name}_sigma'].values) - np.sum(np.log(P_aux[[f'{par_name}_{d}' for d in w * 7 + np.arange(1,7).astype('int64')]].values), axis = 1)
            
            num_remaining = day_range - 1 - 7*(num_anchor_points-1)
            if num_remaining == 7:
                num_remaining = 6
            #print(num_remaining)
            if num_remaining > 0:
                w = num_anchor_points - 1

                week_params_demeaned = np.log(P_aux[[f'{par_name}_{d}' for d in w * 7 + np.arange(1,num_remaining+1).astype('int64')]].values) - np.log(P_aux[f'{par_name}_{w*7}'].values[:,None]) - np.arange(1,num_remaining+1)[None,:] / 7. * (P_m[f'{par_name}_step_{(w+1)}'].values[:,None])
                week_params_std = week_params_demeaned / P_m[f'{par_name}_sigma'].values[:,None]
              
                lpdf += scipy.stats.multivariate_normal.logpdf(week_params_std, cov = V_BB_tf[:num_remaining, :num_remaining]) - float(num_remaining) * np.log(P_m[f'{par_name}_sigma'].values) - np.sum(np.log(P_aux[[f'{par_name}_{d}' for d in w * 7 + np.arange(1,num_remaining+1).astype('int64')]].values), axis = 1)
        elif par_type == 'min_logBB_stat':
            day_range = min(np.max(s_dat.session[:s_dat.T[n],n]) + 1, max_day)
            num_anchor_points = int((day_range - 2) / 7) + 1
            d_EOW = num_anchor_points*7
            vals = spec[par_name]['val']
            P_style = {f'{par_name}_0': {'type': 'min_lognormal',
                                         'val' : [vals[0], vals[1]]},
                       f'{par_name}_sigma': {'type': 'lognormal',
                                         'val' : [vals[2], vals[3]]}}
            for w in range(num_anchor_points):
                P_style[f'{par_name}_step_{w+1}'] = {'type': 'normal',
                                                     'val' : [0., vals[4]]}
            lpdf += compute_lpdf(P_style, P_m)
                
            # Brownian bridge pdf
            V_BB_tf = np.array([[(7. - max(i,j)) * min(i,j) / 7. for j in (1. + np.arange(6))] for i in (1. + np.arange(6))])
                    
            for w in range(num_anchor_points - 1): # Number of weeks-1.
                week_params_demeaned = np.log(-P_aux[[f'{par_name}_{d}' for d in w * 7 + np.arange(1,7).astype('int64')]].values) - np.log(-P_aux[f'{par_name}_{w*7}'].values[:,None]) - np.arange(1,7)[None,:] / 7. * (P_m[f'{par_name}_step_{(w+1)}'].values[:,None])
                week_params_std = week_params_demeaned / P_m[f'{par_name}_sigma'].values[:,None]
                
                lpdf += scipy.stats.multivariate_normal.logpdf(week_params_std, cov = V_BB_tf) - 6. * np.log(P_m[f'{par_name}_sigma'].values) - np.sum(np.log(-P_aux[[f'{par_name}_{d}' for d in w * 7 + np.arange(1,7).astype('int64')]].values), axis = 1)
            num_remaining = day_range - 1 - 7*(num_anchor_points-1)
            if num_remaining == 7:
                num_remaining = 6

            if num_remaining > 0:
                w = num_anchor_points - 1

                week_params_demeaned = np.log(-P_aux[[f'{par_name}_{d}' for d in w * 7 + np.arange(1,num_remaining+1).astype('int64')]].values) - np.log(-P_aux[f'{par_name}_{w*7}'].values[:,None]) - np.arange(1,num_remaining+1)[None,:] / 7. * (P_m[f'{par_name}_step_{(w+1)}'].values[:,None])
                week_params_std = week_params_demeaned / P_m[f'{par_name}_sigma'].values[:,None]

                lpdf += scipy.stats.multivariate_normal.logpdf(week_params_std, cov = V_BB_tf[:num_remaining, :num_remaining]) - float(num_remaining) * np.log(P_m[f'{par_name}_sigma'].values) - np.sum(np.log(-P_aux[[f'{par_name}_{d}' for d in w * 7 + np.arange(1,num_remaining+1).astype('int64')]].values), axis = 1)
        elif par_type == 'BB_stat':
            day_range = min(np.max(s_dat.session[:s_dat.T[n],n]) + 1, max_day)
            num_anchor_points = int((day_range - 2) / 7) + 1
            d_EOW = num_anchor_points*7
            vals = spec[par_name]['val']
            P_style = {f'{par_name}_0': {'type': 'normal',
                                         'val' : [vals[0], vals[1]]},
                       f'{par_name}_sigma': {'type': 'lognormal',
                                         'val' : [vals[2], vals[3]]}}
            for w in range(num_anchor_points):
                P_style[f'{par_name}_step_{w+1}'] = {'type': 'normal',
                                                     'val' : [0., vals[4]]}

            lpdf += compute_lpdf(P_style, P_m)
                
            # Brownian bridge pdf
            V_BB_tf = np.array([[(7. - max(i,j)) * min(i,j) / 7. for j in (1. + np.arange(6))] for i in (1. + np.arange(6))])
                    
            for w in range(num_anchor_points - 1): # Number of weeks-1.
                week_params_demeaned = P_aux[[f'{par_name}_{d}' for d in w * 7 + np.arange(1,7).astype('int64')]].values - P_aux[f'{par_name}_{w*7}'].values[:,None] - np.arange(1,7)[None,:] / 7. * (P_m[f'{par_name}_step_{(w+1)}'].values[:,None])
                week_params_std = week_params_demeaned / P_m[f'{par_name}_sigma'].values[:,None]

                lpdf += scipy.stats.multivariate_normal.logpdf(week_params_std, cov = V_BB_tf) - 6. * np.log(P_m[f'{par_name}_sigma'].values) 
                
            num_remaining = day_range - 1 - 7*(num_anchor_points-1)
            if num_remaining == 7:
                num_remaining = 6
            #print(num_remaining)
            if num_remaining > 0:
                w = num_anchor_points - 1

                week_params_demeaned = P_aux[[f'{par_name}_{d}' for d in w * 7 + np.arange(1,num_remaining+1).astype('int64')]].values - P_aux[f'{par_name}_{w*7}'].values[:,None] - np.arange(1,num_remaining+1)[None,:] / 7. * (P_m[f'{par_name}_step_{(w+1)}'].values[:,None])
                week_params_std = week_params_demeaned / P_m[f'{par_name}_sigma'].values[:,None]
              
                lpdf += scipy.stats.multivariate_normal.logpdf(week_params_std, cov = V_BB_tf[:num_remaining, :num_remaining]) - float(num_remaining) * np.log(P_m[f'{par_name}_sigma'].values) 
        elif par_type == 'logitBB_stat':
            day_range = min(np.max(s_dat.session[:s_dat.T[n],n]) + 1, max_day)
            num_anchor_points = int((day_range - 2) / 7) + 1
            d_EOW = num_anchor_points*7
            vals = spec[par_name]['val']
            P_style = {f'{par_name}_0': {'type': 'logitnormal',
                                         'val' : [vals[0], vals[1]]},
                       f'{par_name}_sigma': {'type': 'lognormal',
                                         'val' : [vals[2], vals[3]]}}
            for w in range(num_anchor_points):
                P_style[f'{par_name}_step_{w+1}'] = {'type': 'normal',
                                                     'val' : [0., vals[4]]}

            lpdf += compute_lpdf(P_style, P_m)
                
            # Brownian bridge pdf
            V_BB_tf = np.array([[(7. - max(i,j)) * min(i,j) / 7. for j in (1. + np.arange(6))] for i in (1. + np.arange(6))])
                    
            for w in range(num_anchor_points - 1): # Number of weeks-1.
                week_params_demeaned = logit(P_aux[[f'{par_name}_{d}' for d in w * 7 + np.arange(1,7).astype('int64')]].values) - logit(P_aux[f'{par_name}_{w*7}'].values[:,None]) - np.arange(1,7)[None,:] / 7. * (P_m[f'{par_name}_step_{(w+1)}'].values[:,None])
                week_params_std = week_params_demeaned / P_m[f'{par_name}_sigma'].values[:,None]
                
                raw = P_aux[[f'{par_name}_{d}' for d in w * 7 + np.arange(1,7).astype('int64')]].values

                lpdf += scipy.stats.multivariate_normal.logpdf(week_params_std, cov = V_BB_tf) - 6. * np.log(P_m[f'{par_name}_sigma'].values) - np.sum(np.log(raw), axis = 1) - np.sum(np.log(1. - raw), axis = 1)
                
            num_remaining = day_range - 1 - 7*(num_anchor_points-1)
            if num_remaining == 7:
                num_remaining = 6
            #print(num_remaining)
            if num_remaining > 0:
                w = num_anchor_points - 1

                week_params_demeaned = P_aux[[f'{par_name}_{d}' for d in w * 7 + np.arange(1,num_remaining+1).astype('int64')]].values - P_aux[f'{par_name}_{w*7}'].values[:,None] - np.arange(1,num_remaining+1)[None,:] / 7. * (P_m[f'{par_name}_step_{(w+1)}'].values[:,None])
                week_params_std = week_params_demeaned / P_m[f'{par_name}_sigma'].values[:,None]
                
                raw = P_aux[[f'{par_name}_{d}' for d in w * 7 + np.arange(1,num_remaining+1).astype('int64')]].values
              
                lpdf += scipy.stats.multivariate_normal.logpdf(week_params_std, cov = V_BB_tf[:num_remaining, :num_remaining]) - float(num_remaining) * np.log(P_m[f'{par_name}_sigma'].values) - np.sum(np.log(raw), axis = 1) - np.sum(np.log(1. - raw), axis = 1)
 
        elif par_type in ['OU', 'logOU', 'min_logOU', 'logitOU']:
            day_range = min(np.max(s_dat.session[:s_dat.T[n],n]) + 1, max_day)
            
            vals = spec[par_name]['val']
            # mu, rate, sigma:
            P_style = {f'{par_name}_mu': {'type': 'normal',
                                         'val' : [vals[0], vals[1]]},
                       f'{par_name}_rate': {'type': 'lognormal',
                                         'val' : [vals[2], vals[3]]},
                       f'{par_name}_stat_std': {'type': 'lognormal',
                                         'val' : [vals[4], vals[5]]}}  
            lpdf += compute_lpdf(P_style, P_m)
            
            session_times = s_dat.session_times[n]
            time_diffs = session_times[1:] - session_times[:-1]
            
            # The stationary variance:
            stat_std = P_m[f'{par_name}_stat_std'].values
            
            # The time decay:
            ratetime_decay = np.exp(- time_diffs[:, None] * P_m[f'{par_name}_rate'].values[None, :])            
           
            # trajectory:
            P_traj = P_aux[[f'{par_name}_{d}' for d in range(day_range)]].copy()
            if par_type == 'logOU':
                P_traj = np.log(P_traj)
            elif par_type == 'min_logOU':
                P_traj = np.log(-P_traj)
            elif par_type == 'logitOU':
                P_traj = logit(P_traj)
            
            lpdf += scipy.stats.norm.logpdf((P_traj[f'{par_name}_0'] - P_m[f'{par_name}_mu']) / stat_std) - np.log(stat_std) # correct for scaling
            
            for d in range(day_range - 1):                 
                lpdf += scipy.stats.norm.logpdf(P_traj[f'{par_name}_{d+1}'].values, 
                                                loc = P_traj[f'{par_name}_{d}'].values * ratetime_decay[d,:] + P_m[f'{par_name}_mu'].values * (1. - ratetime_decay[d,:]),
                                                scale = stat_std * np.sqrt(1. - np.square(ratetime_decay[d,:])))
                                                
        elif par_type in ['OU_fixed_mean', 'logOU_fixed_mean', 'min_logOU_fixed_mean', 'logitOU_fixed_mean']:
            day_range = min(np.max(s_dat.session[:s_dat.T[n],n]) + 1, max_day)
            
            vals = spec[par_name]['val']
            mean = spec[par_name]['fixed_val']
            # mu, rate, sigma:
            P_style = {f'{par_name}_mu': {'type': 'fixed',
                                         'val' : [],
                                         'fixed_val': mean},
                       f'{par_name}_rate': {'type': 'lognormal',
                                         'val' : [vals[0], vals[1]]},
                       f'{par_name}_stat_std': {'type': 'lognormal',
                                         'val' : [vals[2], vals[3]]}}  
            lpdf += compute_lpdf(P_style, P_m)
            
            session_times = s_dat.session_times[n]
            time_diffs = session_times[1:] - session_times[:-1]
            
            # The stationary variance:
            stat_std = P_m[f'{par_name}_stat_std'].values
            
            # The time decay:
            ratetime_decay = np.exp(- time_diffs[:, None] * P_m[f'{par_name}_rate'].values[None, :])            
           
            # trajectory:
            P_traj = P_aux[[f'{par_name}_{d}' for d in range(day_range)]].copy()
            if par_type == 'logOU_fixed_mean':
                P_traj = np.log(P_traj)
            elif par_type == 'min_logOU_fixed_mean':
                P_traj = np.log(-P_traj)
            elif par_type == 'logitOU_fixed_mean':
                P_traj = logit(P_traj)
            
            lpdf += scipy.stats.norm.logpdf((P_traj[f'{par_name}_0'] - P_m[f'{par_name}_mu']) / stat_std) - np.log(stat_std) # correct for scaling
            
            for d in range(day_range - 1): 
                if d == day_range - 2:
                    lpdf_old = lpdf.copy()
                lpdf += scipy.stats.norm.logpdf(P_traj[f'{par_name}_{d+1}'].values, 
                                                loc = P_traj[f'{par_name}_{d}'].values * ratetime_decay[d,:] + P_m[f'{par_name}_mu'].values * (1. - ratetime_decay[d,:]),
                                                scale = stat_std * np.sqrt(1. - np.square(ratetime_decay[d,:])))
                # if d == day_range - 2:
                    # print(f"\n{d+1}, {par_name}:")
                    # large = P_traj[f'{par_name}_{d+1}'].abs() > 5
                    # print(f"num large: {np.sum(large)}")
                    # print((lpdf - lpdf_old)[large])
                    # print(lpdf[large])
                    # print(P_traj[f'{par_name}_{d+1}'][large])
                    # print(P_aux[f'{par_name}_{d+1}'][large])
                    # print(P_traj[f'{par_name}_{d}'][large])
                    
              
    
    return lpdf

    
def compute_lpdf_q_new(P_q, q_spec, m_spec, s_dat, n, max_day = 1000, min_day = 1, q_type = 'robust'):
    """
    min_day and max_day run from 1 (which translate to 'R_mean_{min_day - 1}' and so on.
    min_day and max_day are inclusive. if min_day <= 1, the sigmas are included, otherwise not.
    """
    lpdf = np.zeros(P_q.shape[0])
        
    for par_name in m_spec.keys():
        par_type = m_spec[par_name]['type']
        
        if par_type not in ['logBB_stat', 'min_logBB_stat', 'logitBB_stat', 'BB_stat', 'OU', 'logOU', 'min_logOU', 'logitOU', 'OU_fixed_mean', 'logOU_fixed_mean', 'min_logOU_fixed_mean', 'logitOU_fixed_mean']:
            if min_day < 1.5:
                lpdf += compute_lpdf({par_name: q_spec[par_name]}, P_q)
        elif par_type in ['logBB_stat', 'min_logBB_stat', 'logitBB_stat', 'BB_stat']:
            P_spec = {}
            if min_day < 1.5:
                P_spec[f'{par_name}_0']     = q_spec[f'{par_name}_0']
                P_spec[f'{par_name}_sigma'] = q_spec[f'{par_name}_sigma']

            day_range = min(np.max(s_dat.session[:s_dat.T[n],n]) + 1, max_day)
            num_anchor_points = int((day_range - 2) / 7) + 1

            for d in range(max(min_day-1,0), day_range):
                P_spec[f'{par_name}_{d}'] = q_spec[f'{par_name}_{d}']
            lpdf += compute_lpdf(P_spec, P_q)

            # Add these terms due to transformation of variables wrt the BB prior:
            if par_type in ['logBB_stat']:
                for w in range(num_anchor_points):
                    lpdf += np.log(P_q[f'{par_name}_{(w+1)*7}'].values)
            elif par_type in ['min_logBB_stat']:
                for w in range(num_anchor_points):
                    lpdf += np.log(-P_q[f'{par_name}_{(w+1)*7}'].values)
            elif par_type in ['logitBB_stat']:
                for w in range(num_anchor_points):
                    lpdf += np.log(P_q[f'{par_name}_{(w+1)*7}'].values) + np.log(1. - P_q[f'{par_name}_{(w+1)*7}'].values)
        elif par_type in ['OU', 'logOU', 'min_logOU', 'logitOU']:
            P_spec = {}
            if min_day < 1.5:
                P_spec[f'{par_name}_mu']    = q_spec[f'{par_name}_mu']
                P_spec[f'{par_name}_rate']  = q_spec[f'{par_name}_rate']
                P_spec[f'{par_name}_stat_std'] = q_spec[f'{par_name}_stat_std']
                
            day_range = np.max(s_dat.session[:s_dat.T[n],n]) + 1
            
            for d in range(max(min_day-1,0), day_range):
                P_spec[f'tf_{par_name}_{d}'] = q_spec[f'tf_{par_name}_{d}']
            lpdf += compute_lpdf(P_spec, P_q) 
            
        elif par_type in ['OU_fixed_mean', 'logOU_fixed_mean', 'min_logOU_fixed_mean', 'logitOU_fixed_mean']:
            P_spec = {}
            if min_day < 1.5:
                P_spec[f'{par_name}_mu']    = q_spec[f'{par_name}_mu']
                P_spec[f'{par_name}_rate']  = q_spec[f'{par_name}_rate']
                P_spec[f'{par_name}_stat_std'] = q_spec[f'{par_name}_stat_std']
                
            day_range = np.max(s_dat.session[:s_dat.T[n],n]) + 1
            
            for d in range(max(min_day-1,0), day_range):
                P_spec[f'tf_{par_name}_{d}'] = q_spec[f'tf_{par_name}_{d}']
            lpdf += compute_lpdf(P_spec, P_q) 

            # lpdf_test = compute_lpdf({f"tf_{par_name}_27" : q_spec[f'tf_{par_name}_27']}, P_q)           
            # print(f"tf_{par_name}_27")
            # print(q_spec[f'tf_{par_name}_27'])
            # print(P_q['tf_R_pun_27'])
            # print(lpdf_test[:5])
            
    return lpdf
    
        
def fit_distr(spec_old, param_names, P, weights, lweights_u = None, q_type = None, global_variation_inflation = True, max_df = 8, min_df = 4, mean_type = 'ESS', var_type = 'ESS-gradual'):
    """
    Given an old spec 'spec_old' (dict of distribution 'type' and 'val') and the parameter dataframe P,
    with weights 'weights', fits the distibution for a next twisting function.
        
    returns {'type': d_type, 'val': new_val}
    """
    spec_params = {}
    
    var_inflation = 0.
    
    d = float(len(param_names))
    
    ESS = 1. / np.sum(np.square(weights))
    if not (ESS >= 1.):
        print(f"Warning: the ESS is {ESS}.")
        print(f"First 10 weights: {weights[:10]}")
        print(f"Old spec: {spec_old}")
    ESS_new = None
    
    if var_type[:3] == 'ESS' and global_variation_inflation:
        for param_name in param_names:
            P_vals = P[param_name].values
            
            if spec_old[param_name]['type'] == 'normal' or spec_old[param_name]['type'] == 't':
                mean =  np.sum(weights * P_vals)
                var_inflation += np.square(mean - spec_old[param_name]['val'][0]) / d / ESS #/ np.square(spec_old[param_name]['val'][1])
            elif spec_old[param_name]['type'] == 'lognormal' or spec_old[param_name]['type'] == 'logt':
                P_vals = np.log(P_vals)
                mean =  np.sum(weights * P_vals)
                var_inflation += np.square(mean - spec_old[param_name]['val'][0]) / d / ESS #/ np.square(spec_old[param_name]['val'][1])
            elif spec_old[param_name]['type'] == 'min_lognormal' or spec_old[param_name]['type'] == 'min_logt':
                P_vals = np.log(-P_vals)
                mean =  np.sum(weights * P_vals)
                var_inflation += np.square(mean - spec_old[param_name]['val'][0]) / d / ESS #/ np.square(spec_old[param_name]['val'][1])
            elif spec_old[param_name]['type'] == 'logitt' or spec_old[param_name]['type'] == 'logitnormal':
                P_vals = logit(P_vals)
                mean =  np.sum(weights * P_vals)
                var_inflation += np.square(mean - spec_old[param_name]['val'][0]) / d / ESS
            elif spec_old[param_name]['type'] == 'gamma':
                1+1
            elif spec_old[param_name]['type'] == 'min_gamma':
                1+1
            else:
                sys.exit(f"fit_distr: {spec_old[param_name]['type']} ({param_name}) not a valid distribution type yet.")
    else:
        var_inflation = None
        
    gamma = np.nan
    if var_type[:7] == 'gradual':
        N_T = 1.5 * len(spec_old.keys()) # Number of dimensions * 1.5

        if ESS < N_T:
            ESS_new = 1.            
            
            lower_gamma = 1.
            upper_gamma = 1.
            gamma = 1.
            
            tm = time.time()
            while ESS_new < N_T or ESS_new > N_T * 1.5:
                if ESS_new < N_T:
                    if np.abs(gamma - upper_gamma) < 1e-8:
                        upper_gamma = 2. * upper_gamma
                        gamma = upper_gamma
                    else:
                        lower_gamma = gamma
                        gamma = (gamma + upper_gamma) / 2.
                else:
                    upper_gamma = gamma
                    gamma = (gamma + lower_gamma) / 2.
                    
                lweights_transformed =  lweights_u / gamma
                weights_transformed = np.exp(lweights_transformed - logsumexp_1D(lweights_transformed))
                ESS_new = 1. / np.sum(np.square(weights_transformed))
                
            print(f"Finding gamma = {gamma} for N_T = {N_T} took {time.time() - tm:.2f}s. ESS_new = {ESS_new}.")
        else:
            ESS_new = ESS
            weights_transformed = weights
            
        var_inflation = 0.
        for param_name in param_names:
            P_vals = P[param_name].values
            
            if spec_old[param_name]['type'] == 'normal' or spec_old[param_name]['type'] == 't':
                mean_transformed =  np.sum(weights_transformed * P_vals)
                var_inflation += np.square(mean_transformed - spec_old[param_name]['val'][0]) / d / ESS_new #/ np.square(spec_old[param_name]['val'][1])
            elif spec_old[param_name]['type'] == 'lognormal' or spec_old[param_name]['type'] == 'logt':
                P_vals = np.log(P_vals)
                mean_transformed =  np.sum(weights_transformed * P_vals)
                var_inflation += np.square(mean_transformed - spec_old[param_name]['val'][0]) / d / ESS_new #/ np.square(spec_old[param_name]['val'][1])
            elif spec_old[param_name]['type'] == 'min_lognormal' or spec_old[param_name]['type'] == 'min_logt':
                P_vals = np.log(-P_vals)
                mean_transformed =  np.sum(weights_transformed * P_vals)
                var_inflation += np.square(mean_transformed - spec_old[param_name]['val'][0]) / d / ESS_new #/ np.square(spec_old[param_name]['val'][1])
            elif spec_old[param_name]['type'] == 'logitt' or spec_old[param_name]['type'] == 'logitnormal':
                P_vals = logit(P_vals)
                mean_transformed =  np.sum(weights_transformed * P_vals)
                var_inflation += np.square(mean_transformed - spec_old[param_name]['val'][0]) / d / ESS_new
            elif spec_old[param_name]['type'] == 'gamma':
                1+1
            elif spec_old[param_name]['type'] == 'min_gamma':
                1+1
            else:
                sys.exit(f"fit_distr: {spec_old[param_name]['type']} ({param_name}) not a valid distribution type yet.")     
    else:
        ESS_new = None
        weights_transformed = None
        
    for param_name in param_names:
        P_vals = P[param_name].values

        if spec_old[param_name]['type'] == 'normal':

            smoothed_mean = fit_mean(param_name, weights, P_vals, spec_old, ESS, mean_type = mean_type)
            
            std = fit_var(param_name, weights, P_vals, spec_old, ESS, var_type = var_type, var_inflation = var_inflation, weights_transformed = weights_transformed, ESS_new = ESS_new)
            
            spec_params[param_name] = {'type': spec_old[param_name]['type'],
                                        'val': [smoothed_mean, std]}

        elif spec_old[param_name]['type'] == 'lognormal':

            P_vals = np.log(P_vals)

            smoothed_mean = fit_mean(param_name, weights, P_vals, spec_old, ESS, mean_type = mean_type)
            
            std = fit_var(param_name, weights, P_vals, spec_old, ESS, var_type = var_type, var_inflation = var_inflation, weights_transformed = weights_transformed, ESS_new = ESS_new)

            spec_params[param_name] = {'type': spec_old[param_name]['type'],
                                        'val': [smoothed_mean, std]}

        elif spec_old[param_name]['type'] == 'min_lognormal':

            P_vals = np.log(-P_vals)

            smoothed_mean = fit_mean(par_name, weights, P_vals, spec_old, ESS, mean_type = mean_type)
            
            std = fit_var(par_name, weights, P_vals, spec_old, ESS, var_type = var_type, var_inflation = var_inflation, weights_transformed = weights_transformed, ESS_new = ESS_new)

            spec_params[param_name] = {'type': spec_old[param_name]['type'],
                                        'val': [smoothed_mean, std]}
                                        
        elif spec_old[param_name]['type'] == 'logitnormal':

            P_vals = logit(P_vals)

            smoothed_mean = fit_mean(param_name, weights, P_vals, spec_old, ESS, mean_type = mean_type)
                
            std = fit_var(param_name, weights, P_vals, spec_old, ESS, var_type = var_type, var_inflation = var_inflation, weights_transformed = weights_transformed, ESS_new = ESS_new)
            
            spec_params[param_name] = {'type': spec_old[param_name]['type'],
                                        'val': [smoothed_mean, std]}
            
        elif spec_old[param_name]['type'] == 't':

            smoothed_mean = fit_mean(param_name, weights, P_vals, spec_old, ESS, mean_type = mean_type)
            
            std = fit_var(param_name, weights, P_vals, spec_old, ESS, var_type = var_type, var_inflation = var_inflation, weights_transformed = weights_transformed, ESS_new = ESS_new)

            deg_freedom = min(max(int(ESS / 5), min_df),max_df)
            
            spec_params[param_name] = {'type': spec_old[param_name]['type'],
                                        'val': [smoothed_mean, std, deg_freedom]}

        elif spec_old[param_name]['type'] == 'logt':

            P_vals = np.log(P_vals)

            smoothed_mean = fit_mean(param_name, weights, P_vals, spec_old, ESS, mean_type = mean_type)
                
            std = fit_var(param_name, weights, P_vals, spec_old, ESS, var_type = var_type, var_inflation = var_inflation, weights_transformed = weights_transformed, ESS_new = ESS_new)

            deg_freedom = min(max(int(ESS / 5), min_df),max_df)
            
            spec_params[param_name] = {'type': spec_old[param_name]['type'],
                                        'val': [smoothed_mean, std, deg_freedom]}

        elif spec_old[param_name]['type'] == 'min_logt':

            P_vals = np.log(-P_vals)

            smoothed_mean = fit_mean(param_name, weights, P_vals, spec_old, ESS, mean_type = mean_type)
            
            std = fit_var(param_name, weights, P_vals, spec_old, ESS, var_type = var_type, var_inflation = var_inflation, weights_transformed = weights_transformed, ESS_new = ESS_new)

            deg_freedom = min(max(int(ESS / 5), min_df),max_df)
            
            spec_params[param_name] = {'type': spec_old[param_name]['type'],
                                        'val': [smoothed_mean, std, deg_freedom]}
                                        
        elif spec_old[param_name]['type'] == 'logitt':

            P_vals = np.log(P_vals) - np.log(1. - P_vals)

            smoothed_mean = fit_mean(param_name, weights, P_vals, spec_old, ESS, mean_type = mean_type)
                
            std = fit_var(param_name, weights, P_vals, spec_old, ESS, var_type = var_type, var_inflation = var_inflation, weights_transformed = weights_transformed, ESS_new = ESS_new)

            deg_freedom = min(max(int(ESS / 5), min_df),max_df)
            
            spec_params[param_name] = {'type': spec_old[param_name]['type'],
                                        'val': [smoothed_mean, std, deg_freedom]}
            
        elif spec_old[param_name]['type'] == 'gamma':
            
            # Order the parameter values from small to large
            ranks = np.argsort(P_vals)
            oparams = P_vals[ranks]
            oweights = weights[ranks]
            cdf = np.cumsum(oweights)

#             if ESS > 10.:
#                 print()
#                 # 1000 equally spaced quantiles:
#                 I = 1000
#                 sampp = np.arange(1. / (2.*I), 1, 1. / I)
#                 samp = np.zeros(I)
#                 i = 0; j = 0
#                 while i < I:
#                     if cdf[j] > sampp[i]:
#                         samp[i] = oparams[j]
#                         i += 1
#                     else:
#                         j += 1

#                 loc, _, scale = scipy.stats.gamma.fit(samp, floc = 0)
#                 spec_params[param_name] = {'type': 'gamma',
#                                           'val': [loc, scale]}
#             else:
            mean = np.sum(weights * P_vals)
            print(param_name)
            print(mean)
            sd = np.std(weights * P_vals) + 5/ESS
            print(sd)
            spec_params[param_name] = {'type': 'gamma',
                                          'val': [mean / sd, sd]}
            
            weights_sorted = np.argsort(weights)[::-1]
            weights_high_idx = weights_sorted[:5]
            print(P_vals[weights_high_idx])
            
        elif spec_old[param_name]['type'] == 'min_gamma':
            
            P_vals = -P_vals
            # Order the parameter values from small to large
            ranks = np.argsort(P_vals)
            oparams = P_vals[ranks]
            oweights = weights[ranks]
            cdf = np.cumsum(oweights)

        
#             # 1000 equally spaced quantiles:
#             I = 1000
#             sampp = np.arange(1. / (2.*I), 1, 1. / I)
#             samp = np.zeros(I)
#             i = 0; j = 0
#             while i < I:
#                 if cdf[j] > sampp[i]:
#                     samp[i] = oparams[j]
#                     i += 1
#                 else:
#                     j += 1
            
#             loc, _, scale = scipy.stats.gamma.fit(samp, floc = 0)
#             spec_params[param_name] = {'type': 'min_gamma',
#                                       'val': [loc, scale]}

            mean = np.sum(weights * P_vals)
            print(param_name)
            print(mean)
            sd = np.std(weights * P_vals) + 5/ESS
            print(sd)
            spec_params[param_name] = {'type': 'min_gamma',
                                          'val': [mean / sd, sd]}

        else:
            sys.exit(f"fit_distr: {spec_old[param_name]['type']} ({param_name}) is not a valid distribution type yet.")
        
    return spec_params, gamma

def fit_mean(param_name, weights, P_vals, spec_old, ESS, mean_type = 'ESS'):
    
    mean = np.sum(weights * P_vals)
    
    if mean_type == 'ESS':
        smoothed_mean = spec_old[param_name]['val'][0] + 1. / (1. + 1. / ESS) * (mean - spec_old[param_name]['val'][0])
    elif mean_type == 'gradual':
        smoothed_mean = spec_old[param_name]['val'][0] + .33 * (mean - spec_old[param_name]['val'][0])
    elif mean_type == 'MLE':
        smoothed_mean = mean
        
    return smoothed_mean

def fit_var(param_name, weights, P_vals, spec_old, ESS, var_type = 'ESS', var_inflation = None, weights_transformed = None, ESS_new = None):
            
    if var_type[:3] == 'ESS':   
        mean = np.sum(weights * P_vals)
        
        if var_inflation is not None:
            var_add = np.sum(weights * np.square(P_vals - mean)) \
            + .5 * np.square(mean - spec_old[param_name]['val'][0]) / ESS #/ np.square(spec_old[param_name]['val'][1]) \
            + .5 * var_inflation
        else:
            var_add = np.sum(weights * np.square(P_vals - mean)) \
            + np.square(mean - spec_old[param_name]['val'][0]) / ESS  #/ np.square(spec_old[param_name]['val'][1])
    
        if var_type == 'ESS-gradual':
            std = spec_old[param_name]['val'][1] + .33 * (np.sqrt(var_add) - spec_old[param_name]['val'][1])
        elif var_type == 'ESS-ESS':
            std = spec_old[param_name]['val'][1] + 1. / (1. + 2. / ESS) * (np.sqrt(var_add) - spec_old[param_name]['val'][1])
        elif var_type == 'ESS-gradual-corrected':
            std = np.sqrt(np.square(spec_old[param_name]['val'][1]) + .33 * (var_add - np.square(spec_old[param_name]['val'][1])))
        elif var_type == 'ESS-ESS-corrected':
            std = np.sqrt(np.square(spec_old[param_name]['val'][1]) + 1. / (1. + 2. / ESS) * (var_add - np.square(spec_old[param_name]['val'][1])))
        elif var_type ==  'ESS-gradual-corrected-slower':
            std = np.sqrt(np.square(spec_old[param_name]['val'][1]) + .2 * (var_add - np.square(spec_old[param_name]['val'][1])))
    elif var_type == 'gradual-gradual':   
        mean_transformed = np.sum(weights_transformed * P_vals)
        var_estimate     = np.sum(weights_transformed * np.square(P_vals - mean_transformed))
        
        std = spec_old[param_name]['val'][1] + .33 * (np.sqrt(var_estimate) - spec_old[param_name]['val'][1])
        
    elif var_type == 'gradual-ESS':
        mean_transformed = np.sum(weights_transformed * P_vals)
        var_estimate     = np.sum(weights_transformed * np.square(P_vals - mean_transformed))
        
        std = spec_old[param_name]['val'][1] + 1. / (1. + 2./ESS_new) * (np.sqrt(var_estimate) - spec_old[param_name]['val'][1])
    
    elif var_type == 'gradual-ESS-var_inflated':
        mean_transformed = np.sum(weights_transformed * P_vals)
        var_estimate     = np.sum(weights_transformed * np.square(P_vals - mean_transformed))
                
        if var_inflation is not None:
            var_add = var_estimate \
            + .5 * np.square(mean_transformed - spec_old[param_name]['val'][0]) / ESS #/ np.square(spec_old[param_name]['val'][1]) \
            + .5 * var_inflation
        else:
            var_add = var_estimate \
            + np.square(mean_transformed - spec_old[param_name]['val'][0]) / ESS  #/ np.square(spec_old[param_name]['val'][1])
    
        std = spec_old[param_name]['val'][1] + 1. / (1. + 2./ESS) * (np.sqrt(var_estimate) - spec_old[param_name]['val'][1])
    elif var_type == 'basic-gradual':
        mean = np.sum(weights * P_vals)
        var_add = np.sum(weights * np.square(P_vals - mean))
        std = np.sqrt(np.square(spec_old[param_name]['val'][1]) + .33 * (var_add - np.square(spec_old[param_name]['val'][1])))
        
    
    return std
    

def compute_estimate(I, weights, P, P_aux = None, n = None, AIS_version = True):
    
    # Check if it's a mixture:
    if type(P) == list:
        weights = weights.flatten()
        P = pd.concat(P)
        if P_aux[0] is not None:
            P_aux = pd.concat(P_aux)
        else:
            P_aux = None
    
    fit = {'P': {}}
    
    # Resample based on weights:
    for par_name in P.columns:
        par_values = P[par_name].values
        param_fit = {}

        # Order the parameter values from small to large
        ranks = np.argsort(par_values)
        oparams = par_values[ranks]
        oweights = weights[ranks]
        
        # Mean of the posterior:
        param_fit['val'] = np.sum(oweights * oparams)
        param_fit['mean'] = np.sum(oweights * oparams)
        
        # Maximum weights:
        maxweights = np.argsort(weights)[::-1][:5]
        param_fit['max'] = par_values[maxweights]        
        
        # Credibility interval:
        cdf = np.cumsum(oweights)
        ci_thresh = np.array([.025, .5, .975])
        param_fit['ci'] = np.zeros(2) 
        param_fit['median'] = np.nan
        i = 0; j = 0
        while i < 3:
            if cdf[j] > ci_thresh[i]:
                if i == 0:
                    param_fit['ci'][0] = oparams[j]
                elif i == 1:
                    param_fit['median'] = oparams[j]
                elif i == 2:
                    param_fit['ci'][1] = oparams[j]
                i += 1
            else:
                j += 1


        # 1000 equally spaced quantiles:
        #I = m.I
        sampp = np.arange(1. / (2.*I), 1, 1. / I)
        samp = np.zeros(I)
        i = 0; j = 0
        ci_i = 0
        while i < I:
            if cdf[j] > sampp[i]:
                samp[i] = oparams[j]
                i += 1
            else:
                j += 1
        param_fit['samp'] = samp

        fit['P'][par_name] = param_fit

    if P_aux is not None:
        fit['P_aux'] = {}
        for par_name in P_aux.columns:
            par_values = P_aux[par_name].values
            param_fit = {}

            # Order the parameter values from small to large
            ranks = np.argsort(par_values)
            oparams = par_values[ranks]
            oweights = weights[ranks]

            # Mean of the posterior:
            param_fit['val'] = np.sum(oweights * oparams)
            param_fit['mean'] = np.sum(oweights * oparams)
            
            # Maximum weights:
            maxweights = np.argsort(weights)[::-1][:5]
            param_fit['max'] = par_values[maxweights]            
            
            # Credibility interval:
            cdf = np.cumsum(oweights)
            ci_thresh = np.array([.025, .5, .975])
            param_fit['ci'] = np.zeros(2) 
            param_fit['median'] = np.nan
            i = 0; j = 0
            while i < 3:
                if cdf[j] > ci_thresh[i]:
                    if i == 0:
                        param_fit['ci'][0] = oparams[j]
                    elif i == 1:
                        param_fit['median'] = oparams[j]
                    elif i == 2:
                        param_fit['ci'][1] = oparams[j]
                    i += 1
                else:
                    j += 1
            fit['P_aux'][par_name] = param_fit
    else:
        if AIS_version:
            fit['P_aux'] = {'dummy': {'val': 0., 'ci':[-1.,1.]}}
        
    # fit['ESS'] = 1. / np.sum(np.square(weights))
    # fit['evidence'] = np.sum(weights) - np.log(weights.shape[0])
            
    return fit
    
def resample_systematic(weights, num_resamples):
    ancestries = np.zeros(num_resamples).astype('int32')

    cweights = np.cumsum(weights) * num_resamples
    U = np.random.uniform(0,1) 

    i = 0
    a = 0
    for a in range(num_resamples):
        while cweights[i] < U:
            i += 1

        U += 1. 
        ancestries[a] = i
        
    return ancestries
    
def repfit_summary_fixed_data(models, show_figs = True, subject_names = None, save_folder = None, use_AIS_fits = False, with_true = False, aux_params = [], stat_params = [], llik_type = None):
    """
    Assumes all models have the same llik_style.
    """
    
    if use_AIS_fits:
        fits = [m.q_fits for m in models]
    else:
        fits = [m.fit for m in models]    
    
    if models[0].llik_style in ['advanced_template', 'simple_template']:
        llik_type = [m.llik_type for m in models]
        
        llik_idxs = []
        # Depending on llik_type: choose another llik in the outputs
        for llik_t in llik_type:
            if llik_t == 'NF':
                llik_idxs.append(2)
            elif llik_t == 'All':
                llik_idxs.append(0)
            elif llik_t == 'F':
                llik_idxs.append(1)
    
    m0 = models[0]    
    s_dat0 = m0.subjects_data
        
    subject_names, columns = s_dat0.subject_names_to_columns(subject_names, False)
    color_p = px.colors.qualitative.Plotly
    
    fig_ev = go.Figure()
    fig_ev_norm = go.Figure()
    fig_runtimes = go.Figure()
    fig_MAP_ev = go.Figure()
    fig_MAP_llik = go.Figure()
    fig_MAP_llik_norm = go.Figure()
    fig_MAP_lprior = go.Figure()

    ## Extract the Posterior Means:
    MAP = []
    for m in models:
        MAP.append(m.extract_param_values(subject_names = subject_names, only_PM = True, use_AIS_fits = use_AIS_fits))               
      
    # Compute PM lprior and llik:
    lprior_models_all = np.zeros((len(models), len(columns)))
    llik_models_all   = np.zeros((len(models), len(columns)))
    for k,m in enumerate(models):
        for l,n in enumerate(columns):
            if MAP[k][1] is not None:
                lprior_models_all[k,l] = compute_lpdf_prior_new(MAP[k][0].iloc[[l],:], MAP[k][1].iloc[[l],:], m.spec, m.subjects_data, m, n)[0]
                if m.llik_style == 'advanced_template':
                    llik_models_all[k,l]   = llik_adv(MAP[k][0].iloc[[l],:], MAP[k][1].iloc[[l],:], n, m.subjects_data, m.style, return_Q=False, ESS_it_interval=10000)[llik_idxs[k]][0] 
                elif m.llik_style == 'simple_template':
                    llik_models_all[k,l]   = llik_simple(MAP[k][0].iloc[[l],:], MAP[k][1].iloc[[l],:], n, m.subjects_data, m.style, return_Q=False)[llik_idxs[k]][0] 
                elif m.llik_style == 'custom':
                    llik_models_all[k,l]   = m.llik(MAP[k][0].iloc[[l],:], MAP[k][1].iloc[[l],:], n, m.subjects_data, return_Q=False)[0][0] 
            else:
                lprior_models_all[k,l] = compute_lpdf_prior_new(MAP[k][0].iloc[[l],:], None, m.spec, m.subjects_data, m, n)[0]
                if m.llik_style == 'advanced_template':
                    llik_models_all[k,l]   = llik_adv(MAP[k][0].iloc[[l],:], None, n, m.subjects_data, m.style, return_Q=False, ESS_it_interval=10000)[llik_idxs[k]][0]
                elif m.llik_style == 'simple_template':
                    llik_models_all[k,l]   = llik_simple(MAP[k][0].iloc[[l],:], None, n, m.subjects_data, m.style, return_Q=False)[llik_idxs[k]][0]
                elif m.llik_style == 'custom':
                    llik_models_all[k,l]   = m.llik(MAP[k][0].iloc[[l],:], None, n, m.subjects_data, return_Q=False)[0][0] 
    
    
    # Only take True of the first one:
    if with_true:
        if s_dat0.P_aux_true is not None:
            if m0.llik_style == 'advanced_template':
                llik_ref    = np.array([[llik_adv(m.subjects_data.P_true.iloc[[l],:], m.subjects_data.P_aux_true.iloc[[l],:], n,  m.subjects_data, m.style, return_Q = False, ESS_it_interval = 10000)[llik_idxs[0]][0] for k,m in enumerate(models)] for l,n in enumerate(columns)])
            elif m0.llik_style == 'simple_template':
                llik_ref    = np.array([[llik_simple(m.subjects_data.P_true.iloc[[l],:], m.subjects_data.P_aux_true.iloc[[l],:], n,  m.subjects_data, m.style, return_Q = False)[llik_idxs[0]][0] for k,m in enumerate(models)] for l,n in enumerate(columns)])
            elif m.llik_style == 'custom':
                llik_ref    = np.array([[m.llik(m.subjects_data.P_true.iloc[[l],:], m.subjects_data.P_aux_true.iloc[[l],:], n,  m.subjects_data,  return_Q = False)[0][0] for k,m in enumerate(models)] for l,n in enumerate(columns)])
            lprior_ref = np.array([[compute_lpdf_prior_new(m.subjects_data.P_true.iloc[[l],:], m.subjects_data.P_aux_true.iloc[[l],:], m.spec, m.subjects_data, m, n)[0] for k,m in enumerate(models)] for l,n in enumerate(columns)])
        else:
            if m0.llik_style == 'advanced_template':
                llik_ref    = np.array([[llik_adv(m.subjects_data.P_true.iloc[[l],:], None, n, m.subjects_data, m.style,return_Q=False, ESS_it_interval = 10000)[llik_idxs[0]][0] for k,m in enumerate(models)] for l,n in enumerate(columns)])
            elif m0.llik_style == 'simple_template':
                llik_ref    = np.array([[llik_simple(m.subjects_data.P_true.iloc[[l],:], None, n, m.subjects_data, m.style,return_Q=False)[llik_idxs[0]][0] for k,m in enumerate(models)] for l,n in enumerate(columns)])
            elif m.llik_style == 'custom':
                llik_ref    = np.array([[m.llik(m.subjects_data.P_true.iloc[[l],:], None, n,  m.subjects_data,  return_Q = False)[0][0] for k,m in enumerate(models)] for l,n in enumerate(columns)])
            lprior_ref = np.array([[compute_lpdf_prior_new(m.subjects_data.P_true.iloc[[l],:], None, m.spec, m.subjects_data, m, n)[0] for k,m in enumerate(models)] for l,n in enumerate(columns)])
               
        ref_models = [llik_ref + lprior_ref, llik_ref, lprior_ref]
    
    
    for l, n in enumerate(columns):
        # First plot the model evidences and runtimes:
        evidence_estimates = [fits[k][n]['evidence'][-1] for k,m in enumerate(models)]
        runtimes = [np.sum(m.runtime_it) for m in models]
        
        mean_ev = np.mean(evidence_estimates)
        std_ev  = np.std(evidence_estimates)
        
        fig_ev.add_trace(go.Scatter(x = np.zeros(len(evidence_estimates)) + n,
                                   y = evidence_estimates,
                                   marker_color = models[0].colors[n],
                                   name = f'{models[0].subject_names[n]}',
                                   mode = 'markers'))
        fig_ev.add_trace(go.Scatter(x = np.zeros(len(evidence_estimates)) + n + .2,
                                   y = [mean_ev],
                                   error_y=dict(
                                    type='data',
                                    symmetric=True,
                                    array= [1.96 * std_ev],
                                    thickness = .5),
                                   marker_color = models[0].colors[n],                                    
                                   name = f'{models[0].subject_names[n]}'))
        
        norm_cts = []
        for m in models:
            if m.llik_style in ['advanced_template', 'simple_template']:
                norm_cts.append(m.num_datapoints[n])
            else:
                if llik_type is None or llik_type == 'All':
                    norm_cts.append(m.num_datapoints[n])
                elif llik_type == 'NF':
                    norm_cts.append(m.num_nopractice_nf[n])
                elif llik_type == 'F':
                    norm_cts.append(m.num_nopractice_f[n])
        norm_cts = np.array(norm_cts)
        
        fig_ev_norm.add_trace(go.Scatter(x = np.zeros(len(evidence_estimates)) + n,
                                   y = np.exp(evidence_estimates / norm_cts),
                                   marker_color = models[0].colors[n],
                                   name = f'{models[0].subject_names[n]}',
                                   mode = 'markers'))
        
        fig_runtimes.add_trace(go.Scatter(x = np.zeros(len(evidence_estimates)) + n,
                                   y = runtimes,
                                   name = f'{models[0].subject_names[n]}',
                                   marker_color = models[0].colors[n],                                          
                                   mode = 'markers'))
        fig_runtimes.add_trace(go.Scatter(x = np.zeros(len(runtimes)) + n + .2,
                                   y = [np.mean(runtimes)],
                                   error_y=dict(
                                    type='data',
                                    symmetric=True,
                                    array= [1.96 * np.std(runtimes)],
                                    thickness = .5),
                                   marker_color = models[0].colors[n],                                          
                                   name = f'{models[0].subject_names[n]}'))
        
        # Posterior Mean quantities:
        lprior_models = lprior_models_all[:, l]
        llik_models   = llik_models_all[:,l]
        ltotal_models = llik_models + lprior_models
            
        p_models = [ltotal_models, llik_models, np.exp(llik_models / norm_cts), lprior_models]
        if with_true:
            ref_models = [llik_ref[l] + lprior_ref[l], llik_ref[l], np.exp(llik_ref[l] / norm_cts), lprior_ref[l]]
        
        for j, fig in enumerate([fig_MAP_ev, fig_MAP_llik, fig_MAP_llik_norm, fig_MAP_lprior]):
            if not with_true:
                fig.add_trace(go.Scatter(x = np.zeros(len(evidence_estimates)) + n,
                                           y = p_models[j],
                                           marker_color = models[0].colors[n],
                                           name = f'{models[0].subject_names[n]}',
                                           mode = 'markers'))
            else:
                for k, m in enumerate(models):
                    fig.add_trace(go.Scatter(x = [n, n + .2],
                                               y = [p_models[j][k], ref_models[j][k]],
                                               marker_color = models[0].colors[n],
                                               name = f'Model {k}',
                                               mode = 'markers'))
            fig.add_trace(go.Scatter(x = np.zeros(len(evidence_estimates)) + n + .4,
                                       y = [np.mean(p_models[j])],
                                       error_y=dict(
                                        type='data',
                                        symmetric=True,
                                        array= [1.96 * np.std(p_models[j])],
                                        thickness = .5),
                                       marker_color = models[0].colors[n],                                    
                                       name = f'{models[0].subject_names[n]}'))
            
    fig_ev.update_layout(title = 'Evidence Estimate per model',
                        xaxis_title = 'subject #',
                        yaxis_title = 'log-evidence estimate',
                        xaxis = {'range': [-.5, len(columns) - .5]})
    fig_ev_norm.update_layout(title = 'Average Evidence Estimate per no-practice trial',
                        xaxis_title = 'subject #',
                        yaxis_title = 'log-evidence estimate',
                        xaxis = {'range': [-.5, len(columns) - .5]})  
    fig_runtimes.update_layout(title = 'Runtimes per model',
                        xaxis_title = 'subject #',
                        yaxis_title = 'Runtime (s)',
                              xaxis = {'range': [-.5, len(columns) - .5]})  
    if show_figs:
        fig_ev.show()
        fig_ev_norm.show()
        fig_runtimes.show()
    
    fig_types = ['Evidence', 'llik', 'lik/trial', 'lprior']
    titles = [f'{fig_type} of the Posterior Mean estimates' for fig_type in fig_types]
    for j, fig in enumerate([fig_MAP_ev, fig_MAP_llik, fig_MAP_llik_norm, fig_MAP_lprior]):
        fig.update_layout(title = titles[j],
                        xaxis_title = 'subject #',
                        yaxis_title = fig_types[j],
                        xaxis = {'range': [-.5, len(columns) - .5]})  
        if show_figs:
            fig.show()
    
    if save_folder is not None:
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        if use_AIS_fits:
            folder_name = f'summaries-AIS'
        else:
            folder_name = f'summaries-IBIS'
        summaries_folder = os.path.join(save_folder, folder_name)
        if not os.path.exists(summaries_folder):
            os.mkdir(summaries_folder)
        fig_ev.write_image(os.path.join(summaries_folder, 'evidence.jpeg'))
        fig_ev_norm.write_image(os.path.join(summaries_folder, 'evidence_per_trial.jpeg'))        
        fig_runtimes.write_image(os.path.join(summaries_folder, 'runtimes.jpeg'))
        
        MAP_files = [os.path.join(summaries_folder, f'PM-{fig_type}.jpeg') for fig_type in ['ev', 'llik', 'lik_trial', 'lprior']]
        for j, fig in enumerate([fig_MAP_ev, fig_MAP_llik, fig_MAP_llik_norm, fig_MAP_lprior]):
            fig.write_image(MAP_files[j])
        
    ## MAP estimates
    if with_true:
        P_true = s_dat0.P_true
        P_aux_true = s_dat0.P_aux_true
    else:
        P_true = None
        P_aux_true = None
    
    # Make figures with all the MAP trajectories
    fits_models = []
    if use_AIS_fits:
        for m in models:
            fits_models.append(m.extract_param_values(subject_names = subject_names, only_PM = False, use_AIS_fits = True, AIS_fit_idxs = [-1] * len(subject_names)))
    else:
        for m in models:
            fits_models.append(m.extract_param_values(subject_names = subject_names, only_PM = False, use_AIS_fits = False))
    if use_AIS_fits:
        ests_name = f'stability-AIS'
    else:
        ests_name = f'stability'
    
    plot_multiple_model_ests_time_varying(ests_name, s_dat0, fits_models = fits_models, P_ref = P_aux_true, par_names = aux_params, save_folder = save_folder, show_figs = show_figs, subject_names = subject_names, with_evidence = True)  
    plot_multiple_model_ests(ests_name, s_dat0, fits_models = fits_models, subject_names = subject_names, save_folder = save_folder, P_ref = P_true, show_figs = show_figs, par_names = stat_params)    

def repfit_summary_repdata(m0, s_dat, model_folder = '', num_models_per_dataset = 1, num_datasets = 10, sim_data_folder = '', show_figs = True, subject_names = None, save_folder = None, with_true = True, use_AIS_fits = False, data_base_name = 'Simulated-PM', aux_params = ['R_mean', 'R_asym'], stat_params = ['eps'], llik_type = 'All'):
    subject_names, columns = s_dat.subject_names_to_columns(subject_names, False)
    color_p = px.colors.qualitative.Plotly
    
    # Not needed actually if models exist:
    if m0.llik_style == 'advanced_template':
        style_adv = m0.get_style()
        style_simple = None
        llik_custom = None
    elif m0.llik_style == 'simple_template':
        style_adv = None
        style_simple = m0.get_style()
        llik_custom = None
    elif m0.llik_style == 'custom':
        style_adv = None
        style_simple = None
        llik_custom = m0.llik
      
    ## 1) Load datsets and models:
    s_dats = []    
    models = []
    for k in range(num_datasets):
        s_dats.append(load_PM_dataset(sim_data_folder, s_dat, dataset_idx = k, base_name = data_base_name))
        models_k, _ = check_and_load_repdat(s_dats[k], dataset_idx = k, spec = m0.spec, num_if_new = num_models_per_dataset, model_folder = model_folder, style_adv = style_adv, style_simple = style_simple, llik_custom = llik_custom)
        
        bic = [m.bic[-1] for m in models_k]

        idx_max = np.argmax(bic)
        models.append(models_k[idx_max])   
        
    repfit_summary_fixed_data(models,
                              show_figs = show_figs, 
                              save_folder = save_folder, 
                              use_AIS_fits = use_AIS_fits,
                              with_true = with_true,
                              aux_params = aux_params,
                              stat_params = stat_params,
                              llik_type = llik_type)
                              
def plot_evidence(models, save_folder = None, show_figs = True, subject_names = None, fnf_strat = False, model_names = None, start_it = 0):
    
    if model_names is None:
        model_names = [f'{k}' for k in range(len(models))]
    
    s_dat = models[0].subjects_data
    subject_names, columns = s_dat.subject_names_to_columns(subject_names, False)
    
    colors = [px.colors.qualitative.Plotly[k % 10] for k in range(len(models))]
    
    if save_folder is not None:
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)   
        summaries_folder = os.path.join(save_folder, 'fit_summaries')
        if not os.path.exists(summaries_folder):
            os.mkdir(summaries_folder)
    
    fig_ev = go.Figure()
    for k, m in enumerate(models):        
        fig_ev.add_trace(go.Scatter(x = np.arange(start_it, len(m.evidence)),
                                   y = m.evidence[start_it:],
                                   mode = 'markers+lines',
                                   name = f'Evidence {model_names[k]}',
                                   marker_color = colors[k]))
        if fnf_strat:
            fig_ev.add_trace(go.Scatter(x = np.arange(start_it, len(m.evidence_f)),
                                   y = m.evidence_f[start_it:],
                                   mode = 'markers+lines',
                                   name = f'Evidence_f {model_names[k]}',
                                   marker_color = colors[k]))
            fig_ev.add_trace(go.Scatter(x = np.arange(start_it, len(m.evidence_nf)),
                                   y = m.evidence_nf[start_it:],
                                   mode = 'markers+lines',
                                   name = f'Evidence_nf {model_names[k]}',
                                   marker_color = colors[k]))

    fig_ev.update_layout(title = 'Evidence estimate vs iteration #',
                        xaxis_title = 'Iteration #',
                        yaxis_title = 'Evidence Estimate (log-scale)',
                        yaxis = {'tickformat' : ".0f"})
    
    fig_bic = go.Figure()
    for k, m in enumerate(models):        
        fig_bic.add_trace(go.Scatter(x = np.arange(start_it, len(m.bic)),
                                   y = m.bic[start_it:],
                                   mode = 'markers+lines',
                                   name = f'bic {model_names[k]}',
                                   marker_color = colors[k]))

    fig_bic.update_layout(title = 'iBIC estimate vs iteration #',
                        xaxis_title = 'Iteration #',
                        yaxis_title = 'iBic Estimate (log-scale)',
                        yaxis = {'tickformat' : ".0f"})
        
    if show_figs:
        fig_ev.show()
        fig_bic.show()
        
    if save_folder is not None:        
            fig_ev.write_image(os.path.join(summaries_folder, f'Fit_Model_Evidence.jpeg'))   
            fig_bic.write_image(os.path.join(summaries_folder, f'Fit_Model_iBIC.jpeg'))  
    
    for k, m in enumerate(models):
        fig_subjects = go.Figure()
        for n in columns:
            ev_n = m.fit[n]['evidence']

            fig_subjects.add_trace(go.Scatter(x = np.arange(start_it, len(ev_n)),
                                               y = ev_n[start_it:],
                                               mode = 'markers+lines',
                                               name = m.subject_names[n],
                                             marker_color = m.colors[n]))

        fig_subjects.update_layout(title = f'{model_names[k]}) Subject evidence estimates vs iteration #',
                            xaxis_title = 'Iteration #',
                            yaxis_title = 'Evidence Estimate (log-scale)')

        if show_figs:
            fig_subjects.show()

        if save_folder is not None:
            fig_subjects.write_image(os.path.join(summaries_folder, f'Fit_Subject_Evidence_{model_names[k]}.jpeg'))
        

def plot_true_vs_est_time_varying_one_subject(self, n, par_names = [], par_T = 7, save_folder = None):
    """
    For models that use day varying parameters, plots the parameter estimates as a time series vs the true underlying values.

    - par_names: list of strongs, base names of the parameters (e.g. 'R_r' for 'R_r_0' to 'R_r_28')
    - par_T: int, the number of time steps the parameter varies over, to extract par_name_0 to par_name_par_T.
    """
    s_dat = self.subjects_data

    for par_name in par_names:
        fig = go.Figure()
        fig_lin = go.Figure()

        all_min = None; all_max = None
        if par_name + '_1' in self.spec.keys():
            ests = np.array([self.fit[n]['P'][par_name + '_' + str(d)]['val'] for d in range(par_T)])
            ci = np.array([self.fit[n]['P'][par_name + '_' + str(d)]['ci'] for d in range(par_T)])
        else:
            ests = np.array([self.fit[n]['P_aux'][par_name + '_' + str(d)]['val'] for d in range(par_T)])
            ci = np.array([self.fit[n]['P_aux'][par_name + '_' + str(d)]['ci'] for d in range(par_T)])

        xticks = np.arange(par_T)

        fig.add_trace(go.Scatter(
                x=xticks,
                y=ests,
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array= ci[:, 1] - ests,
                    arrayminus= ests - ci[:,0],
                    thickness = 1.),
                mode = 'markers+lines',
                name = str(self.subject_names[n]),
                line = {'color': self.colors[n]}
                ))

        true_vals = s_dat.P_aux_true[[par_name + '_' + str(d) for d in range(par_T)]].iloc[n,:].values

        fig.add_trace(go.Scatter(
            x = xticks,
            y = true_vals,
            mode = 'markers+lines',
            name = f"{self.subject_names[n]} (true)",
            line = {'color': self.colors[n], 'dash': 'dash'}
        ))

        fig_lin.add_trace(go.Scatter(x = true_vals,
                                    y = ests,
                                    error_y = dict(
                                                type='data',
                                                symmetric=False,
                                                array= ci[:, 1] - ests,
                                                arrayminus= ests - ci[:,0],
                                                thickness = 1.),
                                    mode = 'markers',
                                    name = str(self.subject_names[n]),
                                    line = {'color': self.colors[n]}))

        # Create the first diagonal, x = y, to display on the figure:
        minx = np.min(ci); maxx = np.max(ci)
        cover_x = np.linspace(minx, maxx)

        # min/max over all subjects:
        if all_min is not None:
            all_min = min(all_min, minx)
            all_max = max(all_max, maxx)
        else:
            all_min = minx
            all_max = maxx

        # Create a Linear Regression line
        lm = LinearRegression()
        lm.fit(true_vals.reshape(-1,1), ests)
        cover_y = lm.predict(cover_x.reshape(-1, 1)) 

        fig_lin.add_trace(go.Scatter(x = cover_x,
                                    y = cover_y,
                                    mode = 'lines',
                                    line = {'color': self.colors[n]}))


        fig.update_layout(
                title = self.name +": "+par_name,
                xaxis_title = f'd: {par_name}_d',
                yaxis_title = 'Estimates')
        fig.show()

        fig_lin.add_trace(go.Scatter(x = np.linspace(all_min, all_max),
                                    y = np.linspace(all_min, all_max),
                                    line = dict(color='firebrick', dash='dash')))
        fig_lin.update_layout(title = f"{self.name}: {par_name}",
                             xaxis_title = 'True Values',
                             yaxis_title = 'Estimates')
        fig_lin.show()

        if save_folder is not None:
            results_folder = os.path.join(save_folder, self.name)
            if not os.path.exists(results_folder):
                os.mkdir(results_folder)
            fig.write_image(os.path.join(results_folder, f'true_vs_est_{par_name}_{n}_series.jpeg'))
            fig_lin.write_image(os.path.join(results_folder, f'true_vs_est_{par_name}_{n}_lin.jpeg'))

def plot_true_vs_est_one_subject(self, n, save_folder = None):
    """
    Displays subject parameter estimates vs their true value that was used to generate their trajectories, 
    in a scatter plot with 95% credibility intervals.
    """

    spec = self.spec
    s_dat = self.subjects_data
    if s_dat.P_true is None:
        sys.exit("No true parameters found!")
    P_true = s_dat.P_true

    par_names = spec.keys()
    true_values = [P_true[par_name].iloc[n] for par_name in par_names]
    ests = np.array([self.fit[n]['P'][par_name]['val'] for par_name in par_names])
    ci = np.array([self.fit[n]['P'][par_name]['ci'] for par_name in par_names])

    # Create the first diagonal, x = y, to display on the figure:
    minx = np.min(ci); maxx = np.max(ci)
    cover_x = np.linspace(minx, maxx)

#     # Create a Linear Regression line
#     lm = LinearRegression()
#     lm.fit(true_values.reshape(-1,1), ests)
#     cover_y = lm.predict(cover_x.reshape(-1, 1))     

    fig = go.Figure()
    for k, par_name in enumerate(spec.keys()):
        fig.add_trace(go.Scatter(
                x=[true_values[k]],
                y=[ests[k]],
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array= [ci[k, 1] - ests[k]],
                    arrayminus= [ests[k] - ci[k,0]],
                    thickness = 1.),
                mode = 'markers',
                name = par_name
                ))
    fig.add_trace(go.Scatter(
        x = cover_x,
        y = cover_x,
        mode = 'lines',
        line = dict(color='blue'),
        name = 'x = y'
    ))
#         fig.add_trace(go.Scatter(
#             x = cover_x,
#             y = cover_y,
#             mode = 'lines',
#             line = dict(color='blue'),
#             name = 'Linear Regression'
#         ))
    fig.update_layout(
            title = f"{self.name}, {n}) Consistency",
        xaxis_title = 'True Values',
        yaxis_title = 'Estimates')

    fig.show()

    if save_folder is not None:
        results_folder = os.path.join(save_folder, self.name)
        if not os.path.exists(results_folder):
            os.mkdir(results_folder)
        fig.write_image(os.path.join(results_folder, f"consistency_{n}.jpeg"))

def plot_multiple_model_ests_time_varying_one_subject(model_name, n, s_dat, P_aux_n = [], P_aux_ref = None, par_names = [], par_T = 7, save_folder = None, names = None, colors = None, show_figs = False):
    """
    For models that use day varying parameters, plots the parameter estimates as a time series vs the true underlying values.

    - par_names: list of strongs, base names of the parameters (e.g. 'R_r' for 'R_r_0' to 'R_r_28')
    - par_T: int, the number of time steps the parameter varies over, to extract par_name_0 to par_name_par_T.
    - P_aux_n: list of dictionnaries {par_name : {'val': val, 'ci': ci}}
    - s_dat: subjects_data object that the models were fit to.
    - model_name: string, name of the model
    - P_aux_ref: a Pandas DataFrame of reference trajectories. Usually the true underlying parameter values.

    """

    num_models = len(P_aux_n)
    
    if names is None:
        names = [f"model {k}" for k in range(num_models)]
        
    if colors is None:
        colors = [px.colors.qualitative.Plotly[k%10] for k in range(num_models)]
    
    for par_name in par_names:
        fig = go.Figure()

        for m, P_aux in enumerate(P_aux_n):
            xticks = np.arange(par_T)

            ests = np.array([P_aux[par_name + '_' + str(d)]['val'] for d in range(par_T)])
            ci = np.array([P_aux[par_name + '_' + str(d)]['ci'] for d in range(par_T)])

            fig.add_trace(go.Scatter(
                    x=xticks,
                    y=ests,
                    mode = 'markers+lines',
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array= ci[:, 1] - ests,
                        arrayminus= ests - ci[:,0],
                        thickness = .5),
                    name = names[m],
                    line = {'color': colors[m]}
                    ))
        if P_aux_ref is not None:
            ests = P_aux_ref[[par_name + '_' + str(d) for d in range(par_T)]].values
            fig.add_trace(go.Scatter(
                    x=xticks,
                    y=ests,
                    mode = 'markers+lines',
                    name = 'True',
                    line = {'dash': 'dash'}
                    ))

        fig.update_layout(
                title = f"{s_dat.subject_names[n]}: {par_name}",
                xaxis_title = f'd: {par_name}_d',
                yaxis_title = 'Estimates')
        if show_figs:
            fig.show()

        if save_folder is not None:
            results_folder = os.path.join(save_folder, model_name)
            if not os.path.exists(results_folder):
                os.mkdir(results_folder)
            fig.write_image(os.path.join(results_folder, f'{s_dat.subject_names[n]}_{par_name}.jpeg'))

def plot_weighted_model_ests_time_varying_one_subject(n, s_dat, weights = [], P_aux = [], P_aux_ref = None, ordered = True, par_names = [], par_T = 7, save_folder = None, show_figs = False):
    """
    For models that use day varying parameters, plots the parameter estimates as a time series vs the true underlying values.

    - par_names: list of strongs, base names of the parameters (e.g. 'R_r' for 'R_r_0' to 'R_r_28')
    - par_T: int, the number of time steps the parameter varies over, to extract par_name_0 to par_name_par_T.
    - P_aux: DataFrame
    - s_dat: subjects_data object that the models were fit to.
    - P_aux_ref: a Pandas DataFrame of reference trajectories. Usually the true underlying parameter values.

    """

    S = weights.shape[0]
    if ordered:
        weights_high_idx = np.argsort(weights)[::-1][:7]
    else:
        weights_high_idx = np.random.choice(np.arange(S), size = (10))
    P_aux_high   = P_aux.iloc[weights_high_idx]
    weights_high = weights[weights_high_idx]   
    
    for par_name in par_names:
        fig = go.Figure()

        for m in range(P_aux_high.shape[0]):
            xticks = np.arange(par_T)
            
            ests = P_aux_high.iloc[m][[par_name + '_' + str(d) for d in range(par_T)]].values

            fig.add_trace(go.Scatter(
                    x=xticks,
                    y=ests,
                    mode = 'markers+lines',
                    name = f'{weights_high[m]:.3f}'
                    ))
        if P_aux_ref is not None:
            ests = P_aux_ref[[par_name + '_' + str(d) for d in range(par_T)]].values[0,:]
            if P_aux_ref.shape[0] > 1:
                ci = P_aux_ref[[par_name + '_' + str(d) for d in range(par_T)]].values[1,:]
                fig.add_trace(go.Scatter(
                    x=xticks,
                    y=ests,
                    error_y=dict(
                        type='data',
                        symmetric=True,
                        array= ci,
                        thickness = .5),
                    mode = 'markers+lines',
                    name = 'True',
                    line = {'dash': 'dash', 'color': 'orange'}
                    ))
            else:
                fig.add_trace(go.Scatter(
                        x=xticks,
                        y=ests,
                        mode = 'markers+lines',
                        name = 'True',
                        line = {'dash': 'dash', 'color': 'orange'}
                        ))

        fig.update_layout(
                title = f"{s_dat.subject_names[n]}: {par_name}",
                xaxis_title = f'd: {par_name}_d',
                yaxis_title = 'Estimates')
        if show_figs:
            fig.show()

        if save_folder is not None:
            results_folder = os.path.join(save_folder, model_name)
            if not os.path.exists(results_folder):
                os.mkdir(results_folder)
            fig.write_image(os.path.join(results_folder, f'{s_dat.subject_names[n]}_{par_name}_samples.jpeg'))

def plot_multiple_model_ests_one_subject(model_name, n, s_dat, P_n_models = [], P_aux_ref = None, save_folder = None, names = None, colors = None, show_figs = False):
    """
    For models that use day varying parameters, plots the parameter estimates as a time series vs the true underlying values.

    - par_names: list of strongs, base names of the parameters (e.g. 'R_r' for 'R_r_0' to 'R_r_28')
    - par_T: int, the number of time steps the parameter varies over, to extract par_name_0 to par_name_par_T.
    - P_aux_n: list of dictionnaries {par_name : {'val': val, 'ci': ci}}
    - s_dat: subjects_data object that the models were fit to.
    - model_name: string, name of the model
    - P_aux_ref: a Pandas DataFrame of reference trajectories. Usually the true underlying parameter values.

    """

    num_models = len(P_n_models)
    par_names = [par_name for par_name in P_n[0].keys()]
    
    if names is None:
        names = [f"model {k}" for k in range(num_models)]
        
    if colors is None:
        colors = [px.colors.qualitative.Plotly[k%10] for k in range(num_models)]
    
    for par_name in par_names:
        fig = go.Figure()
        xticks = np.arange(num_models)

        for m, P in enumerate(P_n_models):

            ests = np.array([P[par_name]['val']])
            ci = np.array([P[par_name]['ci']])

            fig.add_trace(go.Scatter(
                    x=[xticks[m]],
                    y=ests,
                    mode = 'markers',
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array= ci[:, 1] - ests,
                        arrayminus= ests - ci[:,0],
                        thickness = .5),
                    name = names[m],
                    line = {'color': colors[m]}
                    ))
#         if P_aux_ref is not None:
#             ests = P_aux[[par_name + '_' + str(d) for d in range(par_T)]].values
#             fig.add_trace(go.Scatter(
#                     x=xticks,
#                     y=ests,
#                     mode = 'markers+lines',
#                     name = 'True',
#                     line = {'dash': 'dash'}
#                     ))

        fig.update_layout(
                title = f"{s_dat.subject_names[n]}: {par_name}",
                xaxis_title = f'd: {par_name}_d',
                yaxis_title = 'Estimates')
        if show_figs:
            fig.show()

        if save_folder is not None:
            results_folder = os.path.join(save_folder, model_name)
            if not os.path.exists(results_folder):
                os.mkdir(results_folder)
            fig.write_image(os.path.join(results_folder, f'{s_dat.subject_names[n]}_{par_name}.jpeg'))
            
            
def convert_q_spec(m, q_spec_old, q_type = 'joint-robust'):
    """
    Converts the q_spec from the independent format to the joint format.
    """
    
    num_mixtures = len(q_spec_old)
    q_spec = [{'var_list' : [], 'ct_vars': [], 'fit_vars': [], 'tf': [], 'mean': [], 'covar': [], 'df': np.nan} for mx in range(num_mixtures)]
    
    for mx in range(num_mixtures):     
        for par_name in q_spec_old.keys():
            q_spec[mx]['var_list'].append(par_name)
            q_spec[mx]['mean'].append(q_spec_old[par_name]['val'][0])
            q_spec[mx]['covar'].append(np.square(q_spec_old[par_name]['val'][1]))
            
            if q_spec_old[par_name]['type'] == 'normal':
                q_spec[mx]['tf'].append('None')
            elif q_spec_old[par_name]['type'] == 'lognormal':
                q_spec[mx]['tf'].append('exp')
            elif q_spec_old[par_name]['type'] == 'min_lognormal':
                q_spec[mx]['tf'].append('min_exp')
            elif q_spec_old[par_name]['type'] == 't':
                q_spec[mx]['tf'].append('None')
            elif q_spec_old[par_name]['type'] == 'logt':
                q_spec[mx]['tf'].append('exp')
            elif q_spec_old[par_name]['type'] == 'min_logt':
                q_spec[mx]['tf'].append('min_exp')
            else:
                sys.exit('convert_q_spec > mistake')
                
            if par_name == 'R_mean_sigma' or par_name == 'R_asym_sigma':
                q_spec[mx]['ct_vars'].append(par_name)
            else:
                q_spec[mx]['fit_vars'].append(par_name)

        q_spec[mx]['mean'] = np.array(q_spec[mx]['mean']) 
        q_spec[mx]['covar'] = np.diag(q_spec[mx]['covar'])    

        if q_type == 'joint-robust':
            q_spec[mx]['df'] = 5
            
        print(q_spec)
            
    return q_spec  
    
def fit_q_mixture_new(m, spec_q_old, P, n, lweights_u = None, global_variation_inflation = True, q_type = None, max_df = 8, min_df = 4, mean_type = 'ESS', var_type = 'ESS', max_day = 1000, test_version = False):
    spec_q = [{} for mx in range(len(spec_q_old))]
    
    m_spec = m.spec
    s_dat = m.subjects_data
    
    params_to_update_jointly = []
    
    for par_name in m.spec:
        par_type = m.spec[par_name]['type']
        
        if par_type == 'fixed':
            for mx in range(len(spec_q_old)):
                spec_q[mx][par_name] = m.spec[par_name]
        elif par_type not in ['logBB_stat', 'min_logBB_stat', 'BB_stat', 'logitBB_stat', 'OU', 'logOU', 'min_logOU', 'logitOU', 'OU_fixed_mean', 'logOU_fixed_mean', 'min_logOU_fixed_mean', 'logitOU_fixed_mean']:
            params_to_update_jointly.append(par_name)
        elif par_type in ['logBB_stat', 'min_logBB_stat', 'BB_stat', 'logitBB_stat']: 
            day_range = min(np.max(s_dat.session[:s_dat.T[n],n]) + 1, max_day)
            num_anchor_points = int((day_range - 2) / 7) + 1
            d_EOW = num_anchor_points*7
    
            vals = m.spec[f'{par_name}']['val']
            for d in range(min(max_day, day_range)):
                params_to_update_jointly.append(f'{par_name}_{d}')
            
            if not test_version:
                params_to_update_jointly.append(f'{par_name}_sigma')
            
            for mx in range(len(spec_q_old)):
                
                if test_version:
                    spec_q[mx][f'{par_name}_sigma'] = {'type': 'lognormal',
                                                       'val': [vals[2],vals[3]]}
                
                for d in range(max_day, day_range):
                    spec_q[mx][f'{par_name}_{d}'] = spec_q_old[mx][f'{par_name}_{d}']
            
            if d_EOW >= day_range:
                params_to_update_jointly.append(f'{par_name}_{d_EOW}')
                
        elif par_type in ['OU', 'logOU', 'min_logOU', 'logitOU']:
            day_range = np.max(s_dat.session[:s_dat.T[n],n]) + 1
            
            vals = m.spec[f'{par_name}']['val']
            
            
            if test_version:
                for mx in range(len(spec_q_old)):
                    spec_q[mx][f'{par_name}_mu']    = {'type': 'normal',
                                                       'val': [vals[0],vals[1]]}
                    spec_q[mx][f'{par_name}_rate']  = {'type': 'lognormal',
                                                       'val': [vals[2],vals[3]]}
                    spec_q[mx][f'{par_name}_stat_std'] = {'type': 'lognormal',
                                                        'val': [vals[4],vals[5]]}
            else:
                params_to_update_jointly.append(f'{par_name}_mu')
                params_to_update_jointly.append(f'{par_name}_rate')
                params_to_update_jointly.append(f'{par_name}_stat_std')
                     
            for d in range(min(max_day, day_range)):
                params_to_update_jointly.append(f'tf_{par_name}_{d}')
        elif par_type in ['OU_fixed_mean', 'logOU_fixed_mean', 'min_logOU_fixed_mean', 'logitOU_fixed_mean']:
            day_range = np.max(s_dat.session[:s_dat.T[n],n]) + 1
            
            vals = m.spec[f'{par_name}']['val']
            mean = m.spec[f'{par_name}']['fixed_val']
            
            if test_version:
                for mx in range(len(spec_q_old)):
                    spec_q[mx][f'{par_name}_mu'] = spec_q_old[mx][f'{par_name}_mu']
                    spec_q[mx][f'{par_name}_rate']  = {'type': 'lognormal',
                                                       'val': [vals[2],vals[3]]}
                    spec_q[mx][f'{par_name}_stat_std'] = {'type': 'lognormal',
                                                        'val': [vals[4],vals[5]]}
            else:
                params_to_update_jointly.append(f'{par_name}_rate')
                params_to_update_jointly.append(f'{par_name}_stat_std')
                for mx in range(len(spec_q_old)):
                    spec_q[mx][f'{par_name}_mu'] = spec_q_old[mx][f'{par_name}_mu']
                    
            for d in range(min(max_day, day_range)):
                params_to_update_jointly.append(f'tf_{par_name}_{d}')

            
            
    gamma = np.empty(lweights_u.shape[0])
    for mx in range(lweights_u.shape[0]):
        weights_mx = np.exp(lweights_u[mx, :] - logsumexp_1D(lweights_u[mx, :]))
        new_spec, gamma_mx = fit_distr(spec_q_old[mx], params_to_update_jointly, P[mx], weights_mx, lweights_u = lweights_u[mx,:], q_type = q_type, global_variation_inflation = global_variation_inflation, max_df = max_df, min_df = min_df, mean_type = mean_type, var_type = var_type)
        spec_q[mx].update(new_spec)
        gamma[mx] = gamma_mx
            
    return spec_q, gamma
    
class subjects_EMA:
    """
    This class is used to load and store the data associated with the RL_EMA project, and to perform common useful functions.
    
    To have more rapid access of data, most data is stored in numpy arrays. This speeds up likelihood calls considerably. It makes the code more cumbersome
    than storing everything in pandas DataFrames however. A better alternative could have been storage in DataFrames, and conversion call in the likelihood 
    function.
    
    Among others, this is typical data that can be called by writing self.X, with self replaced by the object name, and X replaced by:
    - N: int, the number of subjects.
    - subject_names: list of strings, the IDs/names of the subjects.
    - colors: list of strings, indicating the colors associated with all the subjects. These can be used when making figures to keep colors consistent between figures.
    - T: list of ints, the number of trials per subject.
    - Nc: list of ints, number of stimuli per subject, taken as the maximum stimulus ID + 1.
    
    For other data, call 'create_big_subject_df'.
    
    """

    ### Initialization function ###    
    def __init__(self):
        self.min_stim = 18 # The stimulus ID at which no-practice stimuli start. The ones before will be excluded from llik computations.
        
        self.mE_processed = False # Flag for if mood events got processed
        self.sleep_processed = False
        self.activity_processed = False

    
    
    def load_one_subject_df(self, filename, verbose=True):
        """        
        This is a helper function used in the load_subjects function. It reads the behavioral data from a schedule file.

        Arguments:
        - filename: path to file

        Returns:
        - a dataframe with subject data
        """
        if verbose:
            print(f"- {filename}")

        conn = sqlite3.connect(filename, uri=True) # uri = True indicates read-only

        # Determine the list of variables to select from the database
        cursor = conn.execute('select * from stimuli')
        names = [description[0] for description in cursor.description]
        if 'reward' in names:
            variables = ['block', 'trial', 'stim1', 'stim2', 'feedback', 'choice', \
                        'outcome', 'stim_time', 'choice_time', 'feedback_time', 't1.condition AS cond1', 't1.reward AS reward1', 't1.punishment AS punishment1', \
                        't2.condition AS cond2', 't2.reward AS reward2', 't2.punishment AS punishment2'];
        elif 'probability' in names:
            variables = ['block', 'trial', 'stim1', 'stim2', 'feedback', 'choice', \
                         'outcome', 'stim_time', 'choice_time', 'feedback_time', 't1.probability AS reward1', 't1.punishment AS punishment1', \
                         't2.probability AS reward2', 't2.punishment AS punishment2'];

        # Construct the SQL query
        query = 'SELECT ' + ", ".join(variables) + ' FROM trials ' + \
                'LEFT JOIN stimuli t1 ON trials.stim1 = t1.number ' + \
                'LEFT JOIN stimuli t2 ON trials.stim2 = t2.number ' + \
                'WHERE choice_time IS NOT NULL AND stim1>=0 AND (stim2>=-1000 OR stim2 IS NULL) ' + \
                'ORDER BY choice_time ASC'
       
        
        # Read the query results into a pandas DataFrame
        df = pd.read_sql_query(query, conn)
        
        #define the probability for the accept/reject fmri trials
        df.loc[df['stim2'] == -1000, ['reward2', 'punishment2']] = 0.4
                
        # Rows where choice_time=0
        rows_to_exclude = df[df['choice_time'] == 0].index        
        # Rows where choice_time=0 so stimuli were not really learned
        NF_to_exclude = df[(df['choice_time'] == 0) & (df['feedback'] == 1)].index        
        # NF choices between stimuli that were not actually learned
        NF_rows = df[(df['feedback'] == 0) & ((df['stim1'].isin(df.loc[NF_to_exclude, 'stim1'])) | (df['stim2'].isin(df.loc[NF_to_exclude, 'stim2'])) )].index
        # Omitting the rows
        df = df[~((df['choice_time'] == 0) | (df.index.isin(NF_rows)))]
        # Reset index without creating a new column for the old index
        df.reset_index(drop=True, inplace=True)
         
        # Close the database connection
        conn.close()

        del conn

        return df
    
                    
        
    ### Helper function for loading data: ###
    def load_one_subject_df_original(self, filename, verbose=True):
        """        
        This is a helper function used in the load_subjects function. It reads the behavioral data from a schedule file.
        
        Arguments:
        - filename: path to file
        
        Returns:
        - a dataframe with subject data
        """
        if verbose:
            print(f"- {filename}")
        
        conn = sqlite3.connect(filename, uri = True) # uri = True indicates read-only
        
        # There are two data outlines for the stimuli table - older subjects vs newer. First we check which one:
        cursor = conn.execute('select * from stimuli')
        names = [description[0] for description in cursor.description]
        if 'reward' in names:
            variables = ['block', 'trial', 'stim1', 'stim2', 'feedback', 'choice', \
                        'outcome', 'stim_time', 'choice_time', 'feedback_time', 't1.condition AS cond1', 't1.reward AS reward1', 't1.punishment AS punishment1', \
                        't2.condition AS cond2', 't2.reward AS reward2', 't2.punishment AS punishment2'];
        elif 'probability' in names:
            variables = ['block', 'trial', 'stim1', 'stim2', 'feedback', 'choice', \
                         'outcome', 'stim_time', 'choice_time', 'feedback_time', 't1.probability AS reward1', 't1.punishment AS punishment1', \
                         't2.probability AS reward2', 't2.punishment AS punishment2'];
            
    
        query = 'SELECT ' + ", ".join(variables) + ' FROM trials ' + \
                'JOIN stimuli t1 ON trials.stim1 = t1.number ' + \
                'JOIN stimuli t2 ON trials.stim2 = t2.number ' + \
                'WHERE choice_time IS NOT NULL AND stim1>=0 AND stim2>=-1000 ' + \
                'ORDER BY choice_time ASC'

        df = pd.read_sql_query(query, conn)
        
        conn.close()
                
        del conn
        
        return df
    
    ### Loading functions: ###    
    def load_subjects(self, folder = '', verbose = True, with_moods = True, mood_figures = False, start_block = 6, subject_blocks = {}, MZ_PCA_load_folder = None, smoothing_window = 0.25):
        """
        Reads in the subject data from the _schedule.db files in the specified 'folder'. Returns the `subjects_EMA` object itself, with the data loaded.
        This is the main workhorse loading function.
        
        Assumes the schedule files have data stored in ascending time/block order.
        
        Arguments:
        - folder: string, the data folder to find schedule files in.
        - with_moods: Boolean, if True loads some related mood data. It's faster when turned to False.
        - mood_figures: Boolean, if True, saves some mood-related figures in the data folder.
        - start_block: int, the starting block to apply. All data with blocks before this block are dropped unless the subject is specified in `subject_blocks`. Assumes the data is stored in ascending block order.
        - subject_blocks: dictionary, of the form {subject_name : [start_block, end_block]}. Specified the start and end block (inclusive) for the given subject, e.g. '219'.
        - MZ_PCA_load_folder: string or None, if not None the folder to go look for 'MZ_PCA.pk' in.
        """
        
        print(f"Reading the '*_schedule.db' files in the {folder} folder.")
        tm = time.time()
        
        self.subject_names = []
        self.colors = []
        filenames = []
        self.data_folder = folder
        
        N = 0
        self.T = []
        T0 = []
        T1 = []
        for filename in os.listdir(folder):
            if filename.endswith("_schedule.db"):
                filenames.append(filename)
                self.subject_names.append(re.search(".*(?=_schedule.db)", filename).group())
                self.colors.append(px.colors.qualitative.Plotly[N % 10])

                N += 1
                df = self.load_one_subject_df(os.path.join(folder, filename), verbose = False)
                block_n = df.block.values
                
                # Determine the start block and end block;
                if not 'all' in subject_blocks:
                    start_block_n = None
                    for k,v in subject_blocks.items():
                        if k == self.subject_names[-1]:
                            start_block_n = v[0]
                            end_block_n   = v[1]
                    if start_block_n is None:
                        start_block_n = start_block
                        end_block_n   = np.max(block_n)
                else:
                    start_block_n = subject_blocks['all'][0]
                    end_block_n   = subject_blocks['all'][1]
                
                # Determine the start trial and end trial:
                T0_n = -1; T1_n = -1
                t = 0
                while T0_n == -1:
                    if block_n[t] == start_block_n:
                        T0_n = t
                    t+=1
                    if t >= df.shape[0]:
                        sys.exit(f"restrict_subject_blocks > Error: start_block {start_block} not present for the subject {self.subject_names[-1]}.")
                        
                while T1_n == -1:
                    if t == df.shape[0] or (block_n[t] == end_block_n + 1):
                        T1_n = t 
                    t+=1
                    
                self.T.append(T1_n - T0_n)
                T0.append(T0_n); T1.append(T1_n)                
                
        self.T = np.array(self.T).astype('int32')
        self.N = N
        
        print(f"Loading {N} subjects:")        
        
        T = np.max(self.T)          
        
        self.stims    = -np.ones((T,2,N))
        self.p_R      = np.zeros((T,2,N))
        self.p_Pun    = np.zeros((T,2,N))
        self.feedback = np.zeros((T, N))
        self.block    = -np.ones((T,N))
        self.session  = -np.ones((T,N))
        self.C        = - np.ones((T, N)).astype('int64')
        self.C_not        = - np.ones((T, N)).astype('int64')
        self.C_st     = - np.ones((T, N)).astype('int64')
        self.o        = - 2 * np.ones((T,N))
        self.feedback_time = np.zeros((T,N))
        self.stim_time = np.zeros((T,N))
        self.choice_time = np.zeros((T,N))
        self.questionnaires = []
  
        self.trial         = np.zeros((T, N)).astype('int64')
        self.session_trial = np.zeros((T, N)).astype('int64')
        
        if with_moods:
            self.VA_base       = [None for n in range(N)]
            self.VA_time_base  = [None for n in range(N)]
            self.PRA_base      = [None for n in range(N)]
            self.MZ_PCA_base   = [None for n in range(N)]
            self.PRA_time_base = [None for n in range(N)]
            
            self.VA_names = ['Valence', 'Arousal', 'VA1', 'VA2']
            self.PRA_names = ['Anxious', 'Elated', 'Sad', 'Irritable', 'Energetic']
            self.MZ_PCA_names = [f'MZ_PCA{e + 1}' for e in range(3)]
        
            self.mood_names = self.VA_names + self.PRA_names + self.MZ_PCA_names
            self.mood       = [np.zeros((T,N)) for m_name in self.mood_names] # smoothed mood at trial feedback times
                
        n = 0
        for filename in filenames:
            df = self.load_one_subject_df(os.path.join(folder, filename), verbose = verbose)
            
            T_s = self.T[n]               
                
            self.stims[:T_s,:,n] = df[['stim1', 'stim2']].values[T0[n]:T1[n],:]             # (T, 2)
            self.p_R[:T_s,:,n]   = df[['reward1', 'reward2']].values[T0[n]:T1[n],:]         # (T, 2) 
            self.p_Pun[:T_s,:,n] = df[['punishment1', 'punishment2']].values[T0[n]:T1[n],:] # (T, 2)
            self.feedback[:T_s,n] = df['feedback'].values[T0[n]:T1[n]]                    # (T) 
            self.block[:T_s,n]   = df['block'].values[T0[n]:T1[n]]
            self.trial[:T_s,n]   = df['trial'].values[T0[n]:T1[n]]
            
            self.C_st[:T_s,n] = df['choice'].values.astype('int64')[T0[n]:T1[n]]
            self.o[:T_s,n] = df['outcome'].values[T0[n]:T1[n]]
            self.C[:T_s,n] = self.stims[np.arange(T_s).astype('int64'),self.C_st[:T_s,n],n]
            self.C_not[:T_s,n] = self.stims[np.arange(T_s).astype('int64'),1-self.C_st[:T_s,n],n]
            self.stim_time[:T_s,n] = df['stim_time'].values[T0[n]:T1[n]]
            self.choice_time[:T_s,n] = df['choice_time'].values[T0[n]:T1[n]]
            self.feedback_time[:T_s,n] = df['feedback_time'].values[T0[n]:T1[n]]
                        
            if with_moods:
                
                if mood_figures:
                    figure_folder = os.path.join(folder, 'mood_figures')
                    if not os.path.exists(figure_folder):
                        os.mkdir(figure_folder)
                else:
                    figure_folder = None
                    
                self.VA_base[n], self.VA_time_base[n], self.PRA_base[n], self.PRA_time_base[n] = self.load_base_moods(os.path.join(folder, filename))
                
            self.questionnaires.append(self.load_questionnaires(os.path.join(folder, filename), self.subject_names[n]))                
            self.session_trial[:T_s,n] = self.compute_session_trial_num(df.iloc[T0[n]:T1[n],:], os.path.join(folder, filename))            
            n += 1
        
        if with_moods:
            self.with_moods = True
        
            # Compute MZ_PCA:        
            self.compute_MZ_PCA(load_folder = MZ_PCA_load_folder)
            for n in range(N):
                PRA_n = np.concatenate([self.PRA_base[n][e][:, None] for e in range(5)], axis = 1)
                MZ_PCA_base = self.pc.transform(PRA_n)
                self.MZ_PCA_base[n] = [MZ_PCA_base[:,e] for e in range(3)]
                
            # Compute smoothed moods and optionally display them:
            smoothed_moods = self.compute_smoothed_moods(figures_save_folder = figure_folder, smoothing_window = smoothing_window)
            for n in range(N):
                for e in range(4+5+3):
                    self.mood[e][:self.T[n], n] = smoothed_moods[n][e]
        else:
            self.with_moods = False

        self.stims = self.stims.astype('int64')
        self.feedback = self.feedback.astype('int64')
        self.block = self.block.astype('int64')
        
        self.questionnaires = pd.concat(self.questionnaires, axis = 0)
                                
        self.N      = N                            
        self.Nc     = np.max(self.stims,axis=(0,1)) + 1 # (N)
        if np.min(self.stims)<-10: #if the schedule file includes fmri block
           self.Nc     = np.max(self.stims,axis=(0,1)) + 2
        
        self.compute_EV_and_best()
        self.compute_acc()
        self.compute_acc_exp()
        self.compute_new_stimuli_per_block()
        self.compute_stim_types()
        self.compute_stim_stage()
        self.compute_session_times()
        
        self.compute_num_learning_blocks()
        
        self.compute_time_since_last_rep()
        self.compute_last_feedback_time()
        self.compute_last_no_feedback_time()
        self.compute_first_no_feedback_time()
        self.compute_last_test_block()
        self.compute_mean_test_block()
        self.compute_first_test_block()
        self.compute_last_learning_block() 
        self.compute_learning_days()
        self.compute_learning_days_nf_num_appearance()
        
        self.compute_num_nopractice()
        self.compute_num_choice()
        
        self.num_datapoints = self.num_nopractice_f + self.num_nopractice_nf
        
        if with_moods:
            self.compute_learning_mood()
            
        return self
        
        
    def copy_subject_states(self, s_dat, name_spec = ""):
        """
        Given another subjects object, makes a deep copy into the current subject_EMA object.
        
        Arguments:
        - s_dat     : subjects object, of which the trajectory information will be copied.
        - name_spec : string, an optional specifier to be added to all the subject names.
        """
        
        self.T = s_dat.T.copy()
        self.N = s_dat.N
        self.Nc = s_dat.Nc.copy()
        
        self.stims = np.copy(s_dat.stims) # Takes a copy
        self.p_R   = np.copy(s_dat.p_R)
        self.p_Pun = np.copy(s_dat.p_Pun)
        self.feedback = np.copy(s_dat.feedback)
        self.block = np.copy(s_dat.block)
        self.C_st  = np.copy(s_dat.C_st).astype('int32')
        self.C     = np.copy(s_dat.C)
        self.o     = np.copy(s_dat.o)
        
        self.subject_names = [s_name + name_spec for s_name in s_dat.subject_names]
        self.colors = s_dat.colors
        
        self.EV = np.copy(s_dat.EV)
        self.best_stim = s_dat.best_stim.copy()
        self.new_stimuli = s_dat.new_stimuli
        self.new = np.copy(s_dat.new)
        self.num_new = np.copy(s_dat.num_new)
        self.stim_types = np.copy(s_dat.stim_types)
        self.acc = np.copy(s_dat.acc)
        self.chosen_stim_type = np.copy(s_dat.chosen_stim_type)
        self.EV_exp = np.copy(s_dat.EV_exp)
        self.acc_exp = np.copy(s_dat.acc_exp)
        
        self.feedback_time = np.copy(s_dat.feedback_time)
        self.choice_time = np.copy(s_dat.choice_time)
        self.stim_time = np.copy(s_dat.stime_time)
        
        if s_dat.with_moods:
            N = self.N
            self.VA_names     = s_dat.VA_names
            self.PRA_names    = s_dat.PRA_names
            self.MZ_PCA_names = s_dat.MZ_PCA_names
            
            self.VA_base       = [[np.copy(s_dat.VA_base[n][e]) for e in range(4)] for n in range(N)]
            self.VA_time_base  = [np.copy(s_dat.VA_time_base[n]) for n in range(N)]
            self.PRA_base      = [[np.copy(s_dat.PRA_base[n]) for e in range(5)] for n in range(N)]
            self.MZ_PCA_base   = [[np.copy(s_dat.PRA_time_base[n]) for e in range(3)] for n in range(N)]
            self.PRA_time_base = [np.copy(s_dat.PRA_time_base[n]) for n in range(N)]
        
            self.mood_names = s_dat.mood_names
            self.mood       = [np.copy(s_dat.mood[e]) for e in range(len(self.mood_names))] # smoothed mood at trial feedback times
            self.learning_mood       = [np.copy(s_dat.learning_mood[e]) for e in range(len(self.mood_names))]
            self.learning_mood_stim  = [np.copy(s_dat.learning_mood_stim[e]) for e in range(len(self.mood_names))]
               
        self.with_moods = s_dat.with_moods    

        self.questionnaires = s_dat.questionnaires.copy()
        
        self.trial = np.copy(s_dat.trial)
        self.session_trial = np.copy(s_dat.session_trial)
        
        self.compute_num_learning_blocks()
        self.stim_stage = np.copy(s_dat.stim_stage)
        self.compute_session_times()    
            
        self.time_since_last_rep = np.copy(s_dat.time_since_last_rep)
        self.last_feedback_time = pd.DataFrame(s_dat.last_feedback_time)
        self.last_no_feedback_time = pd.DataFrame(s_dat.last_no_feedback_time)
        self.first_no_feedback_time = pd.DataFrame(s_dat.first_no_feedback_time)
        self.last_test_block = pd.DataFrame(s_dat.last_test_block)
        self.first_test_block = pd.DataFrame(s_dat.first_test_block)
        self.mean_test_block = pd.DataFrame(s_dat.mean_test_block)
        self.last_learning_block = pd.DataFrame(s_dat.last_learning_block)
        self.learning_blocks = np.copy(s_dat.learning_blocks)
        self.learning_days   = np.copy(s_dat.learning_days)
        self.learning_days_nf_num_appearance = np.copy(s_dat.learning_days_nf_num_appearance)       
        
        self.compute_num_nopractice()
        
        self.num_choice = np.copy(s_dat.num_choice)
        self.num_r      = np.copy(s_dat.num_r)
        self.num_pun      = np.copy(s_dat.num_pun)
        self.num_appear = np.copy(s_dat.num_appear)
        self.num_nf_appear = np.copy(s_dat.num_nf_appear)
        
        self.data_folder = s_dat.data_folder
        
        return self
     
    def concatenate_subjects(self, s_dat, s_dat_b, MZ_PCA_folder = None, data_folder = None):
        """
        Given two subjects objects, stores the combination of their state and choice_outcome 
        information into this subjects data. Deep copies are not guaranteed.
        
        Requires the maximal T to be the same between the two subjects objects.
        """
        if data_folder is not None:
            self.data_folder = s_dat.data_folder
        
        if np.max(s_dat.T) != np.max(s_dat_b.T):
            sys.exit("This concatenate function requires the maximum number of trials over all subjects to be the same between the two subjects_EMA objects.")
        
        self.T = np.concatenate((s_dat.T, s_dat_b.T))        
        self.N = s_dat.N + s_dat_b.N
        self.Nc = s_dat.Nc + s_dat_b.Nc
        self.subject_names = s_dat.subject_names + s_dat_b.subject_names
        self.colors = s_dat.colors + s_dat_b.colors
        
        self.stims = np.concatenate((s_dat.stims, s_dat_b.stims), axis = 2).astype('int64') 
        self.p_R   = np.concatenate((s_dat.p_R, s_dat_b.p_R), axis = 2)
        self.p_Pun = np.concatenate((s_dat.p_Pun, s_dat_b.p_Pun), axis = 2)
        self.feedback = np.concatenate((s_dat.feedback, s_dat_b.feedback), axis = 1).astype('int64')
        self.block = np.concatenate((s_dat.block, s_dat_b.block), axis = 1).astype('int64')
      
        self.EV = np.concatenate((s_dat.EV, s_dat_b.EV), axis = 2)
        self.best_stim = pd.concat([s_dat.best_stim, s_dat_b.best_stim], axis = 1)
        self.new_stimuli = s_dat.new_stimuli + s_dat_b.new_stimuli
        self.new = np.concatenate([s_dat.new, s_dat_b.new], axis = 1)
        self.num_new = np.concatenate((s_dat.num_new, s_dat_b.num_new), axis =1)
        self.stim_types = np.concatenate((s_dat.stim_types, s_dat_b.stim_types), axis = 2)
        self.chosen_stim_type = np.concatenate((s_dat.chosen_stim_type, s_dat_b.chosen_stim_type), axis = 1)
        self.acc = np.concatenate((s_dat.acc, s_dat_b.acc), axis = 1)
        self.acc_exp = np.concatenate((s_dat.acc_exp, s_dat_b.acc_exp), axis = 1)
        self.EV_exp  = np.concatenate((s_dat.EV_exp, s_dat_b.EV_exp), axis = 2)
        
        self.C = np.concatenate((s_dat.C, s_dat_b.C),axis = 1).astype('int64')
        self.C_st = np.concatenate((s_dat.C_st, s_dat_b.C_st), axis = 1).astype('int64')
        self.o = np.concatenate((s_dat.o, s_dat_b.o), axis = 1)
        self.feedback_time = np.concatenate((s_dat.feedback_time, s_dat_b.feedback_time), axis = 1)
        
        if s_dat.with_moods and s_dat_b.with_moods:
            self.VA_names     = s_dat.VA_names
            self.PRA_names    = s_dat.PRA_names
            self.MZ_PCA_names = s_dat.MZ_PCA_names
            
            self.VA_base       = s_dat.VA_base + s_dat_b.VA_base
            self.VA_time_base  = s_dat.VA_time_base + s_dat_b.VA_time_base
            self.PRA_base      = s_dat.PRA_base + s_dat_b.PRA_base
            self.MZ_PCA_base   = s_dat.MZ_PCA_base + s_dat_b.MZ_PCA_base
            self.PRA_time_base = s_dat.PRA_time_base + s_dat_b.PRA_time_base
        
            self.mood_names = s_dat.mood_names
            self.mood       = [np.concatenate((s_dat.mood[e], s_dat_b.mood[e]), axis = 1) for e in range(len(self.mood_names))] # smoothed mood at trial feedback times
            self.learning_mood       = [np.concatenate((s_dat.learning_mood[e], s_dat_b.learning_mood[e]), axis = 2) for e in range(len(self.mood_names))] 
            self.learning_mood_stim  = [np.concatenate((s_dat.learning_mood_stim[e], s_dat_b.learning_mood_stim[e]), axis = 1) for e in range(len(self.mood_names))] 
            
            
            self.with_moods = True
        else:
            self.with_moods = False
            
        self.questionnaires = pd.concat([s_dat.questionnaires, s_dat_b.questionnaires], axis = 0)
        
        self.trial = np.concatenate((s_dat.trial, s_dat_b.trial), axis = 1)
        self.session_trial = np.concatenate((s_dat.session_trial, s_dat_b.session_trial), axis = 1)
        
        self.compute_num_learning_blocks()
        self.stim_stage = np.concatenate((s_dat.stim_stage, s_dat_b.stim_stage), axis = 2)
        self.compute_session_times()
        
        self.time_since_last_rep = np.concatenate((s_dat.time_since_last_rep, s_dat_b.time_since_last_rep), axis = 2)
        self.last_feedback_time = pd.concat((s_dat.last_feedback_time, s_dat_b.last_feedback_time), axis = 1)
        self.last_no_feedback_time = pd.concat((s_dat.last_no_feedback_time, s_dat_b.last_no_feedback_time), axis = 1)
        self.first_no_feedback_time = pd.concat((s_dat.first_no_feedback_time, s_dat_b.first_no_feedback_time), axis = 1)
        self.last_learning_block = pd.concat((s_dat.last_learning_block, s_dat_b.last_learning_block), axis = 1)
        self.last_test_block = pd.concat((s_dat.last_test_block, s_dat_b.last_test_block), axis = 1)
        self.first_test_block = pd.concat((s_dat.first_test_block, s_dat_b.first_test_block), axis = 1)
        self.mean_test_block = pd.concat((s_dat.mean_test_block, s_dat_b.mean_test_block), axis = 1)
        
        self.learning_blocks = np.concatenate((s_dat.learning_blocks, s_dat_b.learning_blocks), axis = 2)
        self.learning_days = np.concatenate((s_dat.learning_days, s_dat_b.learning_days), axis = 2)
        self.learning_days_nf_num_appearance = np.concatenate((s_dat.learning_days_nf_num_appearance, s_dat_b.learning_days_nf_num_appearance), axis = 1)

        self.num_nopractice_f = np.concatenate((s_dat.num_nopractice_f, s_dat_b.num_nopractice_f))
        self.num_nopractice_nf = np.concatenate((s_dat.num_nopractice_nf, s_dat_b.num_nopractice_nf))
        self.num_datapoints = np.concatenate((s_dat.num_datapoints, s_dat_b.num_datapoints))
       
        self.num_choice = np.concatenate((s_dat.num_choice, s_dat_b.num_choice), axis = 2)
        self.num_r      = np.concatenate((s_dat.num_r, s_dat_b.num_r), axis = 2)
        self.num_pun    = np.concatenate((s_dat.num_pun, s_dat_b.num_pun), axis = 2)
        self.num_appear = np.concatenate((s_dat.num_appear, s_dat_b.num_appear), axis = 2)
        self.num_nf_appear = np.concatenate((s_dat.num_nf_appear, s_dat_b.num_nf_appear), axis = 2)
        
        return self
        
    def save_choices(s_dat, filename):
        """
        A function that is used in automated Parameter Recovery Tests to save simulated choices.
        """
        
        #s_dat.simulate_data(style, P, P_aux = P_aux)
        with h5py.File(filename, 'w') as hf:
            hf.create_dataset('o', data = s_dat.o)
            hf.create_dataset('C', data = s_dat.C)
            hf.create_dataset('C_st', data = s_dat.C_st)
            hf.create_dataset('acc', data = s_dat.acc)
            hf.create_dataset('session', data = s_dat.session)
            hf.create_dataset('stim_types', data = s_dat.stim_types)
            gp = hf.create_group('P')
            for par_name in s_dat.P_true.columns:
                gp.create_dataset(par_name, data = s_dat.P_true[par_name].values)
            if not s_dat.P_aux_true is None:
                gp = hf.create_group('P_aux')
                for par_name in s_dat.P_aux_true.columns:
                    gp.create_dataset(par_name, data = s_dat.P_aux_true[par_name].values) 
                    
    def load_choices(s_dat, filename):
        """
        A function that is used in automated Parameter Recovery Tests to load simulated choices.
        """
        P_true = {}
        P_aux_true = {}
        
        with_P_aux = False
        
        with h5py.File(filename, 'r') as hf:
            s_dat.o = np.array(hf.get('o'))
            s_dat.C = np.array(hf.get('C')).astype('int64')
            s_dat.C_st = np.array(hf.get('C_st')).astype('int64')
            s_dat.acc = np.array(hf.get('acc'))
            s_dat.session = np.array(hf.get('session')).astype('int64')
            s_dat.stim_types = np.array(hf.get('stim_types')).astype('int64')
            gp = hf['P']
            for par_name in gp.keys():
                P_true[par_name] = np.array(gp.get(par_name))
            P_true = pd.DataFrame(P_true)
            if 'P_aux' in hf:
                with_P_aux = True
                gp = hf['P_aux']
                for par_name in gp.keys():
                    P_aux_true[par_name] = np.array(gp.get(par_name))
                P_aux_true = pd.DataFrame(P_aux_true)
        
        if with_P_aux:
            if len(P_aux_true.keys()) == 0:
                P_aux_true = None
        else:
            P_aux_true = None
            
        s_dat.P_aux_true = P_aux_true
        s_dat.P_true = P_true
        
        return s_dat
        
    ### Processing functions: ###
    def load_base_moods(self, filename):
        """
        The goal of this function is to load in the raw mood reports. 
        """        
        num_double_valence = 0        
        
        ## 0) Extract the VA & PRA reports
        query1 = 'SELECT answer_time, answer, answer_timestamp FROM answers WHERE questionnaire_type==0 AND question==0 AND questionnaire_number>0'
        query2 = 'SELECT answer_time, answer, answer_timestamp FROM answers WHERE questionnaire_type==0 AND question==1 AND questionnaire_number>0'

        conn = sqlite3.connect(filename, uri = True)
        df1 = pd.read_sql_query(query1, conn)
        df2 = pd.read_sql_query(query2, conn)
        conn.close()

        ## 1A) Valence, Arousal
        num_reports = df1.shape[0]
        VA_base = [np.zeros((num_reports)) for e in range(4)]
        VA_time_base = np.zeros((num_reports))
        
        for idx,row in df1.iterrows():
            answer = row['answer']
            
            VA_time_base[idx] = row['answer_time']

            match = re.findall("Valence=(.*),", answer)
            VA_base[0][idx] = float(match[0]) / 500.

            match = re.findall("Arousal=(.*)", answer)
            VA_base[1][idx] = float(match[0]) / 500. 
        
        VA_base[2] = np.sqrt(2) * (VA_base[0]   + VA_base[1]) / 2.
        VA_base[3] = np.sqrt(2) * (- VA_base[0] + VA_base[1]) / 2.            
        
        ## 1B) PRA
        num_reports = df2.shape[0]
        PRA_names = ['Anxious', 'Elated', 'Sad', 'Irritable', 'Energetic']
        PRA_base = [np.zeros((num_reports)) for e in range(len(PRA_names))]; 
        PRA_residuals = [np.nan for e in PRA_names]
        PRA_time_base = np.zeros((num_reports))
                
        for idx,row in df2.iterrows():           
            answer = row['answer']
            
            if answer[:7] != 'Valence': # 
                PRA_time_base[idx] = row['answer_time']
                
                for e, PRA_name in enumerate(PRA_names):
                    match_e = re.findall(f"{PRA_name}=([\-\d]*)", answer)
                    PRA_base[e][idx] = (float(match_e[0])) / 100.
            else: # Fix a bug where "Valence=..." would be copied in the PRA slot - take the average of the two surrounding mood reports instead
                num_double_valence += 1
                if idx < num_reports - 1:
                    answer2 = df2.iloc[idx+1,:]['answer']
                else:
                    answer2 = df2.iloc[idx-1,:]['answer']
                    
                PRA_time_base[idx] = row['answer_time']
                
                for e, PRA_name in enumerate(PRA_names):
                    match_e = re.findall(f"{PRA_name}=([\-\d]*)", answer2)
                    PRA_base[e][idx] = .5 * (PRA_base[e][idx-1] + (float(match_e[0])) / 100.)
        
        # Alon's preprocessing step: set all PRA moods with value <= 3 to 0, and between 47 and 53 to 50:
        for e, PRA_name in enumerate(PRA_names):
            PRA_base[e][PRA_base[e] <= .035] = 0.
            #PRA_base[e][(PRA_base[e] <= .535) & (PRA_base[e] >= .465)] = .5
            
        if num_double_valence > 0:
            print(f"   Subject {filename[:3]} has {num_double_valence} double valence copies. The PRA have been interpolated.")
        
        return VA_base, VA_time_base, PRA_base, PRA_time_base
                               

        
    def compute_smoothed_moods(self, figures_save_folder = None, smoothing_window = .25):
        """
        The goal of this function is to compute smoothed moods at feedback times for the given subject. For other mood analyses, the functions can be found in other files.
        
        Concretely, given 'feedback_time' times in the 'ms since 1970' format, this function computes: 
          (1) Gaussian filters with smoothing_window in units of days at feedback_time
                  
        Sometimes, there were duplicated valence reports, with Mood Zoom missing. In this case, we imputed the mean of the previous and next report - not adjusting for reporting time.
        During loading, the number of such entries get printed.

        - filename: string, name of the schedule file (path).
        - feedback_time: numpy array, the times to smooth moods at.
        - display_smoothing: if True, displays the smoothed trajectories, and saves them to 'figures_save_folder'
        
        Returns the numpy arrays of mood quantities.
        
        Interpolated moods and residuals can also be displayed with this function.
        """
        
        mood_smoothed_t_f     = [[np.zeros(self.T[n]) for e in range(4+5+3)] for n in range(self.N)] 
              
        for n in range(self.N):
            feedback_time = self.feedback_time[:self.T[n], n]
            time_base = self.VA_time_base[n]
            PRA_time_base = self.PRA_time_base[n]
            VA_base = self.VA_base[n]
            PRA_base = self.PRA_base[n]
            MZ_PCA_base  = self.MZ_PCA_base[n]
            
            ## 2) Apply Gaussian filters at feedback_times
            for t, t_f in enumerate(feedback_time):
                # VA
                delta_t = (t_f - time_base) / 1000. / 3600. / 24.
                weights = np.exp(-np.square(delta_t) * .5 / np.square(smoothing_window)) 
                
                for e in range(4):
                    mood_smoothed_t_f[n][e][t] = np.sum(self.VA_base[n][e] * weights) / np.sum(weights)
                
                # PRA, MZ_PCA
                delta_t = (t_f - PRA_time_base)  / 1000. / 3600. / 24.
                weights = np.exp(-np.square(delta_t) * .5 / np.square(smoothing_window)) 
                
                for e in range(5):
                    mood_smoothed_t_f[n][e+4][t] = np.sum(self.PRA_base[n][e] * weights) / np.sum(weights)            
                for e in range(3):
                    mood_smoothed_t_f[n][e+4+5][t] = np.sum(self.MZ_PCA_base[n][e] * weights) / np.sum(weights)
            
            ## 3) Optionally, compute Residuals of raw mood reports versus the smooths:
            if figures_save_folder is not None:
                
                
                VA_base_smoothed     = [np.zeros(self.VA_time_base[n].shape) for e in range(4)]
                PRA_base_smoothed    = [np.zeros(self.PRA_time_base[n].shape) for e in range(5)]
                MZ_PCA_base_smoothed = [np.zeros(self.PRA_time_base[n].shape) for e in range(3)]
                for t, t_b in enumerate(time_base):
                    delta_t = (t_b - time_base) / 1000. / 3600. / 24. # 0h after t_b
                    weights = np.exp(-np.square(delta_t) * .5 / np.square(smoothing_window)) # stdev is half a day
                    
                    for  e in range(4):
                        VA_base_smoothed[e][t] = np.sum(VA_base[e] * weights) / np.sum(weights)
                               
                for t, t_b in enumerate(PRA_time_base):
                    delta_t = (t_b - PRA_time_base) / 1000. / 3600. / 24. # 0h after t_b
                    weights = np.exp(-np.square(delta_t) * .5 / np.square(smoothing_window)) 
                    
                    for e in range(5):
                        PRA_base_smoothed[e][t] = np.sum(PRA_base[e] * weights) / np.sum(weights)  
                        
                    for e in range(3):
                        MZ_PCA_base_smoothed[e][t] = np.sum(MZ_PCA_base[e] * weights) / np.sum(weights)
                           
                residuals_folder = os.path.join(figures_save_folder, "residuals")
                if not os.path.exists(residuals_folder):
                    os.mkdir(residuals_folder)
                
                residuals = [None for e in range(4+5+3)]
                for e in range(4):
                    residuals[e] = self.VA_base[n][e] - VA_base_smoothed[e]
                for e in range(5):        
                    residuals[4+e] = self.PRA_base[n][e] - PRA_base_smoothed[e]
                for e in range(3):
                    residuals[4+5+e] = self.MZ_PCA_base[n][e] - MZ_PCA_base_smoothed[e]
                    
                names = self.VA_names + self.PRA_names + self.MZ_PCA_names
                time_bases = [time_base for e in range(4)] + [PRA_time_base for e in range(5+3)] 
                
                for e, name in enumerate(names):
                    fig_res = go.Figure()
                    fig_res.add_trace(go.Scatter(x = (time_bases[e] - time_bases[e][0])/ 1000. / 3600. / 24.,
                                      y = residuals[e],
                                      name = 'Residuals',
                                      marker_color = 'red'))
                    fig_res.update_layout(title = f'{self.subject_names[n]}) {name} residuals',
                                          xaxis_title = 'Days since first measurement',
                                          yaxis_title = 'Residual')
                    fig_res.write_image(os.path.join(residuals_folder, f"{self.subject_names[n]}_{name}_residuals.jpeg"))
                                          
                
            ## 4) Optionally: display the smoothed moods:
            if figures_save_folder is not None:
                time_cover  = np.linspace(time_base[0], PRA_time_base[-1], 1000)
                mood_bases = VA_base + PRA_base + MZ_PCA_base
                time_bases = [time_base for e in range(4)] + [PRA_time_base for e in range(5+3)] 
                names       = self.VA_names + self.PRA_names + self.MZ_PCA_names
                mood_covers = [None for e in range(4 + 5 + 3)]
                
                ranges = [[-1.,1.] for e in range(4)] + [[-.1,1.] for e in range(5)] + [[-1.,1.] for e in range(3)]
                
                for e in range(4 + 5 + 3):
                    mood_covers[e] = np.zeros_like(time_cover)

                    for t, t_f in enumerate(time_cover):
                        delta_t = (t_f - time_bases[e]) / 1000. / 3600. / 24.
                        weights = np.exp(-np.square(delta_t) * .5 / np.square(smoothing_window))
                        mood_covers[e][t] = np.sum(mood_bases[e] * weights) / np.sum(weights)
                
                    fig = go.Figure(go.Scatter(x=(time_cover - time_cover[0])/ 1000. / 3600. / 24., y = mood_covers[e], mode = 'lines', name = 'smoothed', marker_color = 'blue'))
                    fig.add_trace(go.Scatter(x=(time_bases[e] - time_cover[0])/ 1000. / 3600. / 24., y= mood_bases[e], mode = 'markers', name = 'raw', marker_color = 'red'))
                    
                    fig.update_layout(title = f'{self.subject_names[n]}) {names[e]} since the first no-practice measurement', xaxis_title = 'time (days)', yaxis_title = 'score', yaxis = {'range':[-1.,1.]})
                    fig.write_image(os.path.join(figures_save_folder, f'{self.subject_names[n]}_{names[e]}.jpeg'))
                    if self.subject_names[n] in self.subject_names[:1]:
                        fig.show()
                    
                # A separate Figure for some moods plotted together:
                fig = go.Figure()
                for e in range(4):
                    fig.add_trace(go.Scatter(x=(time_cover - time_cover[0])/ 1000. / 3600. / 24., y = mood_covers[e], mode = 'lines', name = names[e]))
                fig.update_layout(title = f'{self.subject_names[n]}) Smoothed VA Dynamics', xaxis_title = 'time (days)', yaxis_title = 'score', yaxis = {'range':[-.1,1.]})
                fig.write_image(os.path.join(figures_save_folder, f'{self.subject_names[n]}_VA.jpeg'))
                if self.subject_names[n] in self.subject_names[:1]:
                    fig.show()      
                    
                fig = go.Figure()
                for e in range(5):
                    fig.add_trace(go.Scatter(x=(time_cover - time_cover[0])/ 1000. / 3600. / 24., y = mood_covers[4+e], mode = 'lines', name = names[2+e]))
                fig.update_layout(title = f'{self.subject_names[n]}) Smoothed Affect Dynamics', xaxis_title = 'time (days)', yaxis_title = 'score', yaxis = {'range':[-.1,1.]})
                fig.write_image(os.path.join(figures_save_folder, f'{self.subject_names[n]}_PRA.jpeg'))
                if self.subject_names[n] in self.subject_names[:1]:
                    fig.show()                        
        
        return mood_smoothed_t_f
        
    def mood_smoother_absolute(self, times_to_smooth_at, subject_name, smoothing_window = .25):
        """
        Given times_to_smooth_at and a subject_name, gives the smoothed moods at the given times with a smoothing_window.
        
        - times_to_smooth_at : array of ints/floats, number of ms since 1970.
        - subject_name       : string, name of the subject to smooth for.
        - smoothing_window   : float, smoothing window in number of days.
        
        Returns:
        - A dataframe with the smoothed moods at the given times for the subject.
        """
        
        _, columns = self.subject_names_to_columns([subject_name], False)
        n = columns[0]
        
        times_to_smooth_at = times_to_smooth_at.astype('float')
              
        time_bases = [self.VA_time_base[n] for e in range(4)] + [self.PRA_time_base[n] for e in range(5+3)]
        mood_bases = self.VA_base[n] + self.PRA_base[n] + self.MZ_PCA_base[n]  

        mood_smoothed = np.zeros((times_to_smooth_at.shape[0],4+5+3))            

        for t, t_f in enumerate(times_to_smooth_at):
            for e in range(4+5+3):                
                delta_t = (t_f - time_bases[e]) / 1000. / 3600. / 24.
                weights = np.exp(-np.square(delta_t) * .5 / np.square(smoothing_window)) 

                mood_smoothed[t,e] = np.sum(mood_bases[e] * weights) / np.sum(weights)

        mood_smoothed = pd.DataFrame(data = mood_smoothed, columns = self.mood_names)
        mood_smoothed['time'] = times_to_smooth_at
        mood_smoothed['subject_name'] = self.subject_names[n]

        return mood_smoothed    
        
    def mood_smoother_relative(self, time_offsets, subject_names = None, smoothing_window = .25):
        """
        Given time_offsets and subject_names, gives the smoothed moods at every time offset from the first times of every session.

        - time_offsets       : array/list of ints, number of hours.
        - subject_names      : list of strings or None, name of the subjects to smooth for.
        - smoothing_window   : float, smoothing window in number of days.

        Returns:
        - A dataframe with the smoothed moods at the given times for the subject.
        """

        _, columns = self.subject_names_to_columns(subject_names, False)

        time_offsets = np.array(time_offsets).astype('int32')

        mood_smoothed_all = [None for n in columns]

        for n in columns:              
            time_bases = [self.VA_time_base[n] for e in range(4)] + [self.PRA_time_base[n] for e in range(5+3)]
            mood_bases = self.VA_base[n] + self.PRA_base[n] + self.MZ_PCA_base[n]  

            task_times = self.create_subject_df(n=n).groupby('session')['feedback_time'].first()

            mood_smoothed = [np.zeros((task_times.shape[0],time_offsets.shape[0])) for e in range(4+5+3)]          

            # Can be sped up with map calls:
            for t, t_f in enumerate(task_times):
                for e in range(4+5+3):
                    for k, offset in enumerate(time_offsets):
                        delta_t = (t_f - time_bases[e]) / 1000. / 3600. / 24. + offset / 24.
                        weights = np.exp(-np.square(delta_t) * .5 / np.square(smoothing_window)) 

                        mood_smoothed[e][t,k] = np.sum(mood_bases[e] * weights) / np.sum(weights)

            mood_smoothed = [pd.DataFrame(data = mood_smoothed[e], columns = [f'{self.mood_names[e]}_{time_offset}' for time_offset in time_offsets]) for e in range(4+5+3)]
            mood_smoothed = pd.concat(mood_smoothed, axis = 1)
            mood_smoothed['time'] = task_times
            mood_smoothed['subject_name'] = self.subject_names[n]
            mood_smoothed['day'] = np.arange(task_times.shape[0]).astype('int32')

            mood_smoothed_all[n] = mood_smoothed

        return pd.concat(mood_smoothed_all, axis = 0).reset_index(drop = True)  
        
    def load_mood_event_scores(self, data_folder = None, mE_figures = False, smoothing_window = .25):   
        """
        This is the main function to load in the mood Event data. 
        
        1. The base reports will be stored in self.base_mE, a list of N dataframes. 
        2. Additionally, smoothed affect scores are stored in self.smoothed_mE at task times,
           in a list of 4 (aggregated, absolute-aggregated, sum, absoulate-sum)-smoothed, 
           of (T,N) arrays.
           
        If mE_figures is True is given, smoothed mood event scores will be stored there.
        
        Arguments:
        - smoothing_window: float, smoothing window in units of days.
        - mE_figures      : boolean, if True, stores smoothed mE figures in self.data_folder.
        - data_folder     : string, the data_folder to load mE scores from. If None, uses self.data_folder.
        """

        if data_folder is None:
            data_folder = self.data_folder
            
        self.base_mE = [None for n in range(self.N)]
        
        tm = time.time()
        if mE_figures:
            wo = "with"
        else:
            wo = "without"
        print(f"Loading mood Event data {wo} figures.")
        
        # 1) Load the raw mood event data:
        n = 0
        for filename in os.listdir(data_folder):
            if filename.endswith("_schedule.db"):
                file = os.path.join(data_folder, filename) 
                subject_name = re.search(".*(?=_schedule.db)", filename).group()
                
                self.base_mE[n] = self.load_base_mood_events(file)
                n+=1 
                
        # 2) Compute smoothed mE
        self.smoothed_mE = self.compute_smoothed_mE(mE_figures = mE_figures, smoothing_window = smoothing_window)
        
        self.mE_processed = True     
        print(f"===> Took {time.time() - tm:.2f}s.")

    def load_base_mood_events(self, filename):
        """
        The goal of this function is to load the base mood event reports.
        """
        
        # 1) Extract Mood Event Data:            
        query = 'SELECT answer_time, answer FROM answers WHERE questionnaire_type==0 AND question==2 AND questionnaire_number>0'

        conn = sqlite3.connect(filename, uri = True)
        df_read = pd.read_sql_query(query, conn)
        conn.close()

        num_reports = df_read.shape[0]
        questionnaire_ID = []
        time = []
        category = []
        description = []
        hours_past = []
        affect   = []
        pleasure_suffering = []
        loved_lonely       = []
        powerful_weak      = []
        threatened_safe    = []

        ID = 0
        for idx,row in df_read.iterrows():
            answer = row['answer']

            answer_time = row['answer_time']

            match_cat   = re.findall("category=([^,]*)", answer)
            match_descr = re.findall("description=([^,]*),", answer)
            match_hours = re.findall("time_ago=([^,]*),", answer)
            match_affect   = re.findall("Good-Bad=([^,]*)", answer)
            match_pl_suff  = re.findall("Physical Pleasure-Physical Pain=([^,]*)", answer)
            match_lov_lon  = re.findall("Loved-Lonely=([^,]*)", answer)
            match_pow_weak = re.findall("Powerful-Weak=([^,]*)", answer)
            match_thr_safe = re.findall("Safe-Threatened=([-]?\d+)", answer)

            for k in range(len(match_cat)):
                questionnaire_ID.append(ID)
                time.append(answer_time)

                category.append(match_cat[k])
                description.append(match_descr[k])
                hours_past.append(match_hours[k])
                affect.append(match_affect[k])
                pleasure_suffering.append(match_pl_suff[k])
                loved_lonely.append(match_lov_lon[k])
                powerful_weak.append(match_pow_weak[k])
                threatened_safe.append(match_thr_safe[k])

            ID += 1
 
        affect = np.array(affect).astype('float')
        affect[(affect > 46.5) & (affect < 53.5)] = 50.
        affect = (affect - 50.) / 50.
        
        df = pd.DataFrame({'questionnaire_ID': questionnaire_ID,
                           'category'        : category,
                           'affect_score'    : affect,
                           'answer_time'            : time,
                           'hours_past'      : hours_past,
                           'description'     : description,
                           'pleasure - suffering' : pleasure_suffering,
                           'loved - lonely'  : loved_lonely,
                           'powerful - weak' : powerful_weak,
                           'threatened - safe' : threatened_safe}
                           )
                           
        df['answer_time'] = df['answer_time'].astype('float')
        df['hours_past'] = df['hours_past'].astype('float')
        df['category'].map({'מחשבה': 'Thought', 'אירוע/פעילות לבד': 'Alone', 'אירוע/פעילות עם אחרים': 'Together'})

        return df
            
    def compute_smoothed_mE(self, mE_figures = False, smoothing_window = .25):
        """
        The goal of this function is to compute smoothed mood event scores at feedback times for the given subject.
        Concretely, given 'feedback_time' times in the 'ms since 1970' format, this function computes:
          (1) Gaussian filters with smoothing_window in units of days at feedback_time
        - filename: string, name of the schedule file (path).
        - feedback_time: numpy array, the times to smooth moods at.
        - display_smoothing: if True, displays the smoothed trajectories, and saves them to 'figures_save_folder'
        Returns the numpy arrays of mood quantities.
        Interpolated moods and residuals can also be displayed with this function.
        """
        mE_all     = [np.zeros((np.max(self.T), self.N)) for e in range(4)] # agg, absagg, sum, abssum
        
        for n in range(self.N):
            feedback_time = self.feedback_time[:self.T[n], n]
            time_base = self.base_mE[n]['answer_time'].values.astype('float')  - self.base_mE[n]['hours_past'].values * 3600. * 1000.
            affect_scores = self.base_mE[n]['affect_score'].values           
            
            ## 2) Apply Gaussian filters at feedback_times
            for t, t_f in enumerate(feedback_time):                
                delta_t = (t_f - time_base) / 1000. / 3600. / 24.
                weights = np.exp(-np.square(delta_t) * .5 / np.square(smoothing_window))
                
                # 0) agg                
                mE_all[0][t,n] = np.sum(affect_scores * weights) / np.sum(weights)
                
                # 1) absagg
                mE_all[1][t,n] = np.sum(np.abs(affect_scores) * weights) / np.sum(weights)
                
                # 2) sum
                mE_all[2][t,n] = np.sum(affect_scores * weights)# / np.sum(weights)
                
                # 3) abssum
                mE_all[3][t,n] = np.sum(np.abs(affect_scores) * weights) #/ np.sum(weights)
                
        if mE_figures:                
            mood_event_folder = os.path.join(self.data_folder, "mood_event_data")
            if not os.path.exists(mood_event_folder):
                os.mkdir(mood_event_folder)
            smoothed_mood_event_folder = os.path.join(mood_event_folder, "Smoothed")
            if not os.path.exists(smoothed_mood_event_folder):
                os.mkdir(smoothed_mood_event_folder)
            
            # Smoothed mood figures:        
            for n in range(self.N):
                df = self.base_mE[n]
                
                fig_mE = go.Figure()
                
                zero_time = df['answer_time'].iloc[0]
                def smooth_at_t(t, df, smoothing_window = smoothing_window, offset = 0.):
                    # offset in hours
                    delta_t = (t - df['answer_time'].values) / 1000. / 3600. + df['hours_past'].values + offset
                    weights = np.exp(-np.square(delta_t) * .5 / np.square(smoothing_window * 24.)) 
                    return np.sum(df['affect_score'].values * weights) / np.sum(weights)
                
                def smooth_at_t_sum(t, df, smoothing_window = smoothing_window, offset = 0.):
                    # offset in hours
                    delta_t = (t - df['answer_time'].values) / 1000. / 3600. + df['hours_past'].values + offset
                    weights = np.exp(-np.square(delta_t) * .5 / np.square(smoothing_window * 24.)) 
                    return np.sum(df['affect_score'].values * weights) #/ np.sum(weights)
                    
                c = 0
                for k, v in {'מחשבה': 'Thought', 'אירוע/פעילות לבד': 'Alone', 'אירוע/פעילות עם אחרים': 'Together'}.items():
                    df_sub = df[df['category'] == k]
                    fig_mE.add_trace(go.Scatter(x= (df_sub['answer_time'] - zero_time)/1000./3600./24.,
                                               y = df_sub['affect_score'],
                                               name = f"{self.subject_names[n]}) {v}",
                                               mode = 'markers',
                                                marker_color = px.colors.qualitative.Plotly[c]
                                    ))
                    if df_sub.shape[0] > 5:
                        task_times_n = np.linspace(np.min(df_sub['answer_time']), np.max(df_sub['answer_time']), 100)
                        smoothed_affect_sub = list(map(lambda t: smooth_at_t_sum(t, df_sub, offset= 0), task_times_n))
    #                         print(smoothed_affect_sub)
    #                         print(task_times_n)
                        fig_mE.add_trace(go.Scatter(x= (task_times_n - zero_time)/1000./3600./24.,
                                                   y = smoothed_affect_sub,
                                                   name = f"{self.subject_names[n]}) {v} sum-smoothed",
                                                   mode = 'lines',
                                                    marker_color = px.colors.qualitative.Plotly[c]
                                                ))
                    c += 1
                
                task_times_n = np.linspace(np.min(df['answer_time']), np.max(df['answer_time']), 100)
                smoothed_affect_sum = list(map(lambda t: smooth_at_t_sum(t, df, offset= 0), task_times_n))

                fig_mE.add_trace(go.Scatter(x= (task_times_n - zero_time)/1000./3600./24.,
                                           y = smoothed_affect_sum,
                                           name = f"{self.subject_names[n]}) All sum-smoothed",
                                           mode = 'lines',
                                            marker_color = 'black'
                                        ))
                fig_mE.update_layout(title = f'{self.subject_names[n]}) Sum-Smoothed Mood Event Affect Scores',
                                    xaxis_title = "Time since first mood event (days)",
                                    yaxis_title = "Affect Score")
                fig_mE.write_image(os.path.join(smoothed_mood_event_folder, f"{self.subject_names[n]}_mE.jpeg"))
                
                smoothed_affect = list(map(lambda t: smooth_at_t(t, df, offset= 0), task_times_n))
                smoothed_affect_sum = list(map(lambda t: smooth_at_t_sum(t, df, offset= 0), task_times_n))
#                
                fig_VA = go.Figure()
                fig_VA.add_trace(go.Scatter(x= (task_times_n - zero_time)/1000./3600./24.,
                                                   y = smoothed_affect,
                                               name = f"Agg Smoothed Affect",
                                               mode = 'lines',
                                                marker_color = px.colors.qualitative.Plotly[4]
                                    ))
                fig_VA.add_trace(go.Scatter(x= (task_times_n - zero_time)/1000./3600./24.,
                                   y = smoothed_affect_sum,
                               name = f"Sum Smoothed Affect",
                               mode = 'lines',
                                marker_color = px.colors.qualitative.Plotly[5]
                    ))
                
                mood0 = self.mood_smoother_absolute(task_times_n, self.subject_names[n])
                fig_VA.add_trace(go.Scatter(x= (task_times_n - zero_time)/1000./3600./24.,
                                   y = mood0['Valence'],
                               name = f"Valence",
                               mode = 'lines',
                                marker_color = px.colors.qualitative.Plotly[0]
                    ))
                fig_VA.add_trace(go.Scatter(x= (task_times_n - zero_time)/1000./3600./24.,
                                   y = mood0['Arousal'],
                               name = f"Arousal",
                               mode = 'lines',
                                marker_color = px.colors.qualitative.Plotly[1]
                    ))
                fig_VA.update_layout(title = f"{self.subject_names[n]}) VA vs Smoothed Affect",
                                    xaxis_title = 'Time (days)',
                                    yaxis_title = 'Affect')
                fig_VA.write_image(os.path.join(smoothed_mood_event_folder, f"{self.subject_names[n]}_VA_mE.jpeg"))  
                
            # Also add some histograms of the base mood:
            raw_mood_event_folder = os.path.join(mood_event_folder, "Raw")
            if not os.path.exists(raw_mood_event_folder):
                os.mkdir(raw_mood_event_folder)
            df = pd.concat(self.base_mE, axis = 0).reset_index(drop = True)
            for k, clm in enumerate(['category', 'hours_past', 'affect_score', 'pleasure - suffering', 'loved - lonely', 'powerful - weak', 'threatened - safe']):
                fig = px.histogram(df, x = clm)
                if k > 2:
                    fig.update_layout(title = clm, xaxis = {'range': [0.,100.]})
                elif k == 2:
                    fig.update_layout(title = clm, xaxis = {'range': [-1.,1.]})
                else:
                    fig.update_layout(title = clm)
                fig.write_image(os.path.join(raw_mood_event_folder, f"{clm}_hist.jpeg"))
                                    
        return mE_all
        
    def mE_smoother_absolute(self, times_to_smooth_at, subject_name, smoothing_window = .25):
        """
        Given times_to_smooth_at and a subject_name, gives the smoothed mood event scores at the given times with a smoothing_window.
        
        - times_to_smooth_at : array of ints/floats, number of ms since 1970.
        - subject_name       : string, name of the subject to smooth for.
        - smoothing_window   : float, smoothing window in number of days.
        
        Returns:
        - A dataframe with the smoothed moods at the given times for the subject.
        """
        
        if not self.mE_processed:
            sys.exit("Run `load_mood_event_scores` first.")
        
        _, columns = self.subject_names_to_columns([subject_name], False)
        n = columns[0]
        
        times_to_smooth_at = times_to_smooth_at.astype('float')
              
        time_base = (self.base_mE[n]['answer_time'] - self.base_mE[n]['hours_past'] * 1000. * 3600.).values
        mood_base = self.base_mE[n]['affect_score'].values

        mood_smoothed = np.zeros((times_to_smooth_at.shape[0],4))            

        for t, t_f in enumerate(times_to_smooth_at):
            delta_t = (t_f - time_base) / 1000. / 3600. / 24.
            weights = np.exp(-np.square(delta_t) * .5 / np.square(smoothing_window)) 

            mood_smoothed[t,0] = np.sum(mood_base * weights) / np.sum(weights)
            mood_smoothed[t,1] = np.sum(np.abs(mood_base) * weights) / np.sum(weights)
            mood_smoothed[t,2] = np.sum(mood_base * weights) #/ np.sum(weights)
            mood_smoothed[t,3] = np.sum(np.abs(mood_base) * weights) #/ np.sum(weights)

        mood_smoothed = pd.DataFrame(data = mood_smoothed, columns = ['agg_affect_score', 'absagg_affect_score', 'sum_affect_score', 'abssum_affect_score'])
        mood_smoothed['time'] = times_to_smooth_at
        mood_smoothed['subject_name'] = self.subject_names[n]

        return mood_smoothed    
        
    def mE_smoother_relative(self, time_offsets, subject_names = None, smoothing_window = .25):
        """
        Given time_offsets and subject_names, gives the smoothed mood event scores at every time offset from the first times of every session.

        - time_offsets       : array/list of ints, number of hours.
        - subject_names      : list of strings or None, name of the subjects to smooth for.
        - smoothing_window   : float, smoothing window in number of days.

        Returns:
        - A dataframe with the smoothed moods at the given times for the subject.
        """
            
        if not self.mE_processed:
            sys.exit("Run `load_mood_event_scores` first.")

        _, columns = self.subject_names_to_columns(subject_names, False)

        time_offsets = np.array(time_offsets).astype('int32')

        mood_smoothed_all = [None for n in columns]
        
        mE_names = ['agg_affect_score', 'absagg_affect_score', 'sum_affect_score', 'abssum_affect_score']

        for n in columns:              
            time_base = (self.base_mE[n]['answer_time'] - self.base_mE[n]['hours_past'] * 1000. * 3600.).values
            mood_base = self.base_mE[n]['affect_score'].values
            
            task_times = self.create_subject_df(n=n).groupby('session')['feedback_time'].first()

            mood_smoothed = [np.zeros((task_times.shape[0],time_offsets.shape[0])) for e in range(4)]          

            # Can be sped up with map calls:
            for t, t_f in enumerate(task_times):
                for k, offset in enumerate(time_offsets):
                    delta_t = (t_f - time_base) / 1000. / 3600. / 24. + offset / 24.
                    weights = np.exp(-np.square(delta_t) * .5 / np.square(smoothing_window)) 

                    mood_smoothed[0][t,k] = np.sum(mood_base * weights) / np.sum(weights)
                    mood_smoothed[1][t,k] = np.sum(np.abs(mood_base) * weights) / np.sum(weights)
                    mood_smoothed[2][t,k] = np.sum(mood_base * weights) #/ np.sum(weights)
                    mood_smoothed[3][t,k] = np.sum(np.abs(mood_base) * weights) #/ np.sum(weights)

            mood_smoothed = [pd.DataFrame(data = mood_smoothed[e], columns = [f'{mE_names[e]}_{time_offset}' for time_offset in time_offsets]) for e in range(4)]
            mood_smoothed = pd.concat(mood_smoothed, axis = 1)
            mood_smoothed['time'] = task_times
            mood_smoothed['subject_name'] = self.subject_names[n]

            mood_smoothed_all[n] = mood_smoothed

        return pd.concat(mood_smoothed_all, axis = 0).reset_index(drop = True)  
        
    def load_sleep_reports(self, data_folder = None, figures = True):   
        """
        This is the main function to load in the sleep data. 
        
        1. The base reports will be stored in self.base_sleep, a list of N dataframes. 
           
        If figures is True, smoothed mood event scores will be stored there.
        
        Arguments:
        - figures      : boolean, if True, stores smoothed mE figures in self.data_folder.
        - data_folder     : string, the data_folder to load mE scores from. If None, uses self.data_folder.
        """

        if data_folder is None:
            data_folder = self.data_folder
            
        self.base_sleep = [None for n in range(self.N)]
        
        tm = time.time()
        if figures:
            wo = "with"
        else:
            wo = "without"
        print(f"Loading sleep data {wo} figures.")
        
        # 1) Load the raw mood event data:
        n = 0
        for filename in os.listdir(data_folder):
            if filename.endswith("_schedule.db"):
                file = os.path.join(data_folder, filename) 
                subject_name = re.search(".*(?=_schedule.db)", filename).group()
                
                self.base_sleep[n] = self.load_base_sleep_report(file)
                n+=1 
                
        if figures:
            sleep_folder = os.path.join(self.data_folder, 'sleep')
            if not os.path.exists(sleep_folder):
                os.mkdir(sleep_folder)
        
            for n in range(self.N):
                df_n = self.base_sleep[n]
                
                for var in ['overall', 'latency', 'woke_many_times', 'woke_early']:
                    subfolder = os.path.join(sleep_folder, var)
                    if not os.path.exists(subfolder):
                        os.mkdir(subfolder)
                
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x= (df_n['answer_time'] - df_n['answer_time'].iloc[0])/1000./3600./24.,
                                               y = df_n[var],
                                               mode = 'markers',
                                                marker_color = 'blue'
                                            ))
                    fig.update_layout(title = f'{self.subject_names[n]}) Raw sleep: {var}',
                                        xaxis_title = "Time since first report (days)",
                                        yaxis_title = f"{var}")
                    fig.write_image(os.path.join(sleep_folder, f"{self.subject_names[n]}_raw_{var}.jpeg"))
                        
        self.sleep_processed = True     
        print(f"===> Took {time.time() - tm:.2f}s.")

    def load_base_sleep_report(self, filename):
        """
        The goal of this function is to load the base mood event reports.
        """
        
        # 1) Extract Mood Event Data:            
        query = 'SELECT answer_time, answer FROM answers WHERE questionnaire_type==5 AND question==0 AND questionnaire_number>0'

        conn = sqlite3.connect(filename, uri = True)
        df_read = pd.read_sql_query(query, conn)
        conn.close()
        
        #  latency=50, woke many times=52, woke early=46, overall=50
        num_reports = df_read.shape[0]
        questionnaire_ID = []
        time = []
        latency = []
        woke_many_times = []
        woke_early = []
        overall   = []

        ID = 0
        for idx,row in df_read.iterrows():
            answer = row['answer']

            time.append(row['answer_time'])

            questionnaire_ID.append(ID)
            latency.append(re.findall("latency=([^,]*)", answer)[0])
            woke_many_times.append(re.findall("woke many times=([^,]*),", answer)[0])
            woke_early.append(re.findall("woke early=([^,]*),", answer)[0])
            overall.append(re.findall("overall=([-]?\d+)", answer)[0])

            ID += 1
        
        df = pd.DataFrame({'questionnaire_ID': questionnaire_ID,
                           'answer_time'     : time,
                           'latency'      : latency,
                           'woke_many_times'     : woke_many_times,
                           'woke_early' : woke_early,
                           'overall'  : overall}
                           )
                           
        for clm in ['latency', 'woke_many_times', 'woke_early', 'overall']:
            #display(df[clm])
            df[clm] = df[clm].astype('float')
            df.loc[(df[clm] > 46.5) & (df[clm] < 53.5), clm] = 50.
            df[clm] = (df[clm] - 50.) / 50.
                           
        df['answer_time'] = df['answer_time'].astype('float')

        return df
        
    def compute_last_and_next_sleep_reports(self, subject_names = None):
        """
        Given subject_names, gives the raw last and next sleep reports before and after the sessions.

        - subject_names      : list of strings or None, name of the subjects to smooth for.

        Returns:
        - A dataframe with the smoothed moods at the given times for the subject.
        """
            
        if not self.sleep_processed:
            sys.exit("Run `load_sleep_reports` first.")

        _, columns = self.subject_names_to_columns(subject_names, False)

        sleep_last_next_all = [None for n in columns]
        
        var_names = ['sleep_latency_0', 'woke_many_times_0', 'woke_early_0', 'sleep_quality_0', 'sleep_latency_24', 'woke_many_times_24', 'woke_early_24', 'sleep_quality_24']

        for n in columns:              
            time_base = self.base_sleep[n]['answer_time'].values            
            task_times = self.create_subject_df(n=n).groupby('session')['feedback_time'].first().values

            sleep_last_next = [np.full((task_times.shape[0]), np.nan) for e in range(8)]          

            t_b = 0 # Counter for the base time
            t_f = 0 # Counter for the task times
            while task_times[0] < time_base[t_b]:
                t_b = t_b + 1
            
            while t_b < time_base.shape[0] - 1 and t_f < task_times.shape[0]:
                if task_times[t_f] > time_base[t_b + 1]:
                    t_b = t_b+1
                else:
                    # last report:
                    for e, sl_name in enumerate(['latency', 'woke_many_times', 'woke_early', 'overall']):
                        sleep_last_next[e][t_f] = self.base_sleep[n][sl_name].iloc[t_b]
                    # next report:
                    for e, sl_name in enumerate(['latency', 'woke_many_times', 'woke_early', 'overall']):
                        sleep_last_next[4 + e][t_f] = self.base_sleep[n][sl_name].iloc[t_b + 1]
                    t_f = t_f+1
            while t_f < task_times.shape[0]:
                # last report:
                for e, sl_name in enumerate(['latency', 'woke_many_times', 'woke_early', 'overall']):
                        sleep_last_next[e][t_f] = self.base_sleep[n][sl_name].iloc[t_b]
                t_f = t_f+1

            sleep_ln = [pd.DataFrame(data = sleep_last_next[e], columns = [var_names[e]]) for e in range(8)]
            sleep_ln = pd.concat(sleep_ln, axis = 1)
            sleep_ln['time'] = task_times
            sleep_ln['subject_name'] = self.subject_names[n]

            sleep_last_next_all[n] = sleep_ln

        return pd.concat(sleep_last_next_all, axis = 0).reset_index(drop = True)  
        
    def load_activity(self, data_file, figures = True, smoothing_window = .25):
        """
        This is the main function to load in the sleep data. 
        
        1. The base reports will be stored in self.base_activity, a list of N dataframes. 
           
        If figures is True, smoothed acticity scores will be stored there.
        TODO: aggregate figures per 6 or 24 hours
        
        Arguments:
        - figures      : boolean, if True, stores smoothed mE figures in self.data_folder.
        - data_folder     : string, the data_folder to load mE scores from. If None, uses self.data_folder.
        """
        
        self.base_activity = [None for n in range(self.N)]
        
        df = pd.read_csv("..\\watch.csv", index_col = 0)
        df = df[df['timestamp'].notna()]
        df['timestamp_middle'] = df['timestamp'] - 900 # - 15min; timestamp is in sec
        df.rename(columns = {'ID':'subject_name'}, inplace = True)
        df['subject_name'] = df['subject_name'].astype('str')
        #df = df[df['mean_ENMO_mg'].notna()]    
        
        print("")
        for n in range(self.N):
            df_n = df[df['subject_name'] == self.subject_names[n]]
        
            self.base_activity[n] = df_n.sort_values('timestamp_middle')
            
        if figures:
            A_folder = os.path.join(self.data_folder, 'activity')
            if not os.path.exists(A_folder):
                os.mkdir(A_folder)
                
            for n in range(self.N):
                print(f"\rMaking figures for subject {n+1:3d}/{self.N:3d}.", end = "")
                df_n = self.base_activity[n]

                fig = go.Figure()

                zero_time = np.min(df_n['timestamp_middle'])
                def smooth_at_t(t, df, smoothing_window = smoothing_window, offset = 0.):
                    # offset in hours
                    delta_t = (t - df['timestamp_middle'].values) / 3600. + offset
                    weights = np.exp(-np.square(delta_t) * .5 / np.square(smoothing_window * 24.)) 
                    return np.nansum(df['mean_ENMO_mg'].values * weights) / np.nansum(weights)
                
                def smooth_at_t_sum(t, df, smoothing_window = smoothing_window, offset = 0.):
                    # offset in hours
                    delta_t = (t - df['timestamp_middle'].values) / 3600. + offset
                    weights = np.exp(-np.square(delta_t) * .5 / np.square(smoothing_window * 24.)) 
                    return np.nansum(df['mean_ENMO_mg'].values * weights) #/ np.sum(weights)

                task_times_n = np.linspace(np.min(df_n['timestamp_middle']), np.max(df_n['timestamp_middle']), 200)
                smoothed_activity = list(map(lambda t: smooth_at_t(t, df_n, offset= 0), task_times_n))
    #            print(f"test {n}")
    #             print(task_times_n)
    #             print(smoothed_activity)
    #             display(df_n.head(10))
    #             display(df_n.tail(10))
    #             print((task_times_n - zero_time)/1000./3600./24.)
                
                fig.add_trace(go.Scatter(x= (df_n['timestamp_middle'] - zero_time)/3600./24.,
                                           y = df_n['mean_ENMO_mg'],
                                           name = f"{self.subject_names[n]}) Raw actvity",
                                           mode = 'markers',
                                            marker_color = 'red',
                                         line = {'width':.33}
                                        ))
                fig.add_trace(go.Scatter(x= (task_times_n - zero_time)/3600./24.,
                                   y = smoothed_activity,
                                   name = f"{self.subject_names[n]}) Smoothed actvity",
                                   mode = 'lines',
                                    marker_color = 'blue'
                                ))
                fig.update_layout(title = f'{self.subject_names[n]}) Smoothed and Raw activity',
                                    xaxis_title = "Time since first non-NA activity (days)",
                                    yaxis_title = "mean_ENMO_mg")
                fig.write_image(os.path.join(A_folder, f"{self.subject_names[n]}_mean_ENMO_mg.jpeg"))  
    #           print(f"test {n} 2")
    #            if n == 11:
    #                print((task_times_n - zero_time))
                
                fig = go.Figure()
                task_times_n = np.linspace(np.min(df_n['timestamp_middle']), np.max(df_n['timestamp_middle']), 200)
                smoothed_activity = list(map(lambda t: smooth_at_t_sum(t, df_n, offset= 0), task_times_n))
    #            print(f"test {n}")
    #             print(task_times_n)
    #             print(smoothed_activity)
    #             display(df_n.head(10))
    #             display(df_n.tail(10))
    #             print((task_times_n - zero_time)/1000./3600./24.)
                
                fig.add_trace(go.Scatter(x= (df_n['timestamp_middle'] - zero_time)/3600./24.,
                                           y = df_n['mean_ENMO_mg'],
                                           name = f"{self.subject_names[n]}) Raw actvity",
                                           mode = 'markers',
                                            marker_color = 'red',
                                         line = {'width':.33}
                                        ))
                fig.add_trace(go.Scatter(x= (task_times_n - zero_time)/3600./24.,
                                   y = smoothed_activity,
                                   name = f"{self.subject_names[n]}) Smoothed sum actvity",
                                   mode = 'lines',
                                    marker_color = 'blue'
                                ))
                fig.update_layout(title = f'{self.subject_names[n]}) Smoothed sum and Raw activity',
                                    xaxis_title = "Time since first non-NA activity (days)",
                                    yaxis_title = "mean_ENMO_mg")
                fig.write_image(os.path.join(A_folder, f"{self.subject_names[n]}_mean_ENMO_mg_sum.jpeg"))  
            
        self.activity_processed = True
    
    def compute_agg_activity(self, subject_names = None, offsets = [0, 6, 12, 18, 24]):
        """
        This function computes the aggregate activity scores before the task times in:
        - 6 hour frames (activity_shortagg)
        - 24 hour time frames (activity_dayagg)
        
        Returns a DataFrame.
        
        NOTE: if a timeframe has a NAN in the activity data, the resulting aggregate score will be nan.
        """
        
        if not self.activity_processed:
            sys.exit("Run `load_activity()` first.")
            
        _, columns = self.subject_names_to_columns(subject_names, False)

        A_last_next_all = [None for n in columns]

        var_names = [f'activity_shortagg_{offset}' for offset in offsets] + [f'activity_dayagg_{offset}' for offset in [0,24]]

        for n in columns:
            df_n = self.base_activity[n]
            ENMO = df_n['mean_ENMO_mg']
            time_base = df_n['timestamp']  * 1000       
            task_times = self.create_subject_df(n=n).groupby('session')['feedback_time'].first().values 

            A_last_next = [np.full((task_times.shape[0]), np.nan) for e in range(2*len(offsets))] 
            
            for k, t in enumerate(task_times):
                for e, offset in enumerate(offsets):
    #                 subselect = df_n[(time_base > t - (6 - offset) * 3600 * 1000) & (time_base < t + offset * 3600 * 1000)]
    #                 #display(subselect)
                    A_last_next[e][k] = ENMO[(time_base > t - (6 - offset) * 3600 * 1000) & (time_base < t + offset * 3600 * 1000)].mean(skipna = False)
                for e, offset in enumerate([0, 24]):
    #                 subselect = df_n[(time_base > t - (24 - offset) * 3600 * 1000) & (time_base < t + offset * 3600 * 1000)]
    #                 #display(subselect)
                    A_last_next[len(offsets) + e][k] = ENMO[(time_base > t - (24 - offset) * 3600 * 1000) & (time_base < t + offset * 3600 * 1000)].mean(skipna = False)

            A_ln = [pd.DataFrame(data = A_last_next[e], columns = [var_names[e]]) for e in range(len(offsets) + 2)]
            A_ln = pd.concat(A_ln, axis = 1)
            A_ln['time'] = task_times
            A_ln['subject_name'] = self.subject_names[n]

            A_last_next_all[n] = A_ln

        return pd.concat(A_last_next_all, axis = 0).reset_index(drop = True)  

                    
    def compute_MZ_PCA(self, load_folder = None):
        """
        Computes the Mood Zoom PCA and stores the result in `save_folder`. If save_folder is None, it is saved in the data_folder.
        
        - load_folder: string or None, if not None, will look for 'pca.pkl' in the load_folder to load precomputed PCA directions.
        """
       
        if load_folder is None:
            save_folder = self.data_folder
               
            PRA_all = np.concatenate([np.concatenate([self.PRA_base[n][e][:, None] for e in range(5)], axis = 1) for n in range(self.N)], axis = 0)

            self.pc = dc.PCA(n_components = 3)
            self.pc.fit(PRA_all)
            df_pca = pd.DataFrame(self.pc.components_, columns = self.PRA_names)
            print('\nCreated new PCA: factor loadings of the first three MZ components:')
            display(df_pca)
            
            pk.dump(self.pc, open(os.path.join(save_folder, "pca.pkl"),"wb"))            
        else:            
            self.pc = pk.load(open(os.path.join(load_folder, "pca.pkl"),'rb'))
            df_pca = pd.DataFrame(self.pc.components_, columns = self.PRA_names)
            print("Loaded given PCA: factor loadings of the first three MZ components:")
            display(df_pca)           

        self.MZ_pca = df_pca
        
    def display_m_mE_CCF(self, show_figs = True):
        """
        This function displays figures cross-correlation functions between the different moods and mood events.
        
        They get saved in self.data_folder.
        """
        if not self.mE_processed:
            sys.exit("Run `load_mood_event_scores` first.")
        
        mood_folder = os.path.join(self.data_folder, "mood_analysis")
        CCF_folder  = os.path.join(mood_folder, "CCF")
        if not os.path.exists(mood_folder):
            os.mkdir(mood_folder)
        if not os.path.exists(CCF_folder):
            os.mkdir(CCF_folder)
            
        df = []
        
        for n in range(self.N):
            first_time = self.feedback_time[0,n]
            
            mood_times = first_time + np.arange(28 * 4 - 1) * 6. * 3600. * 1000. # 6h increments
            
            df_n_m = self.mood_smoother_absolute(mood_times, subject_name = self.subject_names[n])
            df_n_mE = self.mE_smoother_absolute(mood_times, subject_name = self.subject_names[n]).drop(['subject_name', 'time'], axis = 1)
            df.append(pd.concat([df_n_m, df_n_mE], axis = 1))
            
        df = pd.concat(df, axis = 0).reset_index(drop = True)
        
        shifted_df = df.groupby('subject_name').shift(1)
        
        subject_names = df['subject_name'].unique()
        
        affect_names = []
        for clm in df.columns:
            if not clm in ['subject_name', 'time']:
                affect_names.append(clm)

        for name in affect_names:
            df['d_' + name] = df[name] - shifted_df[name]
            
        for a, mood1_b in enumerate(affect_names):
            for mood2_b in affect_names[a+1:]:
                mood1 = "d_" + mood1_b
                mood2 = "d_" + mood2_b
                
                ccf_all = []

                fig_all = go.Figure()

                for n, s_name in enumerate(subject_names):
                    df_s = df[df['subject_name'] == s_name]
                    df_s = df_s.dropna()

                    mean1 = df_s[mood1].mean(); mean2 = df_s[mood1].mean()
                    std1  = df_s[mood1].std();  std2  = df_s[mood2].std()

                    lags = scipy.signal.correlation_lags(df_s.shape[0], df_s.shape[0]) * .25
                    lags_T = lags * .25
                    ccf = scipy.signal.correlate(df_s[mood1] - mean1, df_s[mood2] - mean2) / std1 / std2 / (df_s.shape[0] - np.abs(lags))
                    ccf_all.append(ccf)
                    fig_all.add_trace(go.Scatter(x = lags_T,
                                                y = ccf,
                                                name = s_name,
                                                line = {'width':.33}))


                ccf_all_a = np.array(ccf_all)
                ccf_mean = np.mean(ccf_all_a, axis = 0)
                ccf_se  = np.std(ccf_all_a, axis = 0) / np.sqrt(len(subject_names))

                fig_all.add_trace(go.Scatter(x = lags_T,
                                            y = ccf_mean,
                                            name = 'Average',
                                            #error_y=dict(type='data', array = 1.96 * ccf_se, thickness=4),                             
                                            line = {'width': 3}))
                fig_all.add_trace(go.Scatter(
                        name='Upper Bound',
                        x=lags_T,
                        y=ccf_mean + 1.96 * ccf_se,
                        mode='lines',
                        marker=dict(color="#444"),
                        line=dict(width=0),
                        showlegend=False
                    ))
                fig_all.add_trace(go.Scatter(
                        name='Lower Bound',
                        x=lags_T,
                        y=ccf_mean - 1.96 * ccf_se,
                        marker=dict(color="#444"),
                        line=dict(width=0),
                        mode='lines',
                        fillcolor='rgba(68, 68, 68, 0.3)',
                        fill='tonexty',
                        showlegend=False
                    ))
                fig_all.update_layout(title = f"CCF: {mood1} vs {mood2} - positive lag means {mood1} is later than {mood2}", xaxis_title = 'Lag (days)', yaxis_title = "ccf value", xaxis = {'range': [-7.,7.]})
                
                if show_figs:
                    fig_all.show()
                fig_all.write_image(os.path.join(CCF_folder, f"{mood1}_vs_{mood2}.jpeg"))

        
    def compute_session_trial_num(self, df, filename):
        """
        Adds a column 'session_trial' that resets every time a new session is opened. 
        This is used to know when 10-trial feedback occurs.
        """
        
        # 1) Extract the block and last_trial from the sessions table:
        query = 'SELECT block, last_trial FROM sessions;' # AND questionnaire_number>0
        
        conn = sqlite3.connect(filename, uri = True)
        df_last_trial = pd.read_sql_query(query, conn)
        conn.close()
        
        # 2) Add sessions trial to df:
        session_trial_ar = np.zeros((df.shape[0])).astype('int64') - 1
        
        s = 0 # Counter in the last-trial df
        t = 0
        session_trial = 1
        while t  < df.shape[0]:
            if s >= df_last_trial.shape[0]:
                session_trial_ar[t] = -1
                t += 1
            elif df['block'].iloc[t] > df_last_trial['block'].iloc[s]:
                    session_trial = 1
                    s += 1            
            else:
                if df['trial'].iloc[t] == df_last_trial['last_trial'].iloc[s] + 1:
                    session_trial = 1
                    s += 1

                # Test to be deleted later:
                if df['block'].iloc[t] != df_last_trial['block'].iloc[s]:
                    print("Blocks not equal in compute_session_trial!")
                    
                session_trial_ar[t] = session_trial
                session_trial += 1
                t += 1
                        
        return session_trial_ar
        
    def load_questionnaires(self, filename, subject_name):
        """ 
        Loads questionnaire info.
        
        TODO: add other questionnaires using Alon's code.
        """
        query = 'SELECT score_type, score FROM questionnaires_scores WHERE type==9'

        conn = sqlite3.connect(filename, uri = True)
        df1 = pd.read_sql_query(query, conn)
        conn.close()
        
        return pd.DataFrame({df1['score_type'].iloc[q]: [df1['score'].iloc[q]] for q in range(df1.shape[0])}, index = [subject_name])
        
    def compute_EV_and_best(self):
        """
        Computes the Expected Value and (objective) best stimulus for every trial. Best is defined as having the highest Expected Value, and is np.nan when equal.
        """
        self.EV = self.p_R - self.p_Pun   # (T, 2, N) Expected values of the stims
        
        self.equal_stim = pd.DataFrame(np.abs(self.EV[:,1,:] - self.EV[:,0,:]) < 1e-10)         # (T, N) dataframe, Indicator of where the stims are equally good for every (t,n)
        # where puts a nan where the EVs are equal:
        self.best_stim  = pd.DataFrame(self.EV[:,1,:] > self.EV[:,0,:] + 1e-10).where(~self.equal_stim) # (T, N) dataframe, Indicator of which stim is best
    
    def compute_acc(self):
        """
        Computes the (objective) accuracy of the trials. 1 if the objectively best stimulus was chosen. 0 if not. np.nan in case the EVs are the same.
        """
        with np.errstate(invalid='ignore'):
            self.acc = (np.abs(self.C_st - self.best_stim.values) < 1e-10).astype('float')   # (T, N)
        self.acc[self.best_stim.isna()] = np.nan                        # (T, N)
        
        
        
    def compute_acc_exp(self, arbitrary_threshold = .1):
        """
        Computes the experienced Expected Value of the stimuli and their accuracy according to these.
        """
        self.EV_exp  = np.full(self.stims.shape, np.nan)
        
        for n in range(self.N):
            EV_running = np.full((self.Nc[n]), np.nan)
            N_running  = np.zeros((self.Nc[n])).astype('int32')
            
            for t in range(self.T[n]):
                C = self.C[t,n]
                if self.stims[t,:,n][1]!=-1000: #skip accept-reject fmri blocks
                    # Store the EVs:
                    self.EV_exp[t,:,n] = EV_running[self.stims[t,:,n]]
                    
                    # Update the running number of choices and EV:
                    if self.feedback[t,n] > .5:
                        N_running[C] += 1
                        if not np.isnan(EV_running[C]):
                            EV_running[C] += (self.o[t,n] - EV_running[C]) / N_running[C]
                        else:
                            EV_running[C] = self.o[t,n]
                                
        # Compute the acc_exp:
        # left unchosen, right unchosen, undetermined, practice stim, practice stim, chose right correctly, chose left correctly, chose left incorrectly, chose right incorrectly
        conds = [np.isnan(self.EV_exp[:,0,:]), \
                 np.isnan(self.EV_exp[:,1,:]), \
                 np.abs(self.EV_exp[:,0,:] - self.EV_exp[:,1,:]) <= arbitrary_threshold, \
                 self.stims[:,0,:] < self.min_stim,
                 self.stims[:,1,:] < self.min_stim,
                 (self.EV_exp[:,1,:] > self.EV_exp[:,0,:] + arbitrary_threshold) & (self.C_st > .5), \
                 (self.EV_exp[:,0,:] > self.EV_exp[:,1,:] + arbitrary_threshold) & (self.C_st < .5), \
                 (self.EV_exp[:,1,:] > self.EV_exp[:,0,:] + arbitrary_threshold) & (self.C_st < .5), \
                 (self.EV_exp[:,0,:] > self.EV_exp[:,1,:] + arbitrary_threshold) & (self.C_st > .5)] 
                 
        choices = [np.nan, np.nan, np.nan, np.nan, np.nan, 1., 1., 0., 0.]
        
        self.acc_exp = np.select(conds, choices)
        
        
    def compute_session_times(self):
        """
        Computes 'session' for every trial, defined as the current block minus the minimum block divided by 2 and rounded down. Typically this is an indicator of the day or session.
        self.session_times is an array with the beginning time of every session. 
        """
        self.session = -np.ones_like(self.block)
    
        self.session_times = [[] for n in range(self.N)]
        
        for n in range(self.N):
            block = self.block[0,n]
            self.session_times[n].append(self.feedback_time[0,n] / 3600. / 1000. / 24.)
            self.session[0,n] = 0
            
            for t in range(1,self.T[n]):
                if self.block[t,n] == block + 2:
                    block = self.block[t,n]
                    self.session_times[n].append(self.feedback_time[t,n] / 3600. / 1000. / 24.)
                self.session[t,n] = int((self.block[t,n] - self.block[0,n]) / 2)
            self.session_times[n] = np.array(self.session_times[n])

  
    def compute_learning_days(self, min_block = 6):
        """
        Computes for every stimulus the two learning blocks and days.
        
        Day is defined by the session.
        """
        
        learning_blocks = (np.zeros((np.max(self.Nc), 2, self.N)) - 1).astype('int32')
        learning_days   = (np.zeros((np.max(self.Nc), 2, self.N)) - 1).astype('int32')
        
        for n in range(self.N):
            for t in range(self.T[n]):
                if self.trial[t,n] < 10:
                    if self.feedback[t,n] > .5:
                        block = self.block[t,n]
                        day = self.session[t,n]
                        st1 = self.stims[t,0,n]; st2 = self.stims[t,1,n]
                                               
                        for st in [st1, st2]:
                            if st>-10: #ignore accept-reject trials
                                if learning_blocks[st, 0, n] == -1:
                                    learning_blocks[st, 0, n] = block
                                    learning_days[st,0,n] = day
                                elif learning_blocks[st, 0, n] != block:
                                    learning_blocks[st, 1, n] = block
                                    learning_days[st, 1, n] = day
                                
        self.learning_blocks = learning_blocks
        #self.learning_blocks[-3:,1,:] = np.max(self.block)
        
        self.learning_days = learning_days
        #self.learning_days[-3:,1,:] = np.max(self.session)   
        
    def compute_learning_days_nf_num_appearance(self, min_stim = 18, min_block = 6): 
        """
        Computes for every day the number of NF trials that have a stimulus that was learnt on that day. Day is defined by session.
        """
        self.learning_days_nf_num_appearance = np.zeros((np.max(self.session) + 1, self.N))
        
        for n in range(self.N):
            for t in range(self.T[n]):
                if self.feedback[t,n] < .5 and self.C[t,n] >= min_stim:
                    ldays = self.learning_days[self.C[t,n], :, n]
                    self.learning_days_nf_num_appearance[ldays, n] += 1  
        
    def compute_new_stimuli_per_block(self):
        """
        Creates self.new_stimuli, a list of lists, with self.new_stimuli[n][b] the new stimuli
        in block b for subject n.
        
        Also creates self.new, a (T, N) numpy array, with a 1 if the trial has 2 stimuli that are new to this block,
        and 0 otherwise.
        
        Finally, creates self.num_new, a (num_blocks, N) int-numpy array indicating the number of new stimuli per block.
        """
        self.new_stimuli = [] 
        T = np.max(self.T)
        self.new = np.zeros((T, self.N)).astype('int64')
        for n in range(self.N):
            self.new_stimuli.append([])
            seen_stimuli = []
            new_stimuli_b = []
             
            t = 0         
            while t < self.T[n] - 1:
                b = self.block[t,n]
                stim1 = self.stims[t,0,n]
                stim2 = self.stims[t,1,n]
                if stim1 not in seen_stimuli:
                    seen_stimuli.append(stim1)
                    new_stimuli_b.append(stim1)
                if stim2 not in seen_stimuli:
                    seen_stimuli.append(stim2)
                    new_stimuli_b.append(stim2)
                    
                if stim1 in new_stimuli_b and stim2 in new_stimuli_b:
                    self.new[t,n] = 1
                
                if self.block[t+1,n] != b:
                    self.new_stimuli[n].append(new_stimuli_b)
                    new_stimuli_b = []
                    
                    # Deal with lapses in blocks:
                    b += 1
                    while self.block[t+1,n] > b:
                        self.new_stimuli[n].append([])
                        b+= 1
            
                t+=1
                
            self.new_stimuli[n].append(new_stimuli_b)
            
        self.num_new = np.zeros((np.max(self.block) - np.min(self.block) + 1, self.N)).astype('int64')
        for n in range(self.N):
            for b in range(self.block[self.T[n] - 1, n] - self.block[0,n]):
                self.num_new[b,n] = len(self.new_stimuli[n][b])
                
    def compute_time_since_last_rep(self):
        """
        For every subject:
        - At every trial it computes the time since the last repetition of the stimuli that appear.
          The unit is days since last rep, not rounded (float).
        It stores this in self.time_since_last_rep, a (T, 2, N) numpy array
        """
        self.time_since_last_rep = np.zeros((np.max(self.T), 2, self.N))

        for n in range(self.N):
            st1 = self.stims[:,0,n]
            st2 = self.stims[:,1,n]
            feedback_time = self.feedback_time[:,n]

            stim_last_time = {}

            # Compute the time since the last presentation:
            for t in range(self.T[n]):
                if st1[t] in stim_last_time.keys():
                    self.time_since_last_rep[t,0,n] = feedback_time[t] - stim_last_time[st1[t]]
                stim_last_time[st1[t]] = feedback_time[t]

                if st2[t] in stim_last_time.keys():
                    self.time_since_last_rep[t,1,n] = feedback_time[t] - stim_last_time[st2[t]]
                stim_last_time[st2[t]] = feedback_time[t]

        self.time_since_last_rep = self.time_since_last_rep / (1000. * 3600. * 24.)

    def compute_num_blocks_since_last_block(self):
        """
        For every subject:
        - At every trial it computes the number of blocks since the last repetition of the stimulus.
          It stores this in self.num_blocks_since_last_block: (T, 2, N) numpy array
        """
        self.num_blocks_since_last_block = np.zeros((np.max(self.T), 2, self.N))

        for n in range(self.N):
            st1 = self.stims[:,0,n]
            st2 = self.stims[:,1,n]
            block = self.block[:,n]

            stim_last_block = {}

            # Compute the time since the last presentation:
            for t in range(self.T[n]):
                if st1[t] in stim_last_block.keys():
                    self.num_blocks_since_last_block[t,0,n] = block[t] - stim_last_block[st1[t]]
                stim_last_block[st1[t]] = block[t]

                if st2[t] in stim_last_block.keys():
                    self.num_blocks_since_last_block[t,1,n] = block[t] - stim_last_block[st2[t]]
                stim_last_block[st2[t]] = block[t]

        self.num_blocks_since_last_block = self.num_blocks_since_last_block   
        
    def compute_last_feedback_time(self):
        """
        Creates a pandas DataFrame with indices the stimuli ID and values the last feedback_time
        for the stimuli on their feedback trials.
        """
        self.last_feedback_time = []
        
        for n in range(self.N):
            df = self.create_subject_pre_df(n = n)
            
            df_feedback = df[df['feedback'] > .5]
            
            last_feedback_time1 = df_feedback.groupby('stim1')['feedback_time'].last()
            last_feedback_time2 = df_feedback.groupby('stim2')['feedback_time'].last()
            self.last_feedback_time.append(pd.concat([last_feedback_time1, last_feedback_time2], axis = 1).max(axis = 1))
            
        self.last_feedback_time = pd.concat(self.last_feedback_time, axis = 1)
        self.last_feedback_time.columns = self.subject_names

    def compute_last_no_feedback_time(self):
        """
        Creates a pandas DataFrame with indices the stimuli ID and values the last feedback_time
        for the stimuli on their feedback trials.
        """
        self.last_no_feedback_time = []
        
        for n in range(self.N):
            df = self.create_subject_pre_df(n = n)
            
            df_no_feedback = df[df['feedback'] < .5]
            
            last_no_feedback_time1 = df_no_feedback.groupby('stim1')['feedback_time'].last()
            last_no_feedback_time2 = df_no_feedback.groupby('stim2')['feedback_time'].last()
            self.last_no_feedback_time.append(pd.concat([last_no_feedback_time1, last_no_feedback_time2], axis = 1).max(axis = 1))
            
        self.last_no_feedback_time = pd.concat(self.last_no_feedback_time, axis = 1)
        self.last_no_feedback_time.columns = self.subject_names

    def compute_first_no_feedback_time(self):
        """
        Creates a pandas DataFrame with indices the stimuli ID and values the last feedback_time
        for the stimuli on their feedback trials.
        """
        self.first_no_feedback_time = []
        
        for n in range(self.N):
            df = self.create_subject_pre_df(n = n)
            
            df_no_feedback = df[df['feedback'] < .5]
            
            first_no_feedback_time1 = df_no_feedback.groupby('stim1')['feedback_time'].first()
            first_no_feedback_time2 = df_no_feedback.groupby('stim2')['feedback_time'].first()
            self.first_no_feedback_time.append(pd.concat([first_no_feedback_time1, first_no_feedback_time2], axis = 1).min(axis = 1))
            
        self.first_no_feedback_time = pd.concat(self.first_no_feedback_time, axis = 1)
        self.first_no_feedback_time.columns = self.subject_names        
         
    def compute_last_learning_block(self):
        """
        Creates a pandas DataFrame with indices the stimuli ID and values the last learning block
        for the stimuli on their feedback trials.
        """
        self.last_learning_block = []
        
        for n in range(self.N):
            df = self.create_subject_pre_df(n = n)
            
            df_feedback = df[df['feedback'] > .5]
            
            last_lblock_time1 = df_feedback.groupby('stim1')['block'].last()
            last_lblock_time2 = df_feedback.groupby('stim2')['block'].last()
            self.last_learning_block.append(pd.concat([last_lblock_time1, last_lblock_time2], axis = 1).max(axis = 1))
            
        self.last_learning_block = pd.concat(self.last_learning_block, axis = 1)
        num_nan_rows = np.arange(self.last_learning_block.index.min()).astype('int32')
        nan_df = pd.DataFrame(index = num_nan_rows, columns = self.subject_names, data = np.nan)
        self.last_learning_block.columns = self.subject_names
        self.last_learning_block = pd.concat([nan_df, self.last_learning_block], axis = 0)
        
    def compute_last_test_block(self):
        """
        Creates a pandas DataFrame with indices the stimuli ID and values the last learning block
        for the stimuli on their feedback trials.
        """
        self.last_test_block = []
        
        for n in range(self.N):
            df = self.create_subject_pre_df(n = n)
            
            df_no_feedback = df[df['feedback'] < .5]
            
            last_tblock_time1 = df_no_feedback.groupby('stim1')['block'].last()
            last_tblock_time2 = df_no_feedback.groupby('stim2')['block'].last()
            self.last_test_block.append(pd.concat([last_tblock_time1, last_tblock_time2], axis = 1).max(axis = 1))
            
        self.last_test_block = pd.concat(self.last_test_block, axis = 1)
        num_nan_rows = np.arange(self.last_test_block.index.min()).astype('int32')
        nan_df = pd.DataFrame(index = num_nan_rows, columns = self.subject_names, data = np.nan)
        self.last_test_block.columns = self.subject_names
        self.last_test_block = pd.concat([nan_df, self.last_test_block], axis = 0)
        
    def compute_mean_test_block(self):
        """
        Creates a pandas DataFrame with indices the stimuli ID and values the last learning block
        for the stimuli on their feedback trials.
        """
        self.mean_test_block = []
        
        for n in range(self.N):
            df = self.create_subject_pre_df(n = n)
            
            df_no_feedback = df[df['feedback'] < .5]
            
            mean_tblock_time1 = df_no_feedback.groupby('stim1')['block'].mean()
            mean_tblock_time2 = df_no_feedback.groupby('stim2')['block'].mean()
            self.mean_test_block.append(pd.concat([mean_tblock_time1, mean_tblock_time2], axis = 1).mean(axis = 1))
            
        self.mean_test_block = pd.concat(self.mean_test_block, axis = 1)
        num_nan_rows = np.arange(self.mean_test_block.index.min()).astype('int32')
        nan_df = pd.DataFrame(index = num_nan_rows, columns = self.subject_names, data = np.nan)
        self.mean_test_block.columns = self.subject_names
        self.mean_test_block = pd.concat([nan_df, self.mean_test_block], axis = 0)
        
    def compute_first_test_block(self):
        """
        Creates a pandas DataFrame with indices the stimuli ID and values the last learning block
        for the stimuli on their feedback trials.
        """
        self.first_test_block = []
        
        for n in range(self.N):
            df = self.create_subject_pre_df(n = n)
            
            df_no_feedback = df[df['feedback'] < .5]
            
            first_tblock_time1 = df_no_feedback.groupby('stim1')['block'].first()
            first_tblock_time2 = df_no_feedback.groupby('stim2')['block'].first()
            self.first_test_block.append(pd.concat([first_tblock_time1, first_tblock_time2], axis = 1).min(axis = 1))
            
        self.first_test_block = pd.concat(self.first_test_block, axis = 1)
        num_nan_rows = np.arange(self.first_test_block.index.min()).astype('int32')
        nan_df = pd.DataFrame(index = num_nan_rows, columns = self.subject_names, data = np.nan)
        self.first_test_block.columns = self.subject_names
        self.first_test_block = pd.concat([nan_df, self.first_test_block], axis = 0)
        
    def compute_num_nopractice(self):
        """
        Compute the number of trials with no-practice stimuli on feedback and no-feedback trials and stores these
        in self.num_nopractice_f and self.num_nopractice_nf. 
        
        These correspond to the number of trials being used in the llik evaluations
        """

        self.num_nopractice_f = []
        self.num_nopractice_nf = []
        for n in range(self.N):
            df = self.create_subject_pre_df(n=n)
            
            df_f = df[df['feedback'] > .5]
            df_nf = df[df['feedback'] < .5]
            result = []
            
            for df_x in [df_f,df_nf]:
                result.append(np.sum(np.logical_and(df_x['stim1'] > self.min_stim, df_x['stim2'] > self.min_stim)))
            
            self.num_nopractice_f.append(result[0])
            self.num_nopractice_nf.append(result[1])
            
        self.num_nopractice_f = np.array(self.num_nopractice_f)
        self.num_nopractice_nf = np.array(self.num_nopractice_nf)
        self.num_datapoints    = self.num_nopractice_f + self.num_nopractice_nf

    def compute_learning_mood(self):
        """
        For every stimulus, computes the mood during learning as follows:
          the function takes all learning trials where the stimulus appears in either first or second position, and averages their mood.
          There are at most two different moods during learning, as the stimuli only appears on feedback trials on maximally two days.
        """
        self.learning_mood_stim = [np.zeros((np.max(self.Nc), self.N)) for e in range(4+5+3)]
        
        T = np.max(self.T)
        self.learning_mood = [np.zeros((T, 2, self.N)) for e in range(4+5+3)]
       
        for n in range(self.N):
            df = self.create_subject_pre_df(n=n)
            df = df[df['feedback'] > .5]
            
            
            
            df1 = df.groupby('stim1')[self.mood_names]
            df2 = df.groupby('stim2')[self.mood_names]
            
            stim_N = df1.count() + df2.count()
            mean_mood = (df1.sum() + df2.sum()) / stim_N         
            mean_mood = mean_mood.iloc[np.where(mean_mood.index>-10)] #remove stim -1000 (accecpt-reject trials)
            for e in range(4+5+3):
                self.learning_mood_stim[e][mean_mood.index, n] = mean_mood[self.mood_names[e]]            
                self.learning_mood[e][:,0,n] = self.learning_mood_stim[e][self.stims[:,0,n],n]
                not_accept_reject= np.where(self.stims[:,1,n] != -1000) #omit accept-reject fmri blocks
                self.learning_mood[e][not_accept_reject,1,n] = self.learning_mood_stim[e][self.stims[not_accept_reject,1,n],n]
    
    def compute_num_choice(self):
        """
        Computes how many times the stimuli have been chosen before. Stores this in 
          self.num_choice: (T,2,N) int-array
          
        Also computes the number of appearances and stores it in:
          self.num_appear: (T,2,N) int-array
          self.num_nf_appear for NF trials before 
          
        Does the same for the number of feedback rewards and punishments before and stores these respectively in:
            self.num_r
            self.num_pun
        
        """
        T = np.max(self.T)
        self.num_choice = np.zeros((T,2,self.N)).astype('int64')
        self.num_r    = np.zeros((T,2,self.N)).astype('int64')
        self.num_pun  = np.zeros((T,2,self.N)).astype('int64')
        self.num_appear = np.zeros((T,2,self.N)).astype('int64')
        self.num_nf_appear = np.zeros((T,2,self.N)).astype('int64')
        
        for n in range(self.N):
            num_choice_running = np.zeros((self.Nc[n]))
            num_r_running      = np.zeros((self.Nc[n]))
            num_pun_running      = np.zeros((self.Nc[n]))
            num_appear_running = np.ones((self.Nc[n]))
            num_nf_appear_running = np.ones((self.Nc[n]))
            
            for t in range(self.T[n]):
                if self.stims[t,:,n][1]!=-1000: #skip accept-reject fmri blocks
                    self.num_choice[t,:,n] = num_choice_running[self.stims[t,:,n]]
                    self.num_r[t,:,n]      = num_r_running[self.stims[t,:,n]]
                    self.num_pun[t,:,n]    = num_pun_running[self.stims[t,:,n]]
                    
                    num_choice_running[self.C[t,n]] += 1
                    if self.feedback[t,n] > .5:
                        if self.o[t,n] > .5:
                            num_r_running[self.C[t,n]] += 1
                        elif self.o[t,n] < -.5:
                            num_pun_running[self.C[t,n]] += 1
                    
                    self.num_appear[t,:,n] = num_appear_running[self.stims[t,:,n]]
                    
                    num_appear_running[self.stims[t,:,n]] += 1
                    
                    if self.feedback[t,n] < .5:
                        self.num_nf_appear[t,:,n] = num_nf_appear_running[self.stims[t,:,n]]
                        
                        num_nf_appear_running[self.stims[t,:,n]] += 1
                    
        
    def compute_stim_types(self, subject_names = None, order_names = False):
        """
        Given the stims and their probabilities, computes the coin types.
        0 for 70% reward
        1 for 70% punishment
        2 for 70% neutral
        3 for 50%/0%/50%
        """
        subject_names, columns = self.subject_names_to_columns(subject_names, False)

        N = len(subject_names)
        
        # 0: 70% reward, 1: 70& punishment, 2: 70% neutral, 3: 50% reward/punishment,
        self.stim_types = - np.ones((np.max(self.T), 2, N))
        
        reward_stims = (np.abs(self.p_R[:,:,columns] - .7) <1e-10)
        pun_stims = (np.abs(self.p_Pun[:,:,columns] - .7) <1e-10)
        neutral_stims = (np.abs(self.p_R[:,:,columns] -.15) < 1e-10) & (np.abs(self.p_Pun[:,:,columns] -.15) < 1e-10)
        wild_stims = np.abs(self.p_R[:,:,columns] - .5) < 1e-10
        
        self.stim_types[reward_stims] = 0
        self.stim_types[pun_stims] = 1
        self.stim_types[neutral_stims] = 2
        self.stim_types[wild_stims] = 3
        
        self.chosen_stim_type = - np.ones((np.max(self.T), N))
        for n in range(N):
            self.chosen_stim_type[:self.T[n],n] = self.stim_types[np.arange(self.T[n]),self.C_st[:self.T[n],n],n * np.ones(self.T[n]).astype('int')]
               
    def compute_stim_stage(self):
        """
        For every trial, computes the stimulus stage of every stimulus:
            - 0 if first learning blokc
            - 1 if second learning block
            - 2 if no-feedback
        """
        
        self.stim_stage = np.zeros((np.max(self.T), 2, self.N)).astype('int64')
        
        for n in range(self.N):
            first_lblock = np.zeros(self.Nc[n]) - 1
            
            for t in range(self.T[n]):
                st0 = self.stims[t,0,n]
                st1 = self.stims[t,1,n]
                if st1!=-1000: #skip accept-reject fmri blocks
                    if first_lblock[st0] < 0:
                        first_lblock[st0] = self.block[t,n]
                    elif self.feedback[t,n] > .5 and self.block[t,n] > first_lblock[st0]:
                        self.stim_stage[t,0,n] = 1
                    elif self.feedback[t,n] < .5:
                        self.stim_stage[t,0,n] = 2
                          
                    if first_lblock[st1] < 0:
                        first_lblock[st1] = self.block[t,n]
                    elif self.feedback[t,n] > .5 and self.block[t,n] > first_lblock[st1]:
                        self.stim_stage[t,1,n] = 1
                    elif self.feedback[t,n] < .5:
                        self.stim_stage[t,1,n] = 2
                                        
    def compute_num_learning_blocks(self):
        """        
        Computes the number of block with learning (at least one feedback trial),
        in self.num_learning_blocks, as an (N)-numpy array.
        """
        
        self.num_learning_blocks = np.zeros((self.N)).astype('int64')
        for n in range(self.N):
            df_n = pd.DataFrame({'block': self.block[:,n], 'feedback': self.feedback[:,n]})
            num_feedback_per_block_n = df_n.groupby('block').sum()#transform(lambda x: x.sum())
            self.num_learning_blocks[n] = np.sum(num_feedback_per_block_n.values > 0)
            #self.num_learning_blocks[n] = np.sum(num_feedback_per_block_n > 0) #original, caused warnings message       
            

            
    ### Auxiliary functions: ###
    def gen_outcome(self, p_R_st, p_Pun_st, C_st):
        """
        Generate outcomes according to the EMA environment.
        
        - R_st: (2,S)-array, the reward probabilities of the current stimuli, order (stim1, stim2).
        - Pun_st: (2,S)-array, the punishment probabilities of the current stimuli, order (stim1, stim2).
        - C_st: (S) numpy array, the choices (0 for stim1, 1 for stim2).
        
        Returns:
        - (S) numpy array of outcomes. -1 for punishment, 0 for neutral, 1 for reward
        """
        S = C_st.shape[0]
        R = p_R_st[C_st, np.arange(S)][None,:]; Pun = p_Pun_st[C_st, np.arange(S)][None, :]  # (1, S)
        p = np.concatenate((Pun, 1. - Pun - R, R), axis = 0)                                 # (3, S)
        return multinomial_vect(p) - 1

    def subject_names_to_columns(self, subject_names, order_names):
        """      
        Arguments:
        - subject_names: list of strings
        - order_names: boolean
        
        Given subject_names, returns a tuple, where:
            The first value is the list of all the internal subject names if subject_names is None; it is subject_names if it is not. It's ordered alphabetically if order_names == True.
            The second value is the corresponding list of columns in the internal data, in the same order as the returned subject names list.
        """
        
        if subject_names is None:
            subject_names = self.subject_names
        if not type(subject_names) == list:
            sys.exist("subject_names_to_columns > `subject_names` (first argument) should be a list of subject names or None.")
        if order_names:
            subject_names = sorted(subject_names)
        columns = []
        for s_name in subject_names:
            columns.append(self.subject_names.index(s_name))
            
        return (subject_names, columns)
            
    ### Simulation Function: ###
    def simulate_data(self, style, P_true, debug = False, verbose = True, P_aux = None):
        """
        This function can be used to simulate data. IMPORTANT WARNING: it doesn't allow to simulate all styles that are present in the llik function. 
        It could be made that way by incoporating the simulation as if statements inside the llik function, but it would make it more messy.
        As a general rule, if you try to run it and it crashes, the llik function you're using is not incorporated.
        
        style  : dictionary with (key, value) pairs:
            'Q_style'
            'choice_style'
            'R_style'
            Optionally 'T_style'
        P_true : (N, .) dataframe with the relevant parameters in the columns, N subject rows.
        P_aux  : (N, .) dataframe with the auxiliary parameters.
        
        TODO: Two Q process simulation.
        """
        if P_true.shape[0] != self.N:
            sys.exit("The number of parameters don't correspond to the number of state trajectories!")
                 
        if 'T_style' in style:
            T_style = style['T_style']
            self.T_style = style['T_style']
        else:
            T_style = None
            self.T_style = None

        if 'Q_style_2' in style:
            sys.exit("No Q_style_2 yet!")

        if 'dir' in style['Q_style']:
            sys.exit("No dir-type Q_style yet!")
            
        self.Q_style = style['Q_style']
        self.choice_style = style['choice_style']
        self.R_style = style['R_style']
        self.P_true = pd.DataFrame(P_true) # copy the dataframe
        
        self.P_aux_true = P_aux
        
        if verbose:
            print(f"\nSimulating data with style:")
            print(f"- Q_style     : {self.Q_style}\n- choice_style: {self.choice_style}\n- R_style     : {self.R_style}\n- T_style     : {T_style}\n")

        T = np.max(self.T)
        self.C      = - np.ones((T, self.N)).astype('int64'); 
        self.C_st   = - np.ones((T, self.N)).astype('int64'); 
        self.o      = - 2.*np.ones((T, self.N)).astype('int64');                
        
        if len(self.subject_names) != self.N:
            self.subject_names = [str(n) for n in np.arange(self.N).astype('int64')]
                
        # Number of choices:
        Nc = np.max(self.Nc)
        N  = self.N
        
        if style['R_style'] != 'objective':
            o_subj = True
        else:
            o_subj = None
                 
        tm = time.time()
        for n in range(self.N):
            # The Q_table for the N subjects
            Q = np.zeros((Nc, 1))
            Q_st = np.zeros((2, 1))
            
            # For bayes updates, count tables:
            N_table = np.zeros((Nc, 1))
            
            old_block = self.block[0,n]
            new_block = False
            
            P_n = P_true.iloc[[n],:]
            if P_aux is not None:            
                P_aux_n = P_aux.iloc[[n],:]
            else:
                P_aux_n = None
        
            for t in range(self.T[n]):
                # The Q-values of the current stimuli:
                st = self.stims[t,:,n].reshape((2,1))                # (2,1) 
                Q_st[0,0] = Q[st[0,0], 0]  # (N)
                Q_st[1,0] = Q[st[1,0], 0]  # (N)
                
                feedback = self.feedback[t,n]
                block = self.block[t,n]
                session = self.session[t,n]
                
                # T update:
                if T_style is not None:
                    print('No diga nada.')
                    tslr = self.time_since_last_rep[t,:,n] 
                    trial_time = self.feedback_time[t,n]
                    if block > old_block:
                        new_block = True
                        block_time = self.feedback_time[t,n]
                    T_update_RL_model(Q, st, tslr, trial_time - old_trial_time, feedback, T_style, P_n, P_aux_n, block = block, o_subj = o_subj, block_time = block_time, old_block_time = old_block_time, session = session)
                    if new_block:
                        old_block_time = block_time
                        old_block = block
                    old_trial_time = trial_time                    
                
                # Generate choices:
                C_st = np.random.binomial(1,np.exp(choice_logprobs(Q_st, self.choice_style, P_n, feedback = feedback, P_aux = P_aux_n)))     # (1)
                C    = st[C_st,[0]]                                # (1)
                
                # Generate outcomes:
                p_R_st = self.p_R[t,:,n].reshape((2,1))                                     # (2, 1)
                p_Pun_st = self.p_Pun[t,:,n].reshape((2,1))                                 # (2, 1)
                o = self.gen_outcome(p_R_st, p_Pun_st, C_st)                                # (1)
                
                self.o[t,n]     = o
                
                # Change rewards if subjective reward:
                if self.R_style != 'objective':
                    o = subjective_o(o, self.R_style, P_n, block = block, P_aux = P_aux_n, session = session)
                
                # Update the N-table:
                N_table[C, 0] = N_table[C, 0] + 1
                
                # Update Q:
                update_Q(Q, C, o, self.Q_style, P_n, N_table = N_table, feedback = feedback, P_aux = P_aux_n)
                
                # Save:
                self.C_st[t, n] = C_st
                self.C[t,n]     = C          
                
                # Debug:
                if debug and t < 48:
                    print(Q[:3, :])
                if debug and t == 47:
                    print(N_table[:3,:])
        print(f"Time to simulate {self.N} subjects over {self.T} timesteps:\n==> {time.time() - tm:.3f}s.")
          
        self.compute_acc()

            
    ### Diagnostic functions: ###
    def create_subject_df(self, n = None, subject_name = ""):
        """
        Returns a dataframe with the data of the 'clm'-th subject or the subject indicated by `subject_name`. 
        Pandas allows more efficient analysis code-wise.
        
        Arguments:
        - n           : int, the indicator of the subject in the data (the column of all the arrays that are stored inside self).
        - subject_name: string, the name of the subject.
        """
        
        if (n is None and subject_name == "") or (n is not None and subject_name != ""):
            sys.exit("Specify one and only one of either the subject_name or the column n.")
        
        if subject_name != "":
            names, columns = self.subject_names_to_columns([subject_name], False)
            n = columns[0]
            name = names[0]
        else:
            name = self.subject_names[n]
            
        T = self.T[n]
            
        df = pd.DataFrame({
            'subject_name': name,   
            'session': self.session[:T,n],
            'block': self.block[:T,n],
            'trial': self.trial[:T,n],
            'session_trial': self.session_trial[:T,n],
            'stim1': self.stims[:T,0,n],
            'stim2': self.stims[:T,1,n],
            'feedback': self.feedback[:T,n],
            'C_st' : self.C_st[:T,n],
            'C'    : self.C[:T,n],
            'o'    : self.o[:T,n],
            'stim_time': self.stim_time[:T,n],
            'choice_time': self.choice_time[:T,n],
            'feedback_time': self.feedback_time[:T,n],
            'acc'  : self.acc[:T,n],
            'acc_exp': self.acc_exp[:T,n],
            'EV_exp1': self.EV_exp[:T,0,n],
            'EV_exp2': self.EV_exp[:T,1,n],
            'num_choice1': self.num_choice[:T,0,n],
            'num_choice2': self.num_choice[:T,1,n],
            'num_r1': self.num_r[:T,0,n],
            'num_r2': self.num_r[:T,1,n],
            'num_pun1': self.num_pun[:T,0,n],
            'num_pun2': self.num_pun[:T,1,n],
            'num_appear1': self.num_appear[:T,0,n],
            'num_appear2': self.num_appear[:T,1,n],
            'num_nf_appear1': self.num_nf_appear[:T,0,n],
            'num_nf_appear2': self.num_nf_appear[:T,1,n],
            'time_since_last_rep1': self.time_since_last_rep[:T,0,n],
            'time_since_last_rep2': self.time_since_last_rep[:T,1,n],
            'stim_stage1': self.stim_stage[:T,0,n],
            'stim_stage2': self.stim_stage[:T,1,n],
            'type1': self.stim_types[:T,0,n],
            'type2': self.stim_types[:T,1,n],
            'type_chosen': self.chosen_stim_type[:T,n],
            'p_r1' : self.p_R[:T,0,n],
            'p_pun1':self.p_Pun[:T,0,n],
            'p_r2' : self.p_R[:T,1,n],
            'p_pun2':self.p_Pun[:T,1,n],
            'best_stim': self.best_stim.iloc[:T,n]
        })
        
        
        if self.with_moods:
            df_mood = pd.DataFrame({self.mood_names[e] : self.mood[e][:T,n] for e in range(len(self.mood_names))})
            df_learning_mood1 = pd.DataFrame({f'learning_{self.mood_names[e]}1' : self.learning_mood[e][:T,0,n] for e in range(len(self.mood_names))})
            df_learning_mood2 = pd.DataFrame({f'learning_{self.mood_names[e]}2' : self.learning_mood[e][:T,1,n] for e in range(len(self.mood_names))})
            df_mood = pd.concat([df_mood, df_learning_mood1, df_learning_mood2], axis = 1)
            df_mood.index = df.index
            
            df = pd.concat((df,df_mood), axis = 1)
            
        if self.mE_processed:
            df_mE = pd.DataFrame({'agg_affect_score' : self.smoothed_mE[0][:T,n],
                                  'absagg_affect_score' : self.smoothed_mE[1][:T,n],
                                  'sum_affect_score'    : self.smoothed_mE[2][:T,n],
                                  'abssum_affect_score' : self.smoothed_mE[3][:T,n]})
            df_mE.index = df.index
            df = pd.concat((df,df_mE), axis = 1)
            

        return df
        
    def create_big_subject_df(self, subject_names = None):
        """
        Returns a Pandas DataFrame with the data of all subjects in the list `subject_names`.
        
        Arguments:
        - subject_names: list of subject names. E.g. ['219', '221', ...].
        """
        
        names, columns = self.subject_names_to_columns(subject_names, False)
        
        dfs = []
        
        for n in columns:
            dfs.append(self.create_subject_df(n=n))
            
        return pd.concat(dfs, axis = 0)
    
    def create_subject_pre_df(self, n = None, subject_name = ""):
        """
        Creates a dataframe with the data of the 'clm'-th subject. This makes analysis more efficient using pandas.
        
        - n: int, the indicator of the subject in the data (the column of all the arrays).
        
        This is a version that is used when loading subjects, to construct some of the columns that
        are used in 'create_subject_df'.
        """
        
        if (n is None and subject_name == "") or (n is not None and subject_name != ""):
            sys.exit("Specify one and only one of either the subject_name or the column n.")
        
        if subject_name != "":
            names, columns = self.subject_names_to_columns([subject_name], False)
            n = columns[0]
            
        T = self.T[n]
            
        df = pd.DataFrame({
            'block': self.block[:T,n],
            'stim1': self.stims[:T,0,n],
            'stim2': self.stims[:T,1,n],
            'feedback': self.feedback[:T,n],
            'C_st' : self.C_st[:T,n],
            'C'    : self.C[:T,n],
            'o'    : self.o[:T,n],
            'feedback_time': self.feedback_time[:T,n],
            'acc'  : self.acc[:T,n],
            'time_since_last_rep1': self.time_since_last_rep[:T,0,n],
            'time_since_last_rep2': self.time_since_last_rep[:T,1,n],
            'stim_stage1': self.stim_stage[:T,0,n],
            'stim_stage2': self.stim_stage[:T,1,n],
            'type1': self.stim_types[:T,0,n],
            'type2': self.stim_types[:T,1,n],
            'p_r1' : self.p_R[:T,0,n],
            'p_pun1':self.p_Pun[:T,0,n],
            'p_r2' : self.p_R[:T,1,n],
            'p_pun2':self.p_Pun[:T,1,n],
            'best_stim': self.best_stim.iloc[:T,n],
        })#.reset_index(drop=True)
        
        if self.with_moods:
            df_mood = pd.DataFrame({self.mood_names[e] : self.mood[e][:T,n] for e in range(len(self.mood_names))})
            df_mood.index = df.index
            
            df = pd.concat((df,df_mood), axis = 1)
                    
        return df
        
    def create_subject_df_state(self, n = None, subject_name = ""):
        """
        Creates a dataframe with the data of the 'clm'-th subject. This makes analysis more efficient using pandas.
        
        - n: int, the indicator of the subject in the data (the column of all the arrays).
        """
        
        if (n is None and subject_name == "") or (n is not None and subject_name != ""):
            sys.exit("Specify one and only one of either the subject_name or the column n.")
        
        if subject_name != "":
            names, columns = self.subject_names_to_columns([subject_name], False)
            n = columns[0]
            
        df = pd.DataFrame({
            'block': self.block[:,n],
            'stim1': self.stims[:,0,n],
            'stim2': self.stims[:,1,n],
            'feedback': self.feedback[:,n],
            'num_blocks_since_last_block1': self.num_blocks_since_last_block[:,0,n],
            'num_blocks_since_last_block2': self.num_blocks_since_last_block[:,1,n],
            'type1': self.stim_types[:,0,n],
            'type2': self.stim_types[:,1,n],
            'p_r1' : self.p_R[:,0,n],
            'p_pun1':self.p_Pun[:,0,n],
            'p_r2' : self.p_R[:,1,n],
            'p_pun2':self.p_Pun[:,1,n]
        })
        
        return df
        
    ### Plotting functions ###
    def plot_avg_acc(self):
        """
        Given Choice data and data on the stimuli for N subjects, displays the average accuracy along all subjects, per time step.
        """
        self.avg_acc = np.mean(self.acc, axis = 1) # (T)
        self.avg_acc[self.best_stim.isna().any(axis = 1)] = np.nan                          # (T)
        df = pd.DataFrame({'avg_acc': self.avg_acc})
                
        no_feedback = np.argwhere(self.feedback[:,0] < 1e-10).flatten()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
                x = df.index,
                y = df['avg_acc'].values,
                mode = 'lines',
                line = dict(color='blue', width = 1),
                name = 'avg_acc'
            ))
        fig.add_trace(go.Scatter(
                x = no_feedback,
                y = np.ones_like(no_feedback),
                mode = 'markers',
                opacity = .2,
                marker = dict(
                    line = dict(color='red', width = .2)),
                name = 'no feedback'
        ))
        fig.update_layout(yaxis={
                'range': [0., 1.]  
            })
        fig.show()    
        
    def plot_num_f_choice(self, save_folder = None, subject_names = None, show_figs = False):
        """
        Displays the number of overall stimulus choices startified by type in boxplots.
        
        Arguments:
        - subject_names: list of strings or None, names of the subjects to display. All if None.
        - show_figs: Boolean, displays the figures of True.
        - save_folder: None or String, the path to save the figures if not None.
        """
        subject_names, columns = self.subject_names_to_columns(subject_names, False)
        
        min_stim = self.min_stim
        
        fig = go.Figure()
        for n in columns:
            df = self.create_subject_df(n=n)
            df = df[df['C'] >= self.min_stim]
            df['C'] = pd.Categorical(df['C'])            
            df = df[df['feedback'] > .5]
            
            counts = df['C'].value_counts()
            
            df1 = df.loc[df['C_st'] < .5, ['stim1', 'type1']].groupby('stim1')['type1'].first()
            df2 = df.loc[df['C_st'] < .5, ['stim2', 'type2']].groupby('stim2')['type2'].first()
            
            df_counts = pd.concat((counts, df1), axis = 1)
            df_counts = df_counts[df_counts.index >= min_stim]       
            df_counts.loc[df_counts['type1'].isna(), 'type1'] = df2[df_counts.index[df_counts['type1'].isna()]]
            df_counts['type1'] = df_counts['type1'].astype('int32')
            df_counts.rename(columns = {'C': 'counts', 'type1': 'type'}, inplace = True) 
            df_counts.loc[df_counts['counts'].isna(), 'counts'] = 0
                    
            fig = px.box(df_counts, x="type", y="counts", points="all")
            fig.update_layout(title = f"{self.subject_names[n]}) Number of feedback choices")
            if show_figs:
                fig.show()
            
            if save_folder is not None:
                results_folder = os.path.join(save_folder, 'feedback_choice_rep')
                if not os.path.exists(results_folder):
                    os.mkdir(results_folder)
                fig.write_image(os.path.join(results_folder, f"{self.subject_names[n]}_fchoice_rep.jpeg"))
    
    def plot_learning_curve(self, experienced_EV = False, subject_names = None, smoothing_window = 10, order_names = False, show_figs = True, save_folder = None, return_average = False, name_modifier = ""):
        """
        
        Arguments:
        - subject_names: list of subject names (corresponding to subjects).
        - smoothing window: int, the number of trials to smooth over.
        - return_average: boolean, returns the group average in a numpy array if True.
        - experienced_EV: boolean, uses the experienced Expected Values if True, the objective ones otherwise.
        
        Displays the learning curves in a plotly graph.
        """
        
        subject_names, columns = self.subject_names_to_columns(subject_names, order_names)
        
        block = self.block
        
        if experienced_EV:
            acc_name = 'acc_exp'
        else:
            acc_name = 'acc'
        
        if return_average:
            accs = []
        
        fig = go.Figure()
        for n in columns:
            df  = self.create_subject_df(n=n)
            
            df = df[(df['feedback'] > .5) & (df['stim1'] > 17.) & (df['stim2'] > 17.)]
            #df = df[df['feedback'] > .5]
            
            df = df[['block', 'trial', acc_name]]
            df['feedback_trial'] = df.groupby('block').cumcount()
                        
            acc = df.groupby('feedback_trial')[acc_name].mean()
            
            acc_smoothed = acc.rolling(smoothing_window, min_periods = 1).mean()
            if return_average:
                accs.append(acc_smoothed.values)
            
            fig.add_trace(go.Scatter(x = acc_smoothed.loc[smoothing_window-1:].index,
                                    y = acc_smoothed.loc[smoothing_window-1:],
                                    name = self.subject_names[n],
                                    marker_color = self.colors[n],
                                    mode = 'lines'))
        if experienced_EV:
            title_modif = 'Experienced EV'
            save_modif  = 'exp'
        else:
            title_modif = 'Objective EV'
            save_modif  = 'obj'

        fig.update_layout(
            title = f"Learning curve ({smoothing_window}-running average accuracy) averaged per block, {title_modif}",
            xaxis_title = 'Window',
            yaxis_title = 'Smoothed accuracy',
            # yaxis={
                # 'range': [0.5, 1.]  }
            )
        if show_figs:
            fig.show()
            
        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
        
            fig.write_image(os.path.join(save_folder, f"lcurve_{save_modif}_{name_modifier}.jpeg"))
        
        if return_average:
            accs = np.array(accs)
            return np.nanmean(accs, axis = 0), np.nanstd(accs, axis = 0)
                
    def plot_new_learning_curve(self, experienced_EV = False, save_folder = None, smoothing_window = 10, show_figs = True, return_average = False, name_modifier = ""):
        """
        Constructs and displays the new learning curve of accuracy versus minimum number of 
        previous choices between the 2 stimuli on display. 
        
        Smoothes the learning urve by taking the running average over 'smoothing_window' steps.
        """
        
        fig = go.Figure()
        fig_appear = go.Figure()
        if return_average:
            accs = []
            
        if experienced_EV:
            acc_name = 'acc_exp'
            title_modif = 'Experienced EV'
            save_modif  = 'exp'
        else:
            acc_name = 'acc'
            title_modif = 'Objective EV'
            save_modif  = 'obj'
        
        for n in range(self.N):
            df = self.create_subject_df(n=n)
            
            df = df[(df['feedback'] > .5) & (df['stim1'] > 17.) & (df['stim2'] > 17.)]
            #df = df[df['feedback'] > .5]
            
            df['min_num_choice'] = np.minimum(df['num_choice1'], df['num_choice2'])
            
            acc = df.groupby('min_num_choice')[acc_name].mean()
            #std = df.groupby('min_num_choice')[acc_name].std()            
            acc_smoothed = acc.rolling(smoothing_window, min_periods = 1).mean()
            
            fig.add_trace(go.Scatter(x = acc_smoothed.loc[smoothing_window:].index,
                                    y = acc_smoothed.loc[smoothing_window:],
                                    name = self.subject_names[n],
                                    marker_color = self.colors[n],
                                    mode = 'lines'))
            df['min_num_appear'] =  np.minimum(df['num_appear1'], df['num_appear2'])
            #df['min_num_appear'] = (df['num_appear1']+df['num_appear2'])/2#np.minimum(df['num_appear1'], df['num_appear2'])
            
            acc = df.groupby('min_num_appear')[acc_name].mean()
            acc_smoothed = acc.rolling(smoothing_window, min_periods = 1).mean()
            
            if return_average:
                accs.append(acc_smoothed.values)
            
            fig_appear.add_trace(go.Scatter(x = acc_smoothed.loc[(smoothing_window - 1):].index,
                                    y = acc_smoothed.loc[(smoothing_window - 1):],
                                    name = self.subject_names[n],
                                    marker_color = self.colors[n],
                                    mode = 'lines'))
                                    
            
        fig_appear.update_layout(title = f'Learning Curve vs Number Of Appearances (Smoothing Window: {smoothing_window}), {title_modif}',
                         xaxis_title = 'Minimum # of previous appearances of presented stimuli',
                         yaxis_title = 'Accuracy')
    
    
        fig.update_layout(title = f'Learning Curve vs Number Of Choice (Smoothing Window: {smoothing_window}), {title_modif}',
                         xaxis_title = 'Minimum # of previous choices of presented stimuli',
                         yaxis_title = 'Accuracy')
        if show_figs:
            fig_appear.show()
            fig.show()
        
        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
                
            save_path = os.path.join(save_folder, f'learning_curve_vs_num_choice_{save_modif}_{name_modifier}.jpeg')
            fig.write_image(save_path)
            
            save_path = os.path.join(save_folder, f'learning_curve_vs_num_appear_{save_modif}_{name_modifier}.jpeg')
            fig_appear.write_image(save_path)
            
        if return_average:
            accs = np.array(accs)
            return np.nanmean(accs, axis = 0), np.nanstd(accs, axis = 0)
            
    
        
    def plot_new_learning_curve_max_drop(self, experienced_EV=False, save_folder=None, smoothing_window=10, show_figs=True, return_average=False, name_modifier=""):
        """
        Constructs and displays the new learning curve of accuracy versus minimum number of 
        previous choices between the 2 stimuli on display. 
    
        Smoothes the learning curve by taking the running average over 'smoothing_window' steps.
        """
    
        fig = go.Figure()
        fig_appear = go.Figure()
        if return_average:
            accs = []
        
        if experienced_EV:
            acc_name = 'acc_exp'
            title_modif = 'Experienced EV'
            save_modif  = 'exp'
        else:
            acc_name = 'acc'
            title_modif = 'Objective EV'
            save_modif  = 'obj'
        
        for n in range(self.N):
            df = self.create_subject_df(n=n)            
            df = df[(df['feedback'] > .5) & (df['stim1'] > 17.) & (df['stim2'] > 17.)]           
            df['max_num_appear_temp'] = np.maximum(df['num_appear1'], df['num_appear2'])
            df_first = df[(df['num_appear1'] <= 32) & (df['num_appear2'] <= 32)] #trials where both stimuli appear in their first session
            df_second = df[(df['num_appear1'] > 32) & (df['num_appear2'] > 32)] #trials where both stimuli appear in their second session
            acc_first = df_first.groupby('max_num_appear_temp')[acc_name].mean()            
            acc_second = df_second.groupby('max_num_appear_temp')[acc_name].mean()           
            #acc_first_smoothed = pd.Series([acc_first.iloc[0]] + acc_first.iloc[1:].rolling(smoothing_window, min_periods=2).mean().tolist()).dropna()
            #acc_second_smoothed = (acc_second.rolling(smoothing_window, min_periods=2).mean()).dropna()
            acc = pd.concat([acc_first, acc_second]) 
            acc_smoothed = acc.rolling(smoothing_window, min_periods=2).mean().dropna()
            acc_smoothed = acc_smoothed[::2]
            #acc_smoothed = acc[::2].rolling(smoothing_window, min_periods=2).mean()  
            #acc_smoothed = pd.concat([acc_first_smoothed, acc_second_smoothed])
            x = acc_smoothed.loc[smoothing_window:].index.values            
            #x = np.arange(1, len(acc_smoothed) + 1)
            y = acc_smoothed.loc[smoothing_window:].values
            #y= acc_smoothed.values
            # Find the index of the value where the drop occurs
            drop_idx = np.where(np.diff(y) < -0.04)[0] + 1  # adjust the threshold as needed
            
            # Insert NaN value at the drop point
            #if len(drop_idx) > 0:
             #   x = x.astype(float) 
              #  y = y.astype(float)
               # x = np.insert(x, drop_idx, np.nan)
                #y = np.insert(y, drop_idx, np.nan)
            

            
            fig_appear.add_trace(go.Scatter(x=acc_smoothed.loc[smoothing_window:].index, 
                                            y=acc_smoothed.loc[smoothing_window:], 
                                            name=self.subject_names[n], 
                                            marker_color=self.colors[n], mode='lines'))
            
            if return_average:
                #print(acc_smoothed)
                #accs.append(acc_smoothed.loc[smoothing_window:].values) 
                accs.append(acc_smoothed.values)
            
        
        fig.update_layout(title=f'Learning curve ({title_modif})', 
                          xaxis_title='Max number of previous choices', 
                          yaxis_title='Accuracy')
        
        fig_appear.update_layout(title=f'Learning curve ({title_modif})', 
                                 xaxis_title='Max number of previous appearances', 
                                 yaxis_title='Accuracy')
        
        if save_folder:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            fig.write_image(f'{save_folder}/learning_curve_{save_modif}_choice_{name_modifier}.png')
            fig_appear.write_image(f'{save_folder}/learning_curve_{save_modif}_appear_{name_modifier}.png')
        
        if show_figs:
            fig.show()
            fig_appear.show()

        if return_average:
            accs = np.array(accs)
            return np.nanmean(accs, axis = 0), np.nanstd(accs, axis = 0)


    def plot_new_learning_curve_maxi(self, experienced_EV=False, save_folder=None, smoothing_window=10, show_figs=True, return_average=False, name_modifier=""):
        acc_name = 'acc'

        if return_average:
            accs = []        
       
        for n in range(self.N):
            df = self.create_subject_df(n=n)            
            df = df[(df['feedback'] > .5) & (df['stim1'] > 17.) & (df['stim2'] > 17.) < (df['type1'] > 2.) & (df['type2'] < 2.)]           
            df['max_num_appear_temp'] = np.maximum(df['num_appear1'], df['num_appear2'])
            df_first = df[(df['num_appear1'] <= 32) & (df['num_appear2'] <= 32)] #trials where both stimuli appear in their first session
            df_second = df[(df['num_appear1'] > 32) & (df['num_appear2'] > 32)] #trials where both stimuli appear in their second session
            acc_first = df_first.groupby('max_num_appear_temp')[acc_name].mean()            
            acc_second = df_second.groupby('max_num_appear_temp')[acc_name].mean()           
            acc = pd.concat([acc_first, acc_second])
            acc_smoothed = acc.rolling(smoothing_window, min_periods=1).mean()

            
            if return_average:
                accs.append(acc.loc[smoothing_window:].values) ####-1?

        if return_average:
            accs = np.array(accs)
            ggg=np.nanmean(accs, axis = 0)
            print(ggg)
            return np.nanmean(accs, axis = 0), np.nanstd(accs, axis = 0), df       

    
    
    
    def plot_new_learning_curve_max(self, experienced_EV = False, save_folder = None, smoothing_window = 10, show_figs = True, return_average = False, name_modifier = ""):
        """
        Constructs and displays the new learning curve of accuracy versus minimum number of 
        previous choices between the 2 stimuli on display. 
        
        Smoothes the learning urve by taking the running average over 'smoothing_window' steps.
        """
        
        fig = go.Figure()
        fig_appear = go.Figure()
        if return_average:
            accs = []
            
        if experienced_EV:
            acc_name = 'acc_exp'
            title_modif = 'Experienced EV'
            save_modif  = 'exp'
        else:
            acc_name = 'acc'
            title_modif = 'Objective EV'
            save_modif  = 'obj'
        
        for n in range(self.N):
            df = self.create_subject_df(n=n)
            
            df = df[(df['feedback'] > .5) & (df['stim1'] > 17.) & (df['stim2'] > 17.)]
            
            df['max_num_choice'] = np.maximum(df['num_choice1'], df['num_choice2'])                       
            acc = df.groupby('max_num_choice')[acc_name].mean()
            
            
            #acc_stim1 = df.groupby('num_choice1')[acc_name].mean()
            #std_stim1 = df.groupby('num_appear1')[acc_name].std()
            #acc_stim2 = df.groupby('num_choice2')[acc_name].mean()
            #std_stim2 = df.groupby('num_appear2')[acc_name].std()
            
            #acc_con=pd.concat([acc_stim1,acc_stim2],axis=1)
            #results=acc_con.mean(axis=1, skipna=True)
            #acc=results.fillna(method='ffill')
            acc_smoothed = acc.rolling(smoothing_window, min_periods = 1).mean()
            
          
            
            fig.add_trace(go.Scatter(x = acc_smoothed.loc[smoothing_window:].index,
                                    y = acc_smoothed.loc[smoothing_window:],
                                    name = self.subject_names[n],
                                    marker_color = self.colors[n],
                                    mode = 'lines'))
            
            df['max_num_appear'] = np.maximum(df['num_appear1'], df['num_appear2']) #np.maximum(df['num_appear1'], df['num_appear2'])
            acc = df.groupby('max_num_appear')[acc_name].mean()
            #acc_stim1 = df.groupby('num_appear1')[acc_name].mean()
            #std_stim1 = df.groupby('num_appear1')[acc_name].std()
            #acc_stim2 = df.groupby('num_appear1')[acc_name].mean()
            #std_stim2 = df.groupby('num_appear2')[acc_name].std()
            
            #acc_con=pd.concat([acc_stim1,acc_stim2],axis=1)
            #results=acc_con.mean(axis=1, skipna=True)
            #acc=results.fillna(method='ffill')
            acc_smoothed = acc.rolling(smoothing_window, min_periods = 1).mean()
            
            if return_average:
                accs.append(acc_smoothed.values)
            
            
            fig_appear.add_trace(go.Scatter(x = acc_smoothed.loc[(smoothing_window - 1):].index,
                                    y = acc_smoothed.loc[(smoothing_window - 1):],
                                    name = self.subject_names[n],
                                    marker_color = self.colors[n],
                                    mode = 'lines'))
                                    
            
        fig_appear.update_layout(title = f'Learning Curve vs Number Of Appearances (Smoothing Window: {smoothing_window}), {title_modif}',
                         xaxis_title = 'Maximum # of previous appearances of presented stimuli',
                         yaxis_title = 'Accuracy')
    
    
        fig.update_layout(title = f'Learning Curve vs Number Of Choice (Smoothing Window: {smoothing_window}), {title_modif}',
                         xaxis_title = 'Maximum # of previous choices of presented stimuli',
                         yaxis_title = 'Accuracy')
        if show_figs:
            fig_appear.show()
            fig.show()
        
        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
                
            save_path = os.path.join(save_folder, f'learning_curve_vs_num_choice_{save_modif}_{name_modifier}.jpeg')
            fig.write_image(save_path)
            
            save_path = os.path.join(save_folder, f'learning_curve_vs_num_appear_{save_modif}_{name_modifier}.jpeg')
            fig_appear.write_image(save_path)
            
        if return_average:
            accs = np.array(accs)
            return np.nanmean(accs, axis = 0), np.nanstd(accs, axis = 0)   
       
            
        
    def plot_risk_per_group(self, average_risk_aversion_per_subject_sample1, average_risk_aversion_per_subject_sample2, save_folder = None, show_figs = True):
        fig = go.Figure()
        fig.add_trace(go.Violin(x=average_risk_aversion_per_subject_sample1['risk_aversion'], fillcolor='rgba(255, 20, 147, 0.4)', line_color='black' , box_visible=True, points = 'all', showlegend=False, name= '', marker=dict(color='rgba(255, 20, 147, 0.8)', size =20, line=dict(color='black', width=2.5))))
        fig.add_trace(go.Violin(x=average_risk_aversion_per_subject_sample2['risk_aversion'], fillcolor='rgba(28, 134, 238, 0.4)', line_color='black',  box_visible=True, points = 'all', showlegend=False, name= '', marker=dict(color='rgba(28, 134, 238, 0.4)', size =20, line=dict(color='black', width=2.5))))

       
        
        # Add text annotations
        #fig.add_annotation(
         #   go.layout.Annotation(
          #      text='Always chose safe',
           #     x=10.5, y=-0.51,  # Adjust the y-coordinate to place the text below 0%
            #    showarrow=False,
             #   font=dict(size=36),
            #)
        #)
        #fig.add_annotation(
         #   go.layout.Annotation(
          #      text='Always chose risky',
           #     x=89.5, y=-0.51,  # Adjust the y-coordinate to place the text below 100%
            #    showarrow=False,
             #   font=dict(size=36),
            #)
        #)
        
        fig.update_layout(
            xaxis_title = 'Risk taking',            
            font=dict(
            family="Arial",
            size=50,
            color='black'
            ),
            plot_bgcolor='white',
            legend_traceorder="normal",
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(
                tracegroupgap=10,  # Adjust the gap between legend entries
                font=dict(
                size=60))  # Adjust the font size of the legend                
            )
        
        fig.update_xaxes(showline=True, linewidth=2.7, linecolor='black', ticks='outside', mirror=True, range=[-4, 104], tickvals=[0,50,100], ticktext=['0%', '50%', '100%'], tickmode='array', tickfont=dict(size=46), ticklen=12, tickwidth=3, title_standoff=30)
        fig.update_yaxes(showline=True, linewidth=2.7, linecolor='black', ticks='outside', mirror=True, range=[-0.5,0.3], tickfont=dict(size=42), ticklen=12, tickwidth=3, title_standoff=30)
        
        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
        if show_figs:
            fig.show()           
        if save_folder is not None:
            name = f"risk_violins.svg"
            fig.write_image(os.path.join(save_folder, name), width=1547, height=1105)
            
    def plot_switch_per_group(self, switch_df_first, switch_df_second, save_folder=None, show_figs=True):
        fig = go.Figure()

        # Violin plot for switch_df_first - pun
        fig.add_trace(go.Violin(y=switch_df_first['pun'], fillcolor='rgba(255, 20, 147, 0.4)', line_color='black', box_visible=True, points='all', showlegend=False, name='p', marker=dict(color='rgba(255, 20, 147, 0.8)', size =20, line=dict(color='black', width=2.5)), offsetgroup='group1', width=0.6, side='both'))
        
        # Violin plot for switch_df_second - pun
        fig.add_trace(go.Violin(y=switch_df_second['pun'], fillcolor='rgba(28, 134, 238, 0.4)', line_color='black', box_visible=True, points='all', showlegend=False, name='p', marker=dict(color='rgba(28, 134, 238, 0.4)', size =20, line=dict(color='black', width=2.5)), offsetgroup='group1', width=0.6, side='both'))

        # Violin plot for switch_df_first - n
        fig.add_trace(go.Violin(y=switch_df_first['n'], fillcolor='rgba(255, 20, 147, 0.4)', line_color='black', box_visible=True, points='all', showlegend=False, name='n', marker=dict(color='rgba(255, 20, 147, 0.8)', size =20, line=dict(color='black', width=2.5)), offsetgroup='group2', width=0.6, side='both'))
        
        # Violin plot for switch_df_second - n
        fig.add_trace(go.Violin(y=switch_df_second['n'], fillcolor='rgba(28, 134, 238, 0.4)', line_color='black', box_visible=True, points='all', showlegend=False, name='n', marker=dict(color='rgba(28, 134, 238, 0.4)', size =20, line=dict(color='black', width=2.5)), offsetgroup='group2', width=0.6, side='both'))
        
        # Violin plot for switch_df_first - r
        fig.add_trace(go.Violin(y=switch_df_first['r'], fillcolor='rgba(255, 20, 147, 0.4)', line_color='black', box_visible=True, points='all', showlegend=False, name='r', marker=dict(color='rgba(255, 20, 147, 0.8)', size =20, line=dict(color='black', width=2.5)), offsetgroup='group3', width=0.6, side='both'))
        
        # Violin plot for switch_df_second - r
        fig.add_trace(go.Violin(y=switch_df_second['r'], fillcolor='rgba(28, 134, 238, 0.4)', line_color='black', box_visible=True, points='all', showlegend=False, name='r', marker=dict(color='rgba(28, 134, 238, 0.4)', size =20, line=dict(color='black', width=2.5)), offsetgroup='group3', width=0.6, side='both'))

        # Update layout and legend
        fig.update_layout(
            xaxis_title='Outcome',
            yaxis_title='Chose the same image again',
            font=dict(
                family="Arial",
                size=50,
                color='black'
            ),
            plot_bgcolor='white',
            legend_traceorder="normal",
            margin=dict(l=0, r=0),
            legend=dict(
                tracegroupgap=10,  # Adjust the gap between legend entries
                font=dict(
                    size=60
                )  # Adjust the font size of the legend
            )
        )
        
        # Update x and y axes styling
        fig.update_xaxes(showline=True, linewidth=2.7, linecolor='black', ticks='outside', mirror=True, tickvals=[-0.23, 0.77, 1.77],ticktext=['p','n', 'r'], tickmode='array', tickfont=dict(size=95), ticklen=12, tickwidth=3, title_standoff=40)
        fig.update_yaxes(showline=True, linewidth=2.7, linecolor='black', ticks='outside', mirror=True, range=[0, 1], tickfont=dict(size=42), ticklen=12, tickwidth=3, title_standoff=30)
        
        
        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
        if show_figs:
            fig.show()
        if save_folder is not None:
            name = f"switch_violins.svg"
            fig.write_image(os.path.join(save_folder, name), width=1547, height=1105)
            
    def compare_group_lcurves(self, s_dat_b, experienced_EV = False, smoothing_window = 10, save_folder = None, show_figs = True, group_names = None, new_lcurve = True):
        """
        A function to compare learning curves between two subjects_EMA objects on the population level.
        """
    
        if group_names is None:
            group_names = ['Group_1', 'Group_2']
    
        if new_lcurve:
            acc_avg, acc_std = self.plot_new_learning_curve_max(experienced_EV = experienced_EV, smoothing_window = smoothing_window, show_figs = False, save_folder = save_folder, return_average = True, name_modifier = group_names[0])
            acc_avg_b, acc_std_b = s_dat_b.plot_new_learning_curve_max(experienced_EV = experienced_EV, smoothing_window = smoothing_window, show_figs = False, save_folder = save_folder, return_average = True, name_modifier = group_names[1])
        else:
            acc_avg, acc_std = self.plot_learning_curve(experienced_EV = experienced_EV, smoothing_window = smoothing_window, show_figs = False, save_folder = save_folder, return_average = True, name_modifier = group_names[0])
            acc_avg_b, acc_std_b = s_dat_b.plot_learning_curve(experienced_EV = experienced_EV, smoothing_window = smoothing_window, show_figs = False, save_folder = save_folder, return_average = True, name_modifier = group_names[1])
                
        N0 = self.N
        N1 = s_dat_b.N
                
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x = np.arange(smoothing_window - 9, acc_avg.shape[0]-9),
                                    y = acc_avg[smoothing_window-1:],
                                    error_y=dict(
                                        type='data',
                                        symmetric=True,
                                        array= 1.96 * acc_std[smoothing_window-1:] / np.sqrt(N0),
                                        thickness = .5),
                                    name = group_names[0],
                                    mode = 'lines'))
                                    
        fig.add_trace(go.Scatter(x = np.arange(smoothing_window - 9, acc_avg_b.shape[0]-9),
                                    y = acc_avg_b[smoothing_window-1:],
                                    error_y=dict(
                                        type='data',
                                        symmetric=True,
                                        array= 1.96 * acc_std_b[smoothing_window-1:] / np.sqrt(N1),
                                        thickness = .5),
                                    name = group_names[1],
                                    mode = 'lines'))
        
        if new_lcurve:
            xaxis_title = "Minimum number of appearances"
        else:
            xaxis_title = "Window"
            
        if experienced_EV:
            title_modif = 'Experienced EV'
            save_modif  = 'exp'
        else:
            title_modif = 'Objective EV'
            save_modif  = 'obj'
        
        fig.update_layout(
            title = f"Group learning curves ({smoothing_window}-smoothed), {title_modif}",
            xaxis_title = xaxis_title,
            yaxis_title = 'Average smoothed accuracy',
            yaxis={
                'range': [0.5, 1.]  
            })
            
        if show_figs:
            fig.show()
            
        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
            
            if new_lcurve:
                name = f"grouped_new_lcurve_{save_modif}.jpeg"
            else:
                name = f"grouped_lcurve_{save_modif}.jpeg"
            fig.write_image(os.path.join(save_folder, name))    
    
    def compare_group_lcurves_smoothed_CI(self, s_dat_b, experienced_EV = False, smoothing_window = 10, save_folder = None, show_figs = True, group_names = None, new_lcurve = False, new_lcurve_max_drop = False):
        """
        A function to compare learning curves between two subjects_EMA objects on the population level.
        """
    
        if group_names is None:
            group_names = ['Group 1', 'Group 2']
    
        if new_lcurve:
            acc_avg, acc_std = self.plot_new_learning_curve(experienced_EV = experienced_EV, smoothing_window = smoothing_window, show_figs = False, save_folder = save_folder, return_average = True, name_modifier = group_names[0])
            acc_avg_b, acc_std_b = s_dat_b.plot_new_learning_curve(experienced_EV = experienced_EV, smoothing_window = smoothing_window, show_figs = False, save_folder = save_folder, return_average = True, name_modifier = group_names[1])
        elif new_lcurve_max_drop:
            acc_avg, acc_std = self.plot_new_learning_curve_max_drop(experienced_EV = experienced_EV, smoothing_window = smoothing_window, show_figs = False, save_folder = save_folder, return_average = True, name_modifier = group_names[0])
            acc_avg_b, acc_std_b = s_dat_b.plot_new_learning_curve_max_drop(experienced_EV = experienced_EV, smoothing_window = smoothing_window, show_figs = False, save_folder = save_folder, return_average = True, name_modifier = group_names[1])
        else:
            acc_avg, acc_std = self.plot_learning_curve(experienced_EV = experienced_EV, smoothing_window = smoothing_window, show_figs = False, save_folder = save_folder, return_average = True, name_modifier = group_names[0])
            acc_avg_b, acc_std_b = s_dat_b.plot_learning_curve(experienced_EV = experienced_EV, smoothing_window = smoothing_window, show_figs = False, save_folder = save_folder, return_average = True, name_modifier = group_names[1])
                
        N0 = self.N
        N1 = s_dat_b.N
                
        
        # Find the index of the value where the drop occurs
        drop_idx = np.where(np.diff(acc_avg) < -0.01)[0]+1   # adjust the threshold as needed -0.0603
        
        
        #acc_avg = np.insert(acc_avg, 0, np.nan)
        #acc_std = np.insert(acc_std, 0, np.nan)
        #acc_avg_b = np.insert(acc_avg_b, 0, np.nan)
        #acc_std_b = np.insert(acc_std_b, drop_idx, np.nan)
        # Insert NaN value at the drop point
        if len(drop_idx) > 0:
           #x = x.astype(float) 
           #y = y.astype(float)
           acc_avg = np.insert(acc_avg, drop_idx, np.nan)
           acc_std = np.insert(acc_std, drop_idx, np.nan)
           acc_avg_b = np.insert(acc_avg_b, drop_idx, np.nan)
           acc_std_b = np.insert(acc_std_b, drop_idx, np.nan)
        
        layout = Layout(plot_bgcolor='rgba(0,0,0,0)')
        fig = go.Figure(layout=layout)
        
        fig.add_trace(go.Scatter(x = np.arange(1, len(acc_avg[smoothing_window-2:])+1),
                                    y = acc_avg[smoothing_window-2:],
                                    name = group_names[0],
                                    line=dict(color='rgba(255, 20, 147, 0.8)', width=7), showlegend=False))

        fig.add_trace(go.Scatter(name='Upper Bound',
        x=np.arange(1, drop_idx.item()+1),
        y=acc_avg[smoothing_window-2:drop_idx.item()] + 1.96 * acc_std[smoothing_window-2:drop_idx.item()] / np.sqrt(N0),
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        showlegend=False))
        
        fig.add_trace(go.Scatter(name='Lower Bound',
        x=np.arange(1, drop_idx.item()+1),
        y=acc_avg[smoothing_window-2:drop_idx.item()] - 1.96 * acc_std[smoothing_window-2:drop_idx.item()] / np.sqrt(N0),
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(255, 20, 147, 0.4)',
        fill='tonexty',
        showlegend=False))  
        
        
        fig.add_trace(go.Scatter(name='Upper Bound',
        x=np.arange(drop_idx.item()+1, len(acc_avg[smoothing_window-2:])+1),
        y=acc_avg[drop_idx.item():] + 1.96 * acc_std[drop_idx.item():] / np.sqrt(N0),
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        showlegend=False))
        
        fig.add_trace(go.Scatter(name='Lower Bound',
        x=np.arange(drop_idx.item()+1, len(acc_avg[smoothing_window-2:])+1),
        y=acc_avg[drop_idx.item():] - 1.96 * acc_std[drop_idx.item():] / np.sqrt(N0),
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(255, 20, 147, 0.4)',
        fill='tonexty',
        showlegend=False)) 
      
                                   
        fig.add_trace(go.Scatter(x = np.arange(1, len(acc_avg_b[smoothing_window-2:])+1),
                                    y = acc_avg_b[smoothing_window-2:],
                                    name = group_names[1],
                                    line=dict(color='rgba(28, 134, 238, 0.5)', width=7),showlegend=False))        
        
        
        
        fig.add_trace(go.Scatter(name='Upper Bound',
        x=np.arange(1, drop_idx.item()+1),
        y=acc_avg_b[smoothing_window-2:drop_idx.item()] + 1.96 * acc_std_b[smoothing_window-2:drop_idx.item()] / np.sqrt(N1),
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        showlegend=False))
        
        fig.add_trace(go.Scatter(name='Lower Bound',
        x=np.arange(1, drop_idx.item()+1),
        y=acc_avg_b[smoothing_window-2:drop_idx.item()] - 1.96 * acc_std_b[smoothing_window-2:drop_idx.item()] / np.sqrt(N1),
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(28, 134, 238, 0.4)',
        fill='tonexty',
        showlegend=False)) 
        
        
        fig.add_trace(go.Scatter(name='Upper Bound',
        x=np.arange(drop_idx.item()+1, len(acc_avg_b[smoothing_window-2:])+1),
        y=acc_avg_b[drop_idx.item():] + 1.96 * acc_std_b[drop_idx.item():] / np.sqrt(N1),
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        showlegend=False))
        
        fig.add_trace(go.Scatter(name='Lower Bound',
        x=np.arange(drop_idx.item()+1, len(acc_avg_b[smoothing_window-2:])+1),
        y=acc_avg_b[drop_idx.item():] - 1.96 * acc_std_b[drop_idx.item():] / np.sqrt(N1),
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(28, 134, 238, 0.4)',
        fill='tonexty',
        showlegend=False)) 
        
        
        #add dashed line between the curves:
        # define the vertical dashed line
        line = dict(type='line', xref='x', yref='paper', x0=17, y0=0, x1=17, y1=1, line=dict(dash='dash', color='black'))
        fig.update_layout(shapes=[line])
        
       
        if new_lcurve:
            xaxis_title = "Minimum number of appearances"
        elif new_lcurve_max_drop:
            xaxis_title = "Trial"
        else:
            xaxis_title = "Window"
            
        if experienced_EV:
            title_modif = 'Experienced EV'
            save_modif  = 'exp'
        else:
            title_modif = 'Objective EV'
            save_modif  = 'obj'
        
        xrange = len(np.arange(1, len(acc_avg[smoothing_window-2:])+1))
        
        fig.update_layout(
            #title=dict(text='Group Learning Curves',x=0.5,y=0.9, font=dict(family="Arial", size=38, color='black')),
            xaxis_title = xaxis_title,
            yaxis_title = 'Learning accuracy',
            font=dict(
            family="Arial",
            size=50,
            color='black'
            ),
            legend_traceorder="normal",
            yaxis={
                'range': [0.5, 1]
                },
            xaxis={
                'range': [0, xrange]},
            margin=dict(l=0, r=25),
            legend=dict(
                tracegroupgap=10,  # Adjust the gap between legend entries
                font=dict(
                size=60))  # Adjust the font size of the legend                
            )
            
        fig.update_layout(
            annotations=[
                {
                    'x': 9,
                    'y': 0.3,
                    'text': '<b>DAY 1</b>',
                    'showarrow': False,
                    'font': {
                        'size': 50,
                        'color': 'lightgrey'                        
                    },
                    'xref': 'x',
                    'yref': 'paper',
                    'xanchor': 'center',
                    'yanchor': 'bottom'
                },
                {
                    'x': 25,
                    'y': 0.3,
                    'text': '<b>DAY 2</b>',
                    'showarrow': False,
                    'font': {
                        'size': 50,
                        'color': 'lightgrey'                        
                    },
                    'xref': 'x',
                    'yref': 'paper',
                    'xanchor': 'center',
                    'yanchor': 'bottom'
                }
            ]
        )
           
         
        
        
        fig.update_xaxes(showline=True, linewidth=2.7, linecolor='black', ticks='outside', mirror=True, tickvals=[0,4,8,12,16,21,25,29,33],ticktext=[0,8,16,24,32,40,48,56,64],tickmode='array', tickfont=dict(size=42), ticklen=12, tickwidth=3, title_standoff=30)
        fig.update_yaxes(showline=True, linewidth=2.7, linecolor='black', ticks='outside', mirror=True, tickfont=dict(size=42), ticklen=12, tickwidth=3, title_standoff=30)
        
        
        if show_figs:
            fig.show()
            
        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
            
            if new_lcurve:
                name = f"grouped_new_lcurve_{save_modif}.svg"
            else:
                name = f"grouped_lcurve_{save_modif}.svg"
            fig.write_image(os.path.join(save_folder, name), width=1547, height=1105)
    
   
    
    
    def article_figures(self, s_dat_b, average_risk_aversion_per_subject_sample1, average_risk_aversion_per_subject_sample2,experienced_EV = False, smoothing_window = 10, save_folder = None, show_figs = True, group_names = None, new_lcurve = False, new_lcurve_max_drop = False):
        """
        A function to compare learning curves between two subjects_EMA objects on the population level.
        """
    
        if group_names is None:
            group_names = ['Group_1', 'Group_2']
    
        if new_lcurve:
            acc_avg, acc_std = self.plot_new_learning_curve_article(experienced_EV = experienced_EV, smoothing_window = smoothing_window, show_figs = False, save_folder = save_folder, return_average = True, name_modifier = group_names[0])
            acc_avg_b, acc_std_b = s_dat_b.plot_new_learning_curve_article(experienced_EV = experienced_EV, smoothing_window = smoothing_window, show_figs = False, save_folder = save_folder, return_average = True, name_modifier = group_names[1])
        elif new_lcurve_max_drop:
            acc_avg, acc_std = self.plot_new_learning_curve_max_drop(experienced_EV = experienced_EV, smoothing_window = smoothing_window, show_figs = False, save_folder = save_folder, return_average = True, name_modifier = group_names[0])
            acc_avg_b, acc_std_b = s_dat_b.plot_new_learning_curve_max_drop(experienced_EV = experienced_EV, smoothing_window = smoothing_window, show_figs = False, save_folder = save_folder, return_average = True, name_modifier = group_names[1])
        else:
            acc_avg, acc_std = self.plot_learning_curve(experienced_EV = experienced_EV, smoothing_window = smoothing_window, show_figs = False, save_folder = save_folder, return_average = True, name_modifier = group_names[0])
            acc_avg_b, acc_std_b = s_dat_b.plot_learning_curve(experienced_EV = experienced_EV, smoothing_window = smoothing_window, show_figs = False, save_folder = save_folder, return_average = True, name_modifier = group_names[1])
                
        N0 = self.N
        N1 = s_dat_b.N
                
        layout = Layout(plot_bgcolor='rgba(0,0,0,0)')
        #fig = go.Figure()
        fig = make_subplots(rows=1, cols=3, subplot_titles = ['Group Learning Curves', 'Average Daily Accuracy During No-Feedback Trials', 'Risk Seeking Tendency'], vertical_spacing = 0.1)
        
        fig.add_trace(go.Scatter(x = np.arange(1, len(acc_avg[smoothing_window-1:])+1),
                                    y = acc_avg[smoothing_window-1:],
                                    name = group_names[0],
                                    line=dict(color='rgba(150, 19, 238, 0.8)', width=6), showlegend=False), row=1, col=1)
                                                                       
        fig.add_trace(go.Scatter(name='Upper Bound',
        x=np.arange(1, len(acc_avg[smoothing_window-1:])+1),
        y=acc_avg[smoothing_window-1:] + 1.96 * acc_std[smoothing_window-1:] / np.sqrt(N0),
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        showlegend=False), row=1, col=1)
        
        fig.add_trace(go.Scatter(name='Lower Bound',
        x=np.arange(1, len(acc_avg[smoothing_window-1:])+1),
        y=acc_avg[smoothing_window-1:] - 1.96 * acc_std[smoothing_window-1:] / np.sqrt(N0),
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(150, 19, 238, 0.3)',
        fill='tonexty',
        showlegend=False), row=1, col=1)          
                                   
        fig.add_trace(go.Scatter(x = np.arange(1, len(acc_avg[smoothing_window-1:])+1),
                                    y = acc_avg_b[smoothing_window-1:],
                                    name = group_names[1],
                                    line=dict(color='rgba(33, 190, 47, 0.8)', width=6), showlegend=False), row=1, col=1)
        
        fig.add_trace(go.Scatter(name='Upper Bound',
        x=np.arange(1, len(acc_avg[smoothing_window-1:])+1),
        y=acc_avg_b[smoothing_window-1:] + 1.96 * acc_std_b[smoothing_window-1:] / np.sqrt(N1),
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        showlegend=False), row=1, col=1)
        
        fig.add_trace(go.Scatter(name='Lower Bound',
        x=np.arange(1, len(acc_avg[smoothing_window-1:])+1),
        y=acc_avg_b[smoothing_window-1:] - 1.96 * acc_std_b[smoothing_window-1:] / np.sqrt(N1),
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(33, 190, 47, 0.3)',
        fill='tonexty',
        showlegend=False), row=1, col=1) 
        
        if new_lcurve:
            xaxis_title = "Trial"
        else:
            xaxis_title = "Trial"
            
        if experienced_EV:
            title_modif = 'Experienced EV'
            save_modif  = 'exp'
        else:
            title_modif = 'Objective EV'
            save_modif  = 'obj'
        
        #fig.update_layout(
          #  xaxis2_title = xaxis_title,
           # yaxis2_title = 'Average smoothed accuracy',
            #font=dict(
            #family="Arial",
            #size=26,
            #color='black'
            #),
            #legend_traceorder="normal",
            #yaxis2={
             #   'range': [0.75, 0.95]
              #  },
            #xaxis2={
             #   'range': [0.5, 39.5]})
        
        
        fig.update_xaxes(showline=True, linewidth=1.4, linecolor='black', ticks='outside', mirror=True, row=1, col=1)
        fig.update_yaxes(showline=True, linewidth=1.4, linecolor='black', ticks='outside', mirror=True, row=1, col=1)
           
            

        #second figure:
        
        accs, overall_acc = self.plot_nf_acc_day(experienced_EV = experienced_EV, save_folder = save_folder, show_figs = False, return_values = True, name_modifier = group_names[0])
        accs_b, overall_acc_b = s_dat_b.plot_nf_acc_day(experienced_EV = experienced_EV, save_folder = save_folder, show_figs = False, return_values = True, name_modifier = group_names[1])
               
        df_acc = pd.concat(accs, axis = 1); N0 = len(accs)
        df_acc_b = pd.concat(accs_b, axis = 1); N1 = len(accs_b)
        
        if experienced_EV:
            acc_name = 'acc_exp'
            title_modif = 'Experienced EV'
            save_modif  = 'exp'
        else:
            acc_name = 'acc'
            title_modif = 'Objective EV'
            save_modif  = 'obj'
        
        #layout = Layout(plot_bgcolor='rgba(0,0,0,0)')
        #fig = go.Figure()
                
        fig.add_trace(go.Scatter(x = np.arange(df_acc.shape[0])+1,
                                y = df_acc.mean(axis = 1),
                                error_y=dict(type='data', array= 1.96 * df_acc.std(axis = 1) / np.sqrt(N0)),
                                mode = 'markers+lines',
                                name = group_names[0],
                                line=dict(color='rgba(150, 19, 238, 0.8)', width=3)), row=1, col=2)
                                
        fig.add_trace(go.Scatter(x = np.arange(df_acc_b.shape[0])+1,
                                y = df_acc_b.mean(axis = 1),
                                error_y=dict(type='data', array= 1.96 * df_acc_b.std(axis = 1) / np.sqrt(N1)),
                                mode = 'markers+lines',
                                name = group_names[1],
                                line=dict(color='rgba(33, 190, 47, 0.8)', width=3)), row=1, col=2)
                                
        #fig.update_layout(title = f'Daily Average of No-Feedback Accuracy',title_x=0.5,
         #                yaxis_title = 'Accuracy',
          #               xaxis_title = 'Day',               
           #              yaxis = {'range':[.75,.95]},
            #             xaxis = {'range':[-0.5, 27.5]})
        #fig.update_layout(title = f'Average No-Feedback Accuracy session-by-session, {title_modif}',
         #                yaxis_title = 'Accuracy',
          #               xaxis_title = 'Day',
           #              yaxis = {'range':[.75,.95]})
                         
        fig.update_xaxes(showline=True, linewidth=1.4, linecolor='black', ticks='outside', mirror=True, row=1, col=2)
        fig.update_yaxes(showline=True, linewidth=1.4, linecolor='black', ticks='outside', mirror=True, row=2, col=1)
       
       #third figure:
        fig.add_trace(go.Violin(x=average_risk_aversion_per_subject_sample2['risk_aversion'], name='Sample 2', fillcolor='rgba(33, 190, 47, 0.8)', line_color='black',  opacity=0.6, box_visible=True, points = 'all', showlegend=False), row=1, col=3)
        fig.add_trace(go.Violin(x=average_risk_aversion_per_subject_sample1['risk_aversion'], name='Sample 1', fillcolor='rgba(150, 19, 238, 0.8)', line_color='black' , opacity=0.6, box_visible=True, points = 'all', showlegend=False), row=1, col=3)
       
        fig.update_xaxes(showline=True, linewidth=1.4, linecolor='black', ticks='outside', mirror=True, row=1, col=3)
        fig.update_yaxes(showline=True, linewidth=1.4, linecolor='black', ticks='outside', mirror=True, row=1, col=3)
        #groups = ['Sample 1', 'Sample 2']
        #for group in groups:
         #   fig.add_trace(go.Violin(x=average_risk_aversion_per_subject['Risk Seeking'][average_risk_aversion_per_subject['group'] == group],
          #              y=average_risk_aversion_per_subject['group'][average_risk_aversion_per_subject['group'] == group],
           #             box_visible=True,
            #            meanline_visible=True), row=3, col=1)
        
       #fig.add_trace(go.Scatter(x = np.arange(df_acc.shape[0])+1,
        #                        y = df_acc.mean(axis = 1),
         #                       error_y=dict(type='data', array= 1.96 * df_acc.std(axis = 1) / np.sqrt(N0)),
          #                      mode = 'markers+lines',
           #                     name = group_names[0],
            #                    line=dict(color='rgba(150, 19, 238, 0.8)', width=3)), row=2, col=1)
                                
  
        
        #for all figures:
        fig.update_layout(xaxis=dict(title_text='Minimum number of appearances', range = [0.5, 62.5]), yaxis=dict(title_text='Average Smoothed Accuracy', range = [0.5,1]), xaxis2=dict(title_text='Day', range = [0.5,26.5]),yaxis2=dict(title_text='Accuracy', range = [0.75,0.94]),xaxis3=dict(title_text='Risk Seeking', range = [-1,1]),autosize=False, width=900, height= 1100, legend={'traceorder':'normal'})
        
        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
            
            if new_lcurve:
                name = f"grouped_new_lcurve_{save_modif}.jpeg"
            else:
                name = f"grouped_lcurve_{save_modif}.jpeg"
            fig.write_image(os.path.join(save_folder, name))   
        
        
        fig.show()
       
        

    ##            

         
            
    def compare_strat_lcurves(self, list_of_block_ranges, experienced_EV = False, smoothing_window = 10, save_folder = None, show_figs = True, group_names = None, new_lcurve = True):
        """
        A function to compare learning curves between different blockspans within a subjects_EMA object.
        """
    
        if group_names is None:
            group_names = [f'Group_{k}' for k in range(len(list_of_block_ranges))]
            
        if experienced_EV:
            acc_name = 'acc_exp'
            title_modif = 'Experienced EV'
            save_modif  = 'exp'
        else:
            acc_name = 'acc'
            title_modif = 'Objective EV'
            save_modif  = 'obj'
        
        fig = go.Figure()
                
        for k, block_range in enumerate(list_of_block_ranges):
            accs = []
            for n in range(self.N):
                df = self.create_subject_df(n=n)
                
                df = df[df['feedback'] > .5]
                df = df[df['block'].between(block_range[0], block_range[1], inclusive=True)]
                
                if new_lcurve:                                    
                    df['min_num_choice'] = np.minimum(df['num_choice1'], df['num_choice2'])
                    
                    acc = df.groupby('min_num_choice')[acc_name].mean()
                    
                    acc_smoothed = acc.rolling(smoothing_window, min_periods = 1).mean()                   
                    
                    df['min_num_appear'] = np.minimum(df['num_appear1'], df['num_appear2'])
                    
                    acc = df.groupby('min_num_appear')[acc_name].mean()
                    acc_smoothed = acc.rolling(smoothing_window, min_periods = 1).mean()
                    
                    accs.append(acc_smoothed.values[smoothing_window-1:])
                else:
                    df = df[['block', 'trial', acc_name]]
                    df['feedback_trial'] = df.groupby('block').cumcount()
                                
                    acc = df.groupby('feedback_trial')[acc_name].mean()
                    
                    acc_smoothed = acc.rolling(smoothing_window, min_periods = 1).mean()
                    
                    accs.append(acc_smoothed.values[smoothing_window-1:])
                    
            acc_avg = np.nanmean(accs, axis = 0)
            acc_std = np.nanstd(accs, axis = 0) * 1.96 / np.sqrt(len(accs))
                     
            fig.add_trace(go.Scatter(x = np.arange(smoothing_window - 1, acc_avg.shape[0]),
                                    y = acc_avg,
                                    error_y=dict(
                                        type='data',
                                        symmetric=True,
                                        array= acc_std[smoothing_window-1:],
                                        thickness = .5),
                                    name = group_names[k],
                                    mode = 'lines'))
            
        
        if new_lcurve:
            xaxis_title = "Minimum number of appearances"
        else:
            xaxis_title = "Feedback trial"
            
        
        fig.update_layout(
            title = f"Group learning curves ({smoothing_window}-smoothed), {title_modif}",
            xaxis_title = xaxis_title,
            yaxis_title = 'Average smoothed accuracy',
            yaxis={
                'range': [0.5, 1.]  
            })
            
        if show_figs:
            fig.show()
            
        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
            
            if new_lcurve:
                name = f"grouped_new_lcurve_{save_modif}.jpeg"
            else:
                name = f"grouped_lcurve_{save_modif}.jpeg"
            fig.write_image(os.path.join(save_folder, name))    

            
    
    def plot_nf_acc_day(self, experienced_EV = False, save_folder = None, show_figs = True, return_values = False, name_modifier = ""):
        """
        Plots a time series figure with the average no-feedback accuracy per day, for each subject.
        
        Assumes every two consecutive blocks form a day.
        """
        
        fig = go.Figure()
        fig_agg = go.Figure()
        fig_hist = go.Figure()
        
        accs = []
        overall_acc = []
        
        if experienced_EV:
            acc_name = 'acc_exp'
            title_modif = 'Experienced EV'
            save_modif  = 'exp'
        else:
            acc_name = 'acc'
            title_modif = 'Objective EV'
            save_modif  = 'obj'
        
        for n in range(self.N):
            df = self.create_subject_df(n=n)
            
            #df = df[df['feedback'] < .5]
            df = df[(df['feedback'] < .5) & (df['stim1'] > 17.) & ((df['stim2'] > 17.) | (df['stim2'] < -10.))]
            day_acc = df.groupby('session')[acc_name].mean()
                        
            accs.append(day_acc)
            
            fig.add_trace(go.Scatter(x = np.arange(day_acc.shape[0]),
                                    y = day_acc,
                                    mode = 'markers+lines',
                                    name = self.subject_names[n],
                                    marker_color = self.colors[n]))
                                    
            nopractice_stims = (df['stim1'] >= self.min_stim) & (df['stim2'] >= self.min_stim)
                                    
            fig_agg.add_trace(go.Bar(x = ['NF acc'],
                                y = [df.loc[nopractice_stims, acc_name].mean()],
                                name = self.subject_names[n],
                                marker_color = self.colors[n]))
                                  
            
            
            if show_figs:
                print(f"{self.subject_names[n]}) {df[acc_name].mean()}")
                
            
            overall_acc.append(df.loc[nopractice_stims, acc_name].mean())        
      
        fig_hist.add_trace(go.Histogram(x = overall_acc))
        
        fig.update_layout(title = f'Average No-Feedback Accuracy day-by-day, {title_modif}',
                         yaxis_title = 'Acc',
                         xaxis_title = 'Day',
                         yaxis = {'range':[.5,1.]})
                         
        fig_agg.update_layout(title = f'Average No-Practice No-Feedback Accuracy, {title_modif}',
                         yaxis_title = 'Acc',
                         xaxis_title = 'NF-acc',
                         yaxis = {'range':[.5,1.]})
        
        fig_hist.update_layout(title = f'Average No-Practice No-Feedback Accuracy, {title_modif}',
                         xaxis_title = 'Test accuracy',
                         yaxis_title = 'Participant'
                         )
                         
        if show_figs:
            fig.show()  
            fig_agg.show()
            fig_hist.show()             
        
        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
        
            save_path = os.path.join(save_folder, f'nf_acc_day_{save_modif}_{name_modifier}.jpeg')
            fig.write_image(save_path)
            
            save_path = os.path.join(save_folder, f'nf_acc_{save_modif}_{name_modifier}.jpeg')
            fig_agg.write_image(save_path)
            
        if return_values:
            return accs, overall_acc
    
    
    
    def compare_nf_acc_day(self, s_dat_b, experienced_EV = False, save_folder = None, group_names = ["Group_1", "Group_2"]):
    
        accs, overall_acc = self.plot_nf_acc_day(experienced_EV = experienced_EV, save_folder = save_folder, show_figs = False, return_values = True, name_modifier = group_names[0])
        accs_b, overall_acc_b = s_dat_b.plot_nf_acc_day(experienced_EV = experienced_EV, save_folder = save_folder, show_figs = False, return_values = True, name_modifier = group_names[1])
               
        df_acc = pd.concat(accs, axis = 1); N0 = len(accs)
        df_acc_b = pd.concat(accs_b, axis = 1); N1 = len(accs_b)
        
        if experienced_EV:
            acc_name = 'acc_exp'
            title_modif = 'Experienced EV'
            save_modif  = 'exp'
        else:
            acc_name = 'acc'
            title_modif = 'Objective EV'
            save_modif  = 'obj'
        
        layout = Layout(plot_bgcolor='rgba(0,0,0,0)')
        #fig = go.Figure()
        fig = go.Figure(layout=layout)
        
        fig.add_trace(go.Scatter(x = np.arange(df_acc.shape[0]),
                                y = df_acc.mean(axis = 1),
                                error_y=dict(type='data', array= 1.96 * df_acc.std(axis = 1) / np.sqrt(N0), width=12),
                                mode = 'markers+lines',
                                name = group_names[0],
                                line=dict(color='rgba(255, 20, 147, 0.8)', width=12),showlegend=False))
                                
        fig.add_trace(go.Scatter(x = np.arange(df_acc_b.shape[0]),
                                y = df_acc_b.mean(axis = 1),
                                error_y=dict(type='data', array= 1.96 * df_acc_b.std(axis = 1) / np.sqrt(N1), width=12),
                                mode = 'markers+lines',
                                name = group_names[1],
                                line=dict(color='rgba(28, 134, 238, 0.5)', width=12),showlegend=False))

        fig.update_layout(
            xaxis_title = 'Day',
            yaxis_title = 'Test accuracy',
            font=dict(
            family="Arial",
            size=50,
            color='black'
            ),
            legend_traceorder="normal",
            yaxis={
                'range': [0.5, 1]
                },
            xaxis={
                'range': [-0.1, 23.1]},
            margin=dict(l=0, r=25),
            legend=dict(
                #tracegroupgap=10,  # Adjust the gap between legend entries
                font=dict(
                size=60  # Adjust the font size of the legend
            ))            
            )
                         
        fig.update_xaxes(showline=True, linewidth=2.7, linecolor='black', ticks='outside', mirror=True, tickvals=[0, 5.75, 11.5, 17.25, 23],ticktext=[4,10,16,22,28], tickmode='array', tickfont=dict(size=42),ticklen=12, tickwidth=3, title_standoff=30)
        fig.update_yaxes(showline=True, linewidth=2.7, linecolor='black', ticks='outside', mirror=True, tickfont=dict(size=42),ticklen=12, tickwidth=3, title_standoff=30)
        fig.show()
        
        
        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
        
            save_path = os.path.join(save_folder, f'nf_acc_day_{title_modif}_{group_names[0]}_vs_{group_names[1]}.svg')
            fig.write_image(save_path, width=1547, height=1105)     

        ar_acc_agg = np.array(overall_acc)
        ar_acc_agg_b = np.array(overall_acc_b)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(x = ['NF-acc'],
                            y = [np.nanmean(ar_acc_agg)],
                            error_y=dict(type='data', array= [1.96 * np.nanstd(ar_acc_agg) / np.sqrt(N0)]),
                            name = group_names[0]))
                            
        fig.add_trace(go.Bar(x = ['NF-acc'],
                            y = [np.nanmean(ar_acc_agg_b)],
                            error_y=dict(type='data', array= [1.96 * np.nanstd(ar_acc_agg_b) / np.sqrt(N1)]),
                            name = group_names[1]))
                            
        fig.update_layout(title = f'Average No-Feedback Accuracy overall, {title_modif}',
                         yaxis_title = 'Acc',
                         xaxis_title = 'Group',
                         yaxis = {'range':[.5,1.]})
                         
        fig.show()
        
        if save_folder is not None:        
            save_path = os.path.join(save_folder, f'nf_acc_agg_{save_modif}_{group_names[0]}_vs_{group_names[1]}.svg')
            fig.write_image(save_path)
    ##
            
    def compare_f_acc_day(self, s_dat_b, experienced_EV = False, save_folder = None, group_names = ["Group_1", "Group_2"]):
    
        accs, overall_acc = self.plot_f_acc_day(experienced_EV = experienced_EV, save_folder = save_folder, show_figs = False, return_values = True, name_modifier = group_names[0])
        accs_b, overall_acc_b = s_dat_b.plot_f_acc_day(experienced_EV = experienced_EV, save_folder = save_folder, show_figs = False, return_values = True, name_modifier = group_names[1])
               
        df_acc = pd.concat(accs, axis = 1); N0 = len(accs)
        df_acc_b = pd.concat(accs_b, axis = 1); N1 = len(accs_b)
        
        if experienced_EV:
            acc_name = 'acc_exp'
            title_modif = 'Experienced EV'
            save_modif  = 'exp'
        else:
            acc_name = 'acc'
            title_modif = 'Objective EV'
            save_modif  = 'obj'
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x = np.arange(df_acc.shape[0]),
                                y = df_acc.mean(axis = 1),
                                error_y=dict(type='data', array= 1.96 * df_acc.std(axis = 1) / np.sqrt(N0)),
                                mode = 'markers+lines',
                                name = group_names[0]))
                                
        fig.add_trace(go.Scatter(x = np.arange(df_acc_b.shape[0]),
                                y = df_acc_b.mean(axis = 1),
                                error_y=dict(type='data', array= 1.96 * df_acc_b.std(axis = 1) / np.sqrt(N1)),
                                mode = 'markers+lines',
                                name = group_names[1]))
                                
        fig.update_layout(title = f'Average Feedback Accuracy session-by-session, {title_modif}',
                         yaxis_title = 'Acc',
                         xaxis_title = 'Session',
                         yaxis = {'range':[.5,1.]})
                         
        fig.show()
        
        
        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
        
            save_path = os.path.join(save_folder, f'nf_acc_day_{title_modif}_{group_names[0]}_vs_{group_names[1]}.jpeg')
            fig.write_image(save_path)   

        ar_acc_agg = np.array(overall_acc)
        ar_acc_agg_b = np.array(overall_acc_b)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(x = ['NF-acc'],
                            y = [np.nanmean(ar_acc_agg)],
                            error_y=dict(type='data', array= [1.96 * np.nanstd(ar_acc_agg) / np.sqrt(N0)]),
                            name = group_names[0]))
                            
        fig.add_trace(go.Bar(x = ['NF-acc'],
                            y = [np.nanmean(ar_acc_agg_b)],
                            error_y=dict(type='data', array= [1.96 * np.nanstd(ar_acc_agg_b) / np.sqrt(N1)]),
                            name = group_names[1]))
                            
        fig.update_layout(title = f'Average Feedback Accuracy overall, {title_modif}',
                         yaxis_title = 'Acc',
                         xaxis_title = 'Group',
                         yaxis = {'range':[.5,1.]})
                         
        fig.show()
        
        if save_folder is not None:        
            save_path = os.path.join(save_folder, f'nf_acc_agg_{save_modif}_{group_names[0]}_vs_{group_names[1]}.jpeg')
            fig.write_image(save_path) 
            
    ##
        
            
    def strat_nf_acc_day(self, list_of_block_ranges, experienced_EV = False, save_folder = None, group_names = None):
    
        if group_names is None:
            group_names = [f"Group {k}" for k in range(len(list_of_block_ranges))]
            
        if experienced_EV:
            acc_name = 'acc_exp'
            title_modif = 'Experienced EV'
            save_modif  = 'exp'
        else:
            acc_name = 'acc'
            title_modif = 'Objective EV'
            save_modif  = 'obj'
            
        fig_day = go.Figure()
        fig = go.Figure()
            
        for k, block_range in enumerate(list_of_block_ranges):
            accs = []
            overall_acc = []
            for n in range(self.N):
                df = self.create_subject_df(n=n)
                
                df = df[df['feedback'] < .5]
                df = df[df['block'].between(block_range[0], block_range[1], inclusive=True)]    
            
                day_acc = df.groupby('session')[acc_name].mean()
                        
                accs.append(day_acc)

                overall_acc.append(df[acc_name].mean())
            
            df_acc = pd.concat(accs, axis = 1); N = len(accs)
            
            fig_day.add_trace(go.Scatter(x = np.arange(df_acc.shape[0]),
                        y = df_acc.mean(axis = 1),
                        error_y=dict(type='data', array= 1.96 * df_acc.std(axis = 1) / np.sqrt(N)),
                        mode = 'markers+lines',
                        name = group_names[k]))

            ar_acc_agg = np.array(overall_acc)
            
            fig.add_trace(go.Bar(x = ['NF-acc'],
                                y = [np.nanmean(ar_acc_agg)],
                                error_y=dict(type='data', array= [1.96 * np.nanstd(ar_acc_agg) / np.sqrt(N)]),
                                name = group_names[k]))
                            
        fig_day.update_layout(title = f'Average No-Feedback Accuracy day-by-day, {title_modif}',
                     yaxis_title = 'Acc',
                     xaxis_title = 'Day',
                     yaxis = {'range':[.5,1.]})
                        
        fig.update_layout(title = f'Average No-Feedback Accuracy overall, {title_modif}',
                     yaxis_title = 'Acc',
                     xaxis_title = 'NF-acc',
                     yaxis = {'range':[.5,1.]})
                     
        fig_day.show()
        fig.show()
        
        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
        
            save_path = os.path.join(save_folder, f'nf_acc_day_{title_modif}_{group_names[0]}_vs_{group_names[1]}.jpeg')
            fig_day.write_image(save_path)   
            
            save_path = os.path.join(save_folder, f'nf_acc_agg_{save_modif}_{group_names[0]}_vs_{group_names[1]}.jpeg')
            fig.write_image(save_path) 
                
        
            
    def plot_f_acc_day(self, experienced_EV = False, save_folder = None, show_figs = True, return_values = False, name_modifier = ""):
        """
        Plots a time series figure with the average feedback accuracy per day.
        
        """
        
        fig = go.Figure()
        fig_agg = go.Figure()
        fig_hist = go.Figure()
        
        if experienced_EV:
            acc_name = 'acc_exp'
            title_modif = 'Experienced EV'
            save_modif  = 'exp'
        else:
            acc_name = 'acc'
            title_modif = 'Objective EV'
            save_modif  = 'obj'
        
        accs = []
        overall_acc = []
        
        for n in range(self.N):
            df = self.create_subject_df(n=n)
            
            #df = df[df['feedback'] > .5]
            df = df[(df['feedback'] > .5) & (df['stim1'] > 17.) & (df['stim2'] > 17.)]
            
            day_acc = df.groupby('session')[acc_name].mean()
            
            accs.append(day_acc)
            
            fig.add_trace(go.Scatter(x = np.arange(day_acc.shape[0]),
                                    y = day_acc,
                                    mode = 'markers+lines',
                                    name = self.subject_names[n],
                                    marker_color = self.colors[n]))
                                    
            fig_agg.add_trace(go.Bar(x = ['F acc'],
                                y = [df[acc_name].mean()],
                                name = self.subject_names[n],
                                marker_color = self.colors[n]))                                
            
            
            if show_figs:
                print(f"{self.subject_names[n]}) {df[acc_name].mean()}")
                
            overall_acc.append(df[acc_name].mean())
            
        fig_hist.add_trace(go.Histogram(x = overall_acc, marker=dict(color='gray', line=dict(color='black', width=1))))
        
        fig.update_layout(title = f'Average Feedback Accuracy day-by-day, {title_modif}',
                         yaxis_title = 'Acc',
                         xaxis_title = 'Day', yaxis = {'range':[.5,1.]})
                         
        fig_agg.update_layout(title = f'Average Feedback Accuracy, {title_modif}',
                         yaxis_title = 'Acc',
                         xaxis_title = 'F-acc', yaxis = {'range':[.5,1.]})
        
        
        fig_hist.update_layout(title = '',
                         yaxis= dict(title='Participants',title_font=dict(size=30)),
                         xaxis= dict(title = 'Mean learning accuracy',title_font=dict(size=30), range=[0.65,0.95]),
                         plot_bgcolor='white')
        fig_hist.update_xaxes(showline=True, linewidth=2.7, linecolor='black', ticks='outside', mirror=True, tickfont=dict(size=20), ticklen=12, tickwidth=3, title_standoff=30, tickvals=[0.65,0.7,0.75,0.8,0.85,0.9,0.95])
        fig_hist.update_yaxes(showline=True, linewidth=2.7, linecolor='black', ticks='outside', mirror=True, tickfont=dict(size=20), ticklen=12, tickwidth=3, title_standoff=30)
        
        
                       

        if show_figs:
            fig.show()  
            fig_agg.show()  
            fig_hist.show()            
        
        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
        
            save_path = os.path.join(save_folder, f'f_acc_day_{save_modif}_{name_modifier}.jpeg')
            fig.write_image(save_path)
            
            save_path = os.path.join(save_folder, f'f_acc_{save_modif}_{name_modifier}.jpeg')
            fig_agg.write_image(save_path)
            
            save_path = os.path.join(save_folder, f'hist_{save_modif}_{name_modifier}.svg')
            fig_hist.write_image(save_path)
            
        if return_values:
            return accs, overall_acc
            
    def plot_nf_acc_type_stratified(self, experienced_EV = False, save_folder = None, show_figs = True, return_values = False, name_modifier = ""):
        """
        Displays the nf-accuracy stratified by stimulus type match-up.
        
        Also displays the choice rate.
        """
        fig = go.Figure()
        fig_choice = go.Figure()
        
        if experienced_EV:
            acc_name = 'acc_exp'
            title_modif = 'Experienced EV'
            save_modif  = 'exp'
        else:
            acc_name = 'acc'
            title_modif = 'Objective EV'
            save_modif  = 'obj'
        
        nf_accs = []
        
        for n in range(self.N):
            df = self.create_subject_df(n=n)
            
            df = df[df['feedback'] < .5]
            
            
            df['low_type'] = np.minimum(df['type1'], df['type2'])
            df['high_type'] = np.maximum(df['type1'], df['type2'])
            
            acc = df.groupby(['low_type', 'high_type'])[acc_name].mean()
            #print(acc)
            #print(acc.index)
            
            fig.add_trace(go.Bar(x=[f"({s[0]}, {s[1]})" for s in acc.index.tolist()],
                                y = acc,
                                name = self.subject_names[n],
                                marker_color = self.colors[n]))
            
            df['choose_lower_type'] = (df['C_st'] * df['type2'] + (1 - df['C_st']) * df['type1']) < ((1 - df['C_st']) * df['type2'] + df['C_st'] * df['type1'])
            
            df_choose_lower = df.groupby(['low_type', 'high_type'])['choose_lower_type'].mean()
            
            fig_choice.add_trace(go.Bar(x=[f"({s[0]}, {s[1]})" for s in df_choose_lower.index.tolist()],
                                y = df_choose_lower,
                                name = self.subject_names[n],
                                marker_color = self.colors[n]))
                                
            nf_accs.append(df_choose_lower)
        
        fig.update_layout(title = f'No-feedback Accuracy stratified by stimulus types, {title_modif}',
                         xaxis_title = 'Pair type',
                         yaxis_title = 'Accuracy',
                         yaxis = {'range': [.6,1.]})
        
        
        fig_choice.update_layout(title = 'No-feedback Choice Rate of Lower Type',
                         xaxis_title = 'Pair type',
                         yaxis_title = 'Choice Rate of First Type',
                         yaxis = {'range': [0.,1.]})
                         
        if show_figs:
            fig.show()
            fig_choice.show()
        
        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
        
            save_path = os.path.join(save_folder, f'nf_acc_type_stratified_{save_modif}_{name_modifier}.jpeg')
            fig.write_image(save_path)
            
            save_path = os.path.join(save_folder, f'nf_choice_rate_{name_modifier}.jpeg')
            fig_choice.write_image(save_path)
        
        if return_values:
            return nf_accs
    
    def compare_nf_acc_type_stratified(self, s_dat_b, save_folder = None, group_names = None):

        if group_names is None:
            group_names = ['Group_1', 'Group_2']

        nf_accs   =    self.plot_nf_acc_type_stratified(save_folder = save_folder, show_figs = False, return_values = True, name_modifier = f'{group_names[0]}')
        nf_accs_b = s_dat_b.plot_nf_acc_type_stratified(save_folder = save_folder, show_figs = False, return_values = True, name_modifier = f'{group_names[1]}')

        df_acc   = pd.concat(nf_accs, axis = 1); N0 = len(df_acc)
        df_acc_b = pd.concat(nf_accs_b, axis = 1); N1 = len(df_acc_b)

        fig = go.Figure()

        fig.add_trace(go.Bar(x=[f"({s[0]}, {s[1]})" for s in df_acc.index.tolist()],
                                y = df_acc.mean(axis = 1),
                                error_y=dict(type='data', array= 1.96 * df_acc.std(axis = 1) / np.sqrt(N0)),
                                name = group_names[0]))

        fig.add_trace(go.Bar(x=[f"({s[0]}, {s[1]})" for s in df_acc_b.index.tolist()],
                                y = df_acc_b.mean(axis = 1),
                                error_y=dict(type='data', array= 1.96 * df_acc_b.std(axis = 1) / np.sqrt(N1)),
                                name = group_names[1]))

        fig.update_layout(title = 'No-feedback Choice Rate of Lower Type',
                             xaxis_title = 'Pair type',
                             yaxis_title = 'Choice Rate of First Type',
                             yaxis = {'range': [0.,1.]})

        fig.show()

        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)

            save_path = os.path.join(save_folder, f'nf_choice_rate_{group_names[0]}_vs_{group_names[1]}.jpeg')
            fig.write_image(save_path)           
                         
                                           
    def plot_nf_acc_min_appearances(self, smoothing_window = 5, save_folder = None, show_figs = True, return_values = False, name_modifier = ""):
        """
        Plots the average accuracy over different no-feedback trials vs the minimum number of appearances on nf-trials.
        """
        
        fig = go.Figure()
        fig_smoothed = go.Figure()
        
        accs = []
        
        for n in range(self.N):
            df = self.create_subject_df(n=n)
            
            df = df[df['feedback'] < .5]
            
            df['min_num_nf_appear'] = np.minimum(df['num_nf_appear1'], df['num_nf_appear2'])
            acc = df.groupby('min_num_nf_appear')['acc'].mean()
            
            acc_smoothed = acc.rolling(smoothing_window, min_periods = 1).mean()
            accs.append(acc_smoothed)
            
            fig.add_trace(go.Scatter(x = acc.index,
                                    y = acc,
                                    name = self.subject_names[n],
                                    marker_color = self.colors[n],
                                    mode = 'lines'))
            
            fig_smoothed.add_trace(go.Scatter(x = acc_smoothed.index,
                                    y = acc_smoothed,
                                    name = self.subject_names[n],
                                    marker_color = self.colors[n],
                                    mode = 'lines'))
            
        fig.update_layout(title = 'Nf Acc vs minimum number of nf appearances (unsmoothed)',
                         xaxis_title = 'Minimum Number of Appearances on nf trials',
                         yaxis_title = 'Accuracy')
            
        
        fig_smoothed.update_layout(title = f'Nf Acc vs minimum number of nf appearances ({smoothing_window}-smoothed)',
                         xaxis_title = 'Minimum Number of Appearances on nf trials',
                         yaxis_title = 'Accuracy')
        
        if show_figs:  
            fig.show()        
            fig_smoothed.show()
        
            print("Could perhaps compare increasing accuracy by 'Q-value at the end of learning'?")
        
        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
        
            save_path = os.path.join(save_folder, f'acc_by_nf_appearances_{name_modifier}.jpeg')
            fig.write_image(save_path)
            
            save_path = os.path.join(save_folder, f'acc_by_nf_appearances_smoothed_{name_modifier}.jpeg')
            fig_smoothed.write_image(save_path)
            
        if return_values:
            return accs
            
    def compare_nf_acc_min_appearances(self, s_dat_b, smoothing_window = 5, save_folder = None, group_names = ['Group_1', 'Group_2']):
    
        accs = self.plot_nf_acc_min_appearances(smoothing_window = smoothing_window, save_folder = save_folder, show_figs = False, return_values = True, name_modifier = group_names[0])
        accs_b = s_dat_b.plot_nf_acc_min_appearances(smoothing_window = smoothing_window, save_folder = save_folder, show_figs = False, return_values = True, name_modifier = group_names[1])
        
        df_acc = pd.concat(accs, axis = 1); N0 = len(accs)
        df_acc_b = pd.concat(accs_b, axis = 1); N1 = len(accs_b)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x = df_acc.index,
                                 y = df_acc.mean(axis = 1),
                                 name = group_names[0],
                                 error_y=dict(type='data', array= 1.96 * df_acc.std(axis = 1) / np.sqrt(N0))))
                                 
        fig.add_trace(go.Scatter(x = df_acc_b.index,
                                 y = df_acc_b.mean(axis = 1),
                                 name = group_names[1],
                                 error_y=dict(type='data', array= 1.96 * df_acc_b.std(axis = 1) / np.sqrt(N1))))
                                 
        fig.update_layout(title = f'Nf Acc vs minimum number of nf appearances ({smoothing_window}-smoothed)',
                         xaxis_title = 'Minimum Number of Appearances on nf trials',
                         yaxis_title = 'Accuracy',
                         yaxis = {'range':[.70,1.]})
                         
        fig.show()
        
        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
        
            save_path = os.path.join(save_folder, f'acc_by_nf_appearances_{group_names[0]}_vs_{group_names[1]}.jpeg')
            fig.write_image(save_path)
            
    def plot_coin_type_choice_reps(self, save_figs = False):
        """
        Displays histograms of the stimuli's number of choices, stratified by coin type.
        """

        fig_type_rep = {k : go.Figure() for k in range(4)}

        for n in range(self.N):
            df = self.create_subject_df(n=n)

            df = df[df['feedback'] > .5]

            C_num = df.groupby('C')['C'].count()
            C_type = df.groupby('stim1')['type1'].first()

            C = pd.concat((C_num, C_type), axis = 1)

            #print(np.sum(C['type1'].isna()))
            for k, gp in C.groupby('type1'):
                bins = [-.5, .5, 1.5,2.5,3.5,4.5, 5.5, 10.5, 20.5, 30.5, 150.5] 
                counts, bins = np.histogram(gp['C'], bins=bins)
                x = ["0", "1","2", "3", "4", "5", "6-10", "11-20", "21-30",">30"]

                fig_type_rep[int(k)].add_trace(go.Bar(name = self.subject_names[n],
                                                         marker_color = self.colors[n],
                                                         x = x,
                                                         y = counts
                ))

        types = ['A', 'B', 'C', 'D']
        for k in range(4):
            fig_type_rep[k].update_layout(title = f"Histogram of choice count of stimuli type {types[k]}, feedback trials",
                              xaxis_title = "# Choices",
                              yaxis_title = "Count of stimuli")
            fig_type_rep[k].show()
            if save_figs:
                fig_type_rep[k].write_image(os.path.join('subject_statistics', f'type_choice_count_histogram_{types[k]}.jpeg'))

    def plot_nf_choice_statistics_block_stratified(self, cuts = [0., 2.5 , 4.5, 6.5, 8.5, 10.5, 14.5, 20.5, 45.5, 61.5], save_folder = None):
        """
        TODO: update function to make documentation, output and code more clean.

        Displays a few statistics relating to the accuracy of non-feedback trials over time:
        - For each subject, displays a histogram of the difference in time 
          since the last learning block between the chosen and unchosen stimulus.
        - Displays a figure with the fraction of unique versus total number of stimulus pairs on no-feedback trials, per time group.
        - Displays a figure with the number of unequal stim pairs stratified over the amount of blocks since 
          learningy.
        - Displays a Figure with the accuracy in each time group, over unique pairs.
          Takes the minimum number of blocks since the last learning block of the two stimuli.
          Accuracy is computed as the average of the average accuracy of unqieu pairs.

        TODO: better to look at time in bewteen repetitions. See other functions.

        Arguments:
        - cuts: list, specified the cutting points for number of blocks since learning.
        - save_figs: saves figures if True.
        """
        fig_uniqueness_rate = go.Figure()
        fig_avg_unique_acc = go.Figure()
        fig_unique_total    = go.Figure()
        fig_unique_n       = go.Figure()

        #choice_save_folder = os.path.join('subject_statistics', 'subject_choice')
        if save_folder is not None and not os.path.exists(save_folder):
            os.mkdir(save_folder)

        for n in range(self.N):
            df = self.create_subject_df(n = n)

            df = df[df['feedback'] < 0.5]

            # Last feedback times for the stimuli
            df['time_since_last_lblock1'] = df['block'].values - self.last_learning_block.loc[df['stim1'].values, self.subject_names[n]].values
            df['time_since_last_lblock2'] = df['block'].values - self.last_learning_block.loc[df['stim2'].values, self.subject_names[n]].values

            df['min_blocks_since_last_lblock'] = np.minimum(df['time_since_last_lblock1'].values,df['time_since_last_lblock2'].values) 
            df['max_blocks_since_last_lblock'] = np.maximum(df['time_since_last_lblock1'].values,df['time_since_last_lblock2'].values) 

            df['low_stim'] = np.minimum(df['stim1'].values, df['stim2'].values)
            df['high_stim'] = np.maximum(df['stim1'].values, df['stim2'].values)        

            # Compute the number of unique pairs and their average accuracy per time group:
            df_acc = df.groupby(pd.cut(df['min_blocks_since_last_lblock'], cuts))
            df_gpd_acc = df_acc.apply(lambda x: x.groupby(['low_stim', 'high_stim'])['acc'].mean()).groupby('min_blocks_since_last_lblock')#.droplevel(['low_stim', 'high_stim'])

            total = df_acc['acc'].apply(lambda x: x.shape[0])
            acc_agr = df_acc['acc'].apply(lambda x: x.sum())
            total_unequal = df_acc['acc'].apply(lambda x: x.shape[0] - x.isna().sum())

            df_agr = pd.concat([total, acc_agr, total_unequal], axis = 1)
            df_agr.columns = ['total', 'acc_agr', 'total_unequal']

            total_gpd = df_gpd_acc.apply(lambda x: x.shape[0])
            acc_agr_gpd = df_gpd_acc.apply(lambda x: x.sum())
            total_unequal_gpd = df_gpd_acc.apply(lambda x: x.shape[0] - x.isna().sum())

            df_gpd = pd.concat([total_gpd, acc_agr_gpd, total_unequal_gpd], axis = 1)
            df_gpd.columns = ['total', 'acc_agr', 'total_unequal']

            if n == 0:
                print(f"Subject {self.subject_names[n]}:")
                display(df_agr)
                display(df_gpd)

            fig_uniqueness_rate.add_trace(go.Bar(name = self.subject_names[n],
                                                 x = [str(l) for l in df_agr.index],
                                                 y = df_gpd['total'] / df_agr['total'],
                                                marker_color = self.colors[n]))

            p = df_gpd['acc_agr'] / df_gpd['total_unequal']

            #p_std = 1.96 * np.sqrt(p * (1. - p) / df_gpd['total_unequal'])

            fig_avg_unique_acc.add_trace(go.Scatter(name = self.subject_names[n],
                                               x = [str(l) for l in df_agr.index],
                                               y = p,
                                               mode = 'lines+markers',
                                               #error_y = dict(type = 'data', array = p_std),
                                               marker_color = self.colors[n]))
            
            fig_unique_total.add_trace(go.Bar(name = self.subject_names[n],
                                         x = [str(l) for l in df_agr.index],
                                       y = df_gpd['total'],
                                       marker_color = self.colors[n]))
            
            fig_unique_n.add_trace(go.Bar(name = self.subject_names[n],
                                         x = [str(l) for l in df_agr.index],
                                       y = df_gpd['total_unequal'],
                                       marker_color = self.colors[n]))


            # Compute the distribution of choice time difference of the chosen stimulus:
            #fig_choice_time_diff = go.Figure()
            df['stim_time_diff'] = df['time_since_last_lblock2'] - df['time_since_last_lblock1']
            df['choice_time_diff'] = (df['C_st'] * 2. - 1.) * df['stim_time_diff']

            #df_unique = df.groupby(['low_stim', 'high_stim'])['stim_time_diff'].first()#.apply(lambda x: x['stim_time_diff'].first()) #, 'choice_time_diff'
            #display(df_unique)

            bins  = np.concatenate((np.arange(-40.5, -4.5, 2), (-2.5, 2.5), np.arange(4.5,40.5,2.)))
            counts, bins = np.histogram(df.choice_time_diff, bins=bins)
            bins = 0.5 * (bins[:-1] + bins[1:])

            fig = go.Figure(go.Bar(x=bins, y=counts, marker_color = self.colors[n]))#, color = self.colors[n])
            fig.update_layout(title = f"{self.subject_names[n]}) Time since block(choice) - Time since block(not-choice)",
                             xaxis_title = "block_diff",
                              yaxis_title = "count")
            fig.show()
            if save_folder is not None:
                fig.write_image(os.path.join(save_folder, f'choice_time_{self.subject_names[n]}.jpeg'))


            print(f"{self.subject_names[n]}) (#block_diff > 2)/(#block_diff < -2): {np.sum(df['choice_time_diff'] > 2) / np.sum(df['choice_time_diff'] < -2)}")


            # Compute the number of repetitions of pairs:
            #df_hists.append(df.groupby(['low_stim', 'high_stim'])['block'].count())#.hist(bins = np.arange(12) - .5))

        # Show figures:
        fig_uniqueness_rate.update_layout(
                title = "Fraction of unique non-feedback stimulus pairs in each group",
                xaxis_title = '#Blocks since last learning block, minimum of both stimuli',
                yaxis_title = 'Fraction Unique',
                yaxis={
                    'range': [0., 1.]  
                })
        fig_uniqueness_rate.show()

        fig_avg_unique_acc.update_layout(title = "Average of Unique Non-Feedback Stimuli Pair Accuracy",
                                        xaxis_title = '#Blocks since last learning block, minimum of both stimuli',
                                        yaxis_title = "Accuracy",
                                        yaxis={
                                            'range': [0.5,1.]
                                        })
        fig_avg_unique_acc.show()
        
        fig_unique_total.update_layout(title = "Number of Unique Pairs per group",
                                   xaxis_title = '#Blocks since last learning block, minimum of both stimuli',
                                    yaxis_title = "Number")#,
                                      #yaxis = {'range':[0,80]})
        fig_unique_total.show()

        fig_unique_n.update_layout(title = "Number of Unique Unequal Expected Value Pairs per group",
                                   xaxis_title = '#Blocks since last learning block, minimum of both stimuli',
                                    yaxis_title = "Number")#,
                                  #yaxis = {'range':[0,80]})
        fig_unique_n.show()

        if save_folder is not None:
            fig_uniqueness_rate.write_image(os.path.join(save_folder, 'nf_uniqueness_rate_lblock.jpeg'))
            fig_avg_unique_acc.write_image(os.path.join(save_folder, 'nf_uniqueness_acc_lblock.jpeg'))
            fig_unique_total.write_image(os.path.join(save_folder, 'nf_uniquene_lblock.jpeg'))
            fig_unique_n.write_image(os.path.join(save_folder, 'nf_uniquene_unEV_lblock.jpeg'))
            
    def plot_nf_delta_T(self, cuts = [0.5, 1.5, 3.6, 10.6, 20.6, 26.6, 30.6], save_folder = None):
        """
        This function does the following:
        - For non-feedback trials, computes the accuracy of no-feedback trials stratified on the minimum time since seeing either stimulus before.
          Displays the number of trials in each time group as well.
        - Does the same for the maximum.
        - Displays a histogram of the time in between stimulus presentations.

        It does this based on feedback time.
        """
        # Min, Max, Single Stim figure
        fig_tc = [go.Figure(), go.Figure(), go.Figure()] # Total Counts
        fig_uc = [go.Figure(), go.Figure(), go.Figure()] # Unequal Counts: counts of unequal expected value
        fig_acc = [go.Figure(), go.Figure()] # Average accuracy in each group.

        for n in range(self.N):
            df  = self.create_subject_df(n=n) 
            df = df[df['feedback'] < 1e-10] # Only non-feedback trials

            df['min_time_since_last_rep'] = np.minimum(df['time_since_last_rep1'], df['time_since_last_rep2'])
            df['max_time_since_last_rep'] = np.maximum(df['time_since_last_rep1'], df['time_since_last_rep2'])

            df_min = df.groupby(pd.cut(df['min_time_since_last_rep'], cuts))
            df_max = df.groupby(pd.cut(df['max_time_since_last_rep'], cuts))

            df_gps = [df_min, df_max]

            # Display the number of trials in each:
            for k, df_m in enumerate(df_gps):
                total_counts = df_m['block'].count()
                unequal_counts = df_m['acc'].apply(lambda x: x.shape[0] - x.isna().sum())
                acc = df_m['acc'].mean()

                fig_tc[k].add_trace(go.Bar(name = self.subject_names[n],
                                          marker_color = self.colors[n],
                                          x = [str(l) for l in total_counts.index],
                                          y = total_counts.values))

                fig_uc[k].add_trace(go.Bar(name = self.subject_names[n],
                                          marker_color = self.colors[n],
                                          x = [str(l) for l in total_counts.index],
                                          y = unequal_counts.values))

                fig_acc[k].add_trace(go.Scatter(name = self.subject_names[n],
                                          marker_color = self.colors[n],
                                          x = [str(l) for l in total_counts.index],
                                               y = acc.values))

            df1 = df.groupby(pd.cut(df['time_since_last_rep1'], cuts))
            df2 = df.groupby(pd.cut(df['time_since_last_rep2'], cuts))

            total_counts = df1['block'].count() + df2['block'].count()
            unequal_counts = df1['acc'].apply(lambda x: x.shape[0] - x.isna().sum()) + df2['acc'].apply(lambda x: x.shape[0] - x.isna().sum())

            fig_tc[2].add_trace(go.Bar(name = self.subject_names[n],
                                      marker_color = self.colors[n],
                                      x = [str(l) for l in total_counts.index],
                                     y = total_counts.values))
            fig_uc[2].add_trace(go.Bar(name = self.subject_names[n],
                                      marker_color = self.colors[n],
                                      x = [str(l) for l in total_counts.index],
                                     y = unequal_counts.values))



        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
                
        spec = ["minimum", "maximum", "all"]
        for k in range(3):
            fig_tc[k].update_layout(title = f"nf-trials in '\u0394 T ({spec[k]})' group (days)",
                                 xaxis_title = "Time since last presentation (days)",
                                 yaxis_title = "Count")

            fig_uc[k].update_layout(title = f"Unequal EV nf-trials in '\u0394 T ({spec[k]})' group (days)",
                                 xaxis_title = "Time since last presentation (days)",
                                 yaxis_title = "Count")
            if k < 2:
                fig_acc[k].update_layout(title = f"Accuracy of nf-trials in '\u0394 T ({spec[k]})' group (days)",
                                     xaxis_title = "Time since last presentation (days)",
                                     yaxis_title = "Accuracy",
                                        yaxis={
                                                'range': [0.5,1.]
                                            })

            fig_tc[k].show()
            fig_uc[k].show()
            if k < 2:
                fig_acc[k].show()

            if save_folder is not None:               
                save_path_tc = os.path.join(save_folder, f"nf_time_diff_total_count_{spec[k]}.jpeg")
                save_path_uc = os.path.join(save_folder, f"nf_time_diff_unequal_count_{spec[k]}.jpeg")
                save_path_acc = os.path.join(save_folder, f"nf_time_diff_acc_{spec[k]}.jpeg")

                fig_tc[k].write_image(save_path_tc)
                fig_uc[k].write_image(save_path_uc)
                if k < 2:
                    fig_acc[k].write_image(save_path_acc)


    def plot_nf_delta_block(self, cuts = [-0.5, 0.5, 2.5, 4.5, 6.5, 20.5, 40.5, 52.5, 160.5], save_figs = False, save_folder = 'data28'):
        """
        For data loaded with load_subjects_state() ONLY:
        - Displays a histogram of the number of blocks in betwen stimulus repetitions on no-feedback trials.
          1) For all trials
          2) For trials of unequal EV.
          
        Arguments:
        - cuts: list, gives the histogram bins. 
        - save_figs: boolean, if True, saves the figures to save_folder.
        - save_folder: path to folder to save figures.
        
        """
        fig_tc = go.Figure()
        fig_uc = go.Figure()

        for n in range(self.N):
            df = self.create_subject_df_state(n = n)
            df = df[df['feedback'] < .5]

            df['unequal'] = np.logical_and(df['type1'].isin([2.,3.]), df['type2'].isin([2.,3.]))

            df1 = df.groupby(pd.cut(df['num_blocks_since_last_block1'], cuts))
            df2 = df.groupby(pd.cut(df['num_blocks_since_last_block2'], cuts))


            total_counts = df1['block'].count() + df2['block'].count()
            display(total_counts)
            unequal_counts = df1['unequal'].apply(lambda x: x.shape[0] - x.sum()) + df2['unequal'].apply(lambda x: x.shape[0] - x.sum())
            display(unequal_counts)

            fig_tc.add_trace(go.Bar(name = self.subject_names[n],
                                          marker_color = self.colors[n],
                                          x = [str(l) for l in total_counts.index],
                                          y = total_counts.values))

            fig_uc.add_trace(go.Bar(name = self.subject_names[n],
                                          marker_color = self.colors[n],
                                          x = [str(l) for l in total_counts.index],
                                          y = unequal_counts.values))

        fig_tc.update_layout(title = f"nf-stimuli in '\u0394 block' group (days)",
                                 xaxis_title = "Num blocks since last presentation",
                                 yaxis_title = "Count")

        fig_uc.update_layout(title = f"Unequal EV nf-stimuli in '\u0394 block' group (days)",
                                 xaxis_title = "Num blocks since last presentation",
                                 yaxis_title = "Count")

        fig_tc.show()
        fig_uc.show()

        if save_figs:
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)

            save_path_tc = os.path.join(save_folder, f"nf_block_diff_total_count.jpeg")
            save_path_uc = os.path.join(save_folder, f"nf_block_diff_unequal_count.jpeg")

            fig_tc.write_image(save_path_tc)
            fig_uc.write_image(save_path_uc)
            
    def plot_nf_stim_rep(self, save_figs = False):
        """
        Computes the amount of repetitions of stimuli and stimuli pairs on no-feedback trials.
        Displays these in figures, with the option to save.

        The stimuli repetition counts are stored in:
        - self.stim_rep_count: (Nc, N) numpy array

        TODO: pair repetition count.
        """

        self.stim_rep_count = np.zeros((np.max(self.Nc), self.N))

        fig_rep = go.Figure()

        for n in range(self.N):
            df = self.create_subject_df(n=n)
            df = df[df['feedback'] < 1e-10]

            df1 = df.groupby('stim1')['block'].count()
            df2 = df.groupby('stim2')['block'].count()

            count = df1+df2

            self.stim_rep_count[count.index, n] = count.values

            bins = [-.5, .5, 5.5, 25.5, 75.5, 150.5, 1000.] 
            counts, bins = np.histogram(self.stim_rep_count[:,n], bins=bins)
            x = ["0", "1-5", "6-25", "26-75","76-150", ">150"]

            fig_rep.add_trace(go.Bar(name = self.subject_names[n],
                                          marker_color = self.colors[n],
                                          x = x,
                                          y = counts))

        fig_rep.update_layout(title = "Histogram of repetition count of stimuli on NF trials",
                              xaxis_title = "# Repetitions",
                              yaxis_title = "Count of stimuli")
        fig_rep.show()

        if save_figs:
            fig_rep.write_image(os.path.join('subject_statistics', 'nf_type_repetition.jpeg'))
 
    def plot_nf_acc_var_stratified(self, save_folder = None, var_stratification = 'valence_last'):
        """
        Computes the average accuracy on no-feedback trials, startified by high var_stratification (above median) and low var_stratification (below median).
        
        - var_stratification: string, name of the variable to stratify on.
        """
        
        fig = go.Figure()
        
        for n in range(self.N):
            df = self.create_subject_df(n=n)
            
            df['high_var'] = df[var_stratification] > np.median(df[var_stratification])
            
            df = df[df['feedback'] < .5]
            df = df[np.logical_and(df['stim1'] > 16,  df['stim2'] > 16)]
            
            acc_valence = df.groupby('high_var')['acc'].mean()
            
            fig.add_trace(go.Scatter(x =acc_valence.index, y = acc_valence.values,
                                    name = self.subject_names[n],
                                    marker_color = self.colors[n],
                                    mode = 'lines+markers'))
            
        fig.update_layout(title = f"No-feedback accuracy stratified by {var_stratification}",
                         xaxis_title = f"High {var_stratification}",
                         yaxis_title = "Accuracy",
                         yaxis = {'range':[.7,1.]})
        fig.show()
        
        if save_folder is not None:
            fig.write_image(os.path.join(save_folder, f'nf_acc_{var_stratification}.jpeg'))
            
    ### A lot of switch rate code - perhaps it could be made much shorter:
    def plot_switch_rate(self, paired = True, stratification = None, subject_names = None, name = 'Subjects', order_names = False, num_subjects_per_plot = 1, per_subject = True):
        """
        This is a wrapper function for the different switch_rate plot functions. See the individual functions for more info.
        
        Arguments:
        - paired: boolean, if True, plots the paired switch rate. Thesingle stimulus switch rate otherwise. The former means that after a given stimulus pair, we look at the next trial with the same pair and whether the same choice is made. If stimulus, we only look at the next trial wthat has that stimulus, and wether the same choice is made.
        - stratification: one of [None, 'block', 'trial']. No stratification if None. Stratification over blocks with 2 new stimuli or not if 'block'. Stratification over trials with 2 new stimuli in the given block or not.
        - subject_names: list of strings. Names of the subjects to display.
        - order_names: boolean, whether to order the subject_names alphabetically beforehand or not. This may be useuful to display '204' next to '204_fit' for example.
        - per_subject: boolean. If True, the switch rates after punishment ('pun'), neutral ('n') and reward ('r') are plotted together per subject. If False, plots bar charts for these three separately.
        - num_subjects_per_plot: if per_subject == True, the number of subjects to display per figure.
        
        'name' is currently not being used.
        
        TODO: write all the switch rate functions in one function. It's not very code-efficient now...
        """
        
        if paired:
            if stratification is None:
                self.plot_paired_switch_rate(subject_names = subject_names, name = name, order_names = order_names, num_subjects_per_plot = num_subjects_per_plot, per_subject = per_subject)
            elif stratification == 'block':
                self.plot_paired_switch_rate_block_stratified(subject_names = subject_names, name = name, order_names = order_names, num_subjects_per_plot = num_subjects_per_plot, per_subject = per_subject)
            elif stratification == 'trial':
                self.plot_paired_switch_rate_trials_stratified(subject_names = subject_names, name = name, order_names = order_names, num_subjects_per_plot = num_subjects_per_plot, per_subject = per_subject)
            else:
                sys.exit("Not a correct 'stratification' (None, 'block' or 'trial').")
        else:
            if stratification is None:
                self.plot_stim_switch_rate(subject_names = subject_names, name = name, order_names = order_names, num_subjects_per_plot = num_subjects_per_plot, per_subject = per_subject)
            elif stratification == 'block':
                self.plot_stim_switch_rate_block_stratified(subject_names = subject_names, name = name, order_names = order_names, num_subjects_per_plot = num_subjects_per_plot, per_subject = per_subject)
            elif stratification == 'trial':
                self.plot_stim_switch_rate_trials_stratified(subject_names = subject_names, name = name, order_names = order_names, num_subjects_per_plot = num_subjects_per_plot, per_subject = per_subject)
            else:
                sys.exit("Not a correct 'stratification' (None, 'block' or 'trial').")
     
    def plot_stim_switch_rate(self, subject_names = None, name = 'Subjects', order_names = False, num_subjects_per_plot = 1, per_subject = True):
        """
        If stimulus i was chosen in trial t with outcome o, this function checks whether 
        stimulus i was chosen as well the next time. For each outcome, this gets averaged over
        all stimuli to compute the average switch rate after punishment, neutral and reward.
        
        Ignores the first 6 blocks and no-feedback trials. 
        """

        subject_names, columns = self.subject_names_to_columns(subject_names, order_names)

        N = len(columns)

        # Store the switch rates per subject in here:
        pun_s_rate = np.zeros(N)
        n_s_rate   = np.zeros(N)
        r_s_rate   = np.zeros(N)
        
        # Store the standard deviation for the confidence intervals (normal approximation)):
        pun_sd = np.zeros(N)
        n_sd   = np.zeros(N)
        r_sd   = np.zeros(N)

        i = 0
        for n in columns:
            switch = np.zeros(3) # The amount of times a switch occured after punishment, neutral and reward respectively
            no_switch = np.zeros(3)
           
            t = 0
            while t < self.T - 1:
                while np.abs(self.feedback[t,n]) < 1e-10 and t < self.T - 1:
                    t += 1
                
                block = self.block[t,n]  
                c = self.C[t,n]
                check = 0
                if block > 5:
                    s = t + 1
                    while s < self.T and (self.block[s,n] - block == 0):
                        if np.abs(self.feedback[s,n]) < 1e-10:
                            s+=1
                        else:
                            if self.stims[s,0,n] == c or self.stims[s,1,n] == c:
                                check += 1
                                if self.C[s,n] == c:
                                    no_switch[int(self.o[t,n] + 1)] += 1
                                else:
                                    switch[int(self.o[t,n] + 1)] += 1
                                break
                            s += 1
                if check > 1:
                    print(f"Warning: more than 1 successor taken for t={t}, n={n}!")
                t += 1
                
            pun_s_rate[i] = switch[0] / (switch[0] + no_switch[0])
            n_s_rate[i] = switch[1] / (switch[1] + no_switch[1])
            r_s_rate[i] = switch[2] / (switch[2] + no_switch[2])
            
            pun_sd[i] = np.sqrt( pun_s_rate[i] * (1 - pun_s_rate[i]) / (switch[0] + no_switch[0]))
            n_sd[i] = np.sqrt( n_s_rate[i] * (1 - n_s_rate[i]) / (switch[1] + no_switch[1]))
            r_sd[i] = np.sqrt( r_s_rate[i] * (1 - r_s_rate[i]) / (switch[2] + no_switch[2]))
            
            i+=1
                    
        df = {'pun': pun_s_rate, 'n': n_s_rate, 'r': r_s_rate, 'subject': subject_names, 'columns': columns,
             'pun_sd':pun_sd, 'n_sd': n_sd, 'r_sd': r_sd}
        
        if per_subject:
            self.display_switch_rates_per_subject(df, num_subjects_per_plot = num_subjects_per_plot, switch_rate_type = 'Stim')
        else:
            self.display_switch_rate_all_subjects(df)
            
    def plot_stim_switch_rate_block_stratified(self, subject_names = None, name = 'Subjects', order_names = False, num_subjects_per_plot = 1, per_subject = True):
        """
        If stimulus i was chosen in trial t with outcome o, this function checks whether 
        stimulus i was chosen as well the next time. For each outcome, this gets averaged over
        all stimuli to compute the average switch rate after punishment, neutral and reward.
        
        Ignores the first 6 blocks and no-feedback trials. 
        
        Stratifies over blocks with at least two new stimuli and less.
        """

        subject_names, columns = self.subject_names_to_columns(subject_names, order_names)

        N = len(columns)

        # Store the switch rates per subject in here:
        pun_s_rate = np.zeros((2*N))
        n_s_rate   = np.zeros((2*N))
        r_s_rate   = np.zeros((2*N))
        
        # Store the standard deviation for the confidence intervals (normal approximation)):
        pun_sd = np.zeros((2*N))
        n_sd   = np.zeros((2*N))
        r_sd   = np.zeros((2*N))

        i = 0
        for n in columns:
            #print(f"\n{self.subject_names[n]}")
            switch = np.zeros((2,3)) # The amount of times a switch occured after punishment, neutral and reward respectively
            no_switch = np.zeros((2,3))
           
            t = 0
            b0 = self.block[0,n]
            while t < self.T - 1:
                while np.abs(self.feedback[t,n]) < 1e-10 and t < self.T - 1:
                    t += 1
                    
                block = self.block[t,n]
                if len(self.new_stimuli[n][block - b0]) > 1:
                    u = 0
                else: 
                    u = 1
                #print(f"{block:2d}) {u}")

                
                c = self.C[t,n]
                check = 0
                if block > 5:
                    s = t + 1
                    while s < self.T and (self.block[s,n] - block == 0):
                        if np.abs(self.feedback[s,n]) < 1e-10:
                            s+=1
                        else:
                            if self.stims[s,0,n] == c or self.stims[s,1,n] == c:
                                check += 1
                                if self.C[s,n] == c:
                                    no_switch[u,int(self.o[t,n] + 1)] += 1
                                else:
                                    switch[u,int(self.o[t,n] + 1)] += 1
                                break
                            s += 1
                if check > 1:
                    print(f"Warning: more than 1 successor taken for t={t}, n={n}!")
                t += 1
            
            for v in range(2):
                pun_s_rate[2*i + v] = switch[v,0] / (switch[v,0] + no_switch[v,0])
                n_s_rate[2*i + v] = switch[v,1] / (switch[v,1] + no_switch[v,1])
                r_s_rate[2*i + v] = switch[v,2] / (switch[v,2] + no_switch[v,2])

                pun_sd[2*i + v] = np.sqrt( pun_s_rate[2*i + v] * (1 - pun_s_rate[2*i + v]) / (switch[v,0] + no_switch[v,0]))
                n_sd[2*i + v] = np.sqrt( n_s_rate[2*i + v] * (1 - n_s_rate[2*i + v]) / (switch[v,1] + no_switch[v,1]))
                r_sd[2*i + v] = np.sqrt( r_s_rate[2*i + v] * (1 - r_s_rate[2*i + v]) / (switch[v,2] + no_switch[v,2]))
#                 pun_sd[2*i + v] = (switch[v,0] + no_switch[v,0]) * 0.01
#                 n_sd[2*i + v] = (switch[v,1] + no_switch[v,1]) *.01
#                 r_sd[2*i + v] = (switch[v,2] + no_switch[v,2]) * 0.01
            
           
            
            i+=1
#         print(pun_sd)
#         print(n_sd)
#         print(r_sd)
        
        subject_names = list(itertools.chain.from_iterable([[s_name + " (>=2 new)", s_name + " (<=1 new)"] for s_name in subject_names]))
        columns = list(itertools.chain.from_iterable([[clm, clm] for clm in columns]))
        df = {'pun': pun_s_rate, 'n': n_s_rate, 'r': r_s_rate, 'subject': subject_names, 'columns': columns,
             'pun_sd':pun_sd, 'n_sd': n_sd, 'r_sd': r_sd}
        
        if per_subject:
            self.display_switch_rates_per_subject(df, num_subjects_per_plot = 2*num_subjects_per_plot, switch_rate_type = 'Stim')
        else:
            self.display_switch_rate_all_subjects(df)
            
    def plot_stim_switch_rate_var_stratified(self, var_stratification = 'valence_last', subject_names = None, name = 'Subjects', order_names = False, num_subjects_per_plot = 1, per_subject = True):
        """
        If stimulus i was chosen in trial t with outcome o, this function checks whether 
        stimulus i was chosen as well the next time in any trials with that stimulus. For each outcome, this gets averaged over
        all stimuli to compute the average switch rate after punishment, neutral and reward.
        
        Ignores the first 6 blocks and no-feedback trials. 
        
        Stratifies over blocks where the 'var_stratification' (string) is higher or lower than the median.
        """

        subject_names, columns = self.subject_names_to_columns(subject_names, order_names)

        N = len(columns)

        # Store the switch rates per subject in here:
        pun_s_rate = np.zeros((2*N))
        n_s_rate   = np.zeros((2*N))
        r_s_rate   = np.zeros((2*N))
        
        # Store the standard deviation for the confidence intervals (normal approximation)):
        pun_sd = np.zeros((2*N))
        n_sd   = np.zeros((2*N))
        r_sd   = np.zeros((2*N))

        i = 0
        for n in columns:
            
            var_str = self.create_subject_df(n=n)[var_stratification].values
            var_high = (var_str > np.median(var_str)).astype('int64')
            
            switch = np.zeros((2,3)) # The amount of times a switch occured after punishment, neutral and reward respectively
            no_switch = np.zeros((2,3))
           
            t = 0
            b0 = self.block[0,n]
            while t < self.T - 1:
                while np.abs(self.feedback[t,n]) < 1e-10 and t < self.T - 1:
                    t += 1
                    
                u = var_high[t]
                block = self.block[t,n]
                
                c = self.C[t,n]
                check = 0
                if block > 5:
                    s = t + 1
                    while s < self.T and (self.block[s,n] - block == 0):
                        if np.abs(self.feedback[s,n]) < 1e-10:
                            s+=1
                        else:
                            if self.stims[s,0,n] == c or self.stims[s,1,n] == c:
                                check += 1
                                if self.C[s,n] == c:
                                    no_switch[u,int(self.o[t,n] + 1)] += 1
                                else:
                                    switch[u,int(self.o[t,n] + 1)] += 1
                                break
                            s += 1
                if check > 1:
                    print(f"Warning: more than 1 successor taken for t={t}, n={n}!")
                t += 1
            
            for v in range(2):
                pun_s_rate[2*i + v] = switch[v,0] / (switch[v,0] + no_switch[v,0])
                n_s_rate[2*i + v] = switch[v,1] / (switch[v,1] + no_switch[v,1])
                r_s_rate[2*i + v] = switch[v,2] / (switch[v,2] + no_switch[v,2])

                pun_sd[2*i + v] = np.sqrt( pun_s_rate[2*i + v] * (1 - pun_s_rate[2*i + v]) / (switch[v,0] + no_switch[v,0]))
                n_sd[2*i + v] = np.sqrt( n_s_rate[2*i + v] * (1 - n_s_rate[2*i + v]) / (switch[v,1] + no_switch[v,1]))
                r_sd[2*i + v] = np.sqrt( r_s_rate[2*i + v] * (1 - r_s_rate[2*i + v]) / (switch[v,2] + no_switch[v,2]))
#                 pun_sd[2*i + v] = (switch[v,0] + no_switch[v,0]) * 0.01
#                 n_sd[2*i + v] = (switch[v,1] + no_switch[v,1]) *.01
#                 r_sd[2*i + v] = (switch[v,2] + no_switch[v,2]) * 0.01
            
           
            
            i+=1
#         print(pun_sd)
#         print(n_sd)
#         print(r_sd)
        
        subject_names = list(itertools.chain.from_iterable([[s_name + f" (low {var_stratification})", s_name + f" (high {var_stratification})"] for s_name in subject_names]))
        columns = list(itertools.chain.from_iterable([[clm, clm] for clm in columns]))
        df = {'pun': pun_s_rate, 'n': n_s_rate, 'r': r_s_rate, 'subject': subject_names, 'columns': columns,
             'pun_sd':pun_sd, 'n_sd': n_sd, 'r_sd': r_sd}
        
        if per_subject:
            self.display_switch_rates_per_subject(df, num_subjects_per_plot = 2*num_subjects_per_plot, switch_rate_type = 'Stim')
        else:
            self.display_switch_rate_all_subjects(df)
            
                    
    def plot_stim_switch_rate_trials_stratified(self, subject_names = None, name = 'Subjects', order_names = False, num_subjects_per_plot = 1, per_subject = True):
        """
        If stimulus i was chosen in trial t with outcome o, this function checks whether 
        stimulus i was chosen as well the next time. For each outcome, this gets averaged over
        all stimuli to compute the average switch rate after punishment, neutral and reward.
        
        Ignores the first 6 blocks and no-feedback trials. 
        
        Stratifies over trials where there are two new stimuli versus less than two.
        """

        subject_names, columns = self.subject_names_to_columns(subject_names, order_names)

        N = len(columns)

        # Store the switch rates per subject in here:
        pun_s_rate = np.zeros((2*N))
        n_s_rate   = np.zeros((2*N))
        r_s_rate   = np.zeros((2*N))
        
        # Store the standard deviation for the confidence intervals (normal approximation)):
        pun_sd = np.zeros((2*N))
        n_sd   = np.zeros((2*N))
        r_sd   = np.zeros((2*N))

        i = 0
        for n in columns:
            switch = np.zeros((2,3)) # The amount of times a switch occured after punishment, neutral and reward respectively
            no_switch = np.zeros((2,3))
           
            t = 0
            b0 = self.block[0,n]
            while t < self.T - 1:
                while np.abs(self.feedback[t,n]) < 1e-10 and t < self.T - 1:
                    t += 1
                    
                block = self.block[t,n]
                if np.abs(self.new[t,n] - 1) < 1e-10:
                    u = 0
                else: 
                    u = 1
                
                c = self.C[t,n]
                check = 0
                if block > 5:
                    s = t + 1
                    while s < self.T and (self.block[s,n] - block == 0):
                        if np.abs(self.feedback[s,n]) < 1e-10:
                            s+=1
                        else:
                            if self.stims[s,0,n] == c or self.stims[s,1,n] == c:
                                check += 1
                                if self.C[s,n] == c:
                                    no_switch[u,int(self.o[t,n] + 1)] += 1
                                else:
                                    switch[u,int(self.o[t,n] + 1)] += 1
                                break
                            s += 1
                if check > 1:
                    print(f"Warning: more than 1 successor taken for t={t}, n={n}!")
                t += 1
                
            for v in range(2):
                pun_s_rate[2*i + v] = switch[v,0] / (switch[v,0] + no_switch[v,0])
                n_s_rate[2*i + v] = switch[v,1] / (switch[v,1] + no_switch[v,1])
                r_s_rate[2*i + v] = switch[v,2] / (switch[v,2] + no_switch[v,2])

                pun_sd[2*i + v] = np.sqrt( pun_s_rate[2*i + v] * (1 - pun_s_rate[2*i + v]) / (switch[v,0] + no_switch[v,0]))
                n_sd[2*i + v] = np.sqrt( n_s_rate[2*i + v] * (1 - n_s_rate[2*i + v]) / (switch[v,1] + no_switch[v,1]))
                r_sd[2*i + v] = np.sqrt( r_s_rate[2*i + v] * (1 - r_s_rate[2*i + v]) / (switch[v,2] + no_switch[v,2]))
             
            i+=1
        
        subject_names = list(itertools.chain.from_iterable([[s_name + " (new-trial)", s_name + " (not-new-trial)"] for s_name in subject_names]))
        columns = list(itertools.chain.from_iterable([[clm, clm] for clm in columns]))
        df = {'pun': pun_s_rate, 'n': n_s_rate, 'r': r_s_rate, 'subject': subject_names, 'columns': columns,
             'pun_sd':pun_sd, 'n_sd': n_sd, 'r_sd': r_sd}
        
        if per_subject:
            self.display_switch_rates_per_subject(df, num_subjects_per_plot = 2*num_subjects_per_plot, switch_rate_type = 'Stim')
        else:
            self.display_switch_rate_all_subjects(df)
                 
    def plot_paired_switch_rate(self, subject_names = None, name = "Subjects", order_names = False, num_subjects_per_plot = 1, per_subject = True):
        """
        If stimulus i was chosen in trial t versus stimulus j, with outcome o, this function checks whether 
        stimulus i was chosen as well the next time when it got paired with j. For each outcome, this gets averaged over
        all stimuli pairs (i,j) to compute the average switch rate after punishment, neutral and reward.

        Ignores the first 6 blocks and no-feedback trials. 
        """
        
        subject_names, columns = self.subject_names_to_columns(subject_names, order_names)

        N = len(columns)

        # Store the switch rates per subject in here:
        pun_s_rate = np.zeros(N)
        n_s_rate   = np.zeros(N)
        r_s_rate   = np.zeros(N)
        
        # Store the standard deviation for the confidence intervals (normal approximation)):
        pun_sd = np.zeros(N)
        n_sd   = np.zeros(N)
        r_sd   = np.zeros(N)

        i = 0
        for n in columns:
            switch = np.zeros(3) # The amount of times a switch occured after punishment, neutral and reward respectively
            no_switch = np.zeros(3)
            
            t = 0
            while t < self.T[n] - 1:
                while np.abs(self.feedback[t,n]) < 1e-10 and t < self.T[n]- 1:
                    t += 1
                
                block = self.block[t,n]  
                check = 0
                if block > 5:
                    s = t + 1
                    while s < self.T[n]and (self.block[s,n] - block == 0):
                        if np.abs(self.feedback[s,n]) < 1e-10:
                            s+=1
                        else:
                            same_pair = ((self.stims[s,0,n] == self.stims[t,0,n]) and (self.stims[s,1,n] == self.stims[t,1,n])) \
                                    or ((self.stims[s,0,n] == self.stims[t,1,n]) and (self.stims[s,1,n] == self.stims[t,0,n]))

                            if same_pair:
                                check += 1
                                #successor_found = True
                                if self.C[s,n] == self.C[t,n]:
                                    no_switch[int(self.o[t,n] + 1)] += 1
                                else:
                                    switch[int(self.o[t,n] + 1)] += 1
                                break
                                
                            s += 1
                if check > 1:
                    print("Warning: more than 1 successor taken!")
                t += 1

            pun_s_rate[i] = switch[0] / (switch[0] + no_switch[0])
            n_s_rate[i] = switch[1] / (switch[1] + no_switch[1])
            r_s_rate[i] = switch[2] / (switch[2] + no_switch[2]) 
            
            pun_sd[i] = np.sqrt( pun_s_rate[i] * (1 - pun_s_rate[i]) / (switch[0] + no_switch[0]))
            n_sd[i] = np.sqrt( n_s_rate[i] * (1 - n_s_rate[i]) / (switch[1] + no_switch[1]))
            r_sd[i] = np.sqrt( r_s_rate[i] * (1 - r_s_rate[i]) / (switch[2] + no_switch[2]))

            i+=1

        df = {'pun': pun_s_rate, 'n': n_s_rate, 'r': r_s_rate, 'subject': subject_names, 'columns': columns,
             'pun_sd':pun_sd, 'n_sd':n_sd, 'r_sd':r_sd}
        
        if per_subject:
            self.display_switch_rates_per_subject(df, num_subjects_per_plot = num_subjects_per_plot)
        else:
            self.display_switch_rate_all_subjects(df)

        return df
        
        
    def plot_paired_switch_rate_with_threshold(self, subject_names=None, name="Subjects", order_names=False, num_subjects_per_plot=1, per_subject=True, threshold=5):
        """
        If stimulus i was chosen in trial t versus stimulus j, with outcome o, this function checks whether 
        stimulus i was chosen as well the next time when it got paired with j. For each outcome, this gets averaged over
        all stimuli pairs (i,j) to compute the average switch rate after punishment, neutral and reward.

        Ignores the first 6 blocks and no-feedback trials. 
        """

        subject_names, columns = self.subject_names_to_columns(subject_names, order_names)

        N = len(columns)

        # Store the switch rates per subject in here:
        pun_s_rate = np.zeros(N)
        n_s_rate = np.zeros(N)
        r_s_rate = np.zeros(N)

        # Store the standard deviation for the confidence intervals (normal approximation)):
        pun_sd = np.zeros(N)
        n_sd = np.zeros(N)
        r_sd = np.zeros(N)

        i = 0
        for n in columns:
            switch = np.zeros(3)  # The amount of times a switch occurred after punishment, neutral, and reward respectively
            no_switch = np.zeros(3)

            pair_counter = {}  # Dictionary to keep track of the count for each pair

            t = 0
            while t < self.T[n] - 1:
                while np.abs(self.feedback[t, n]) < 1e-10 and t < self.T[n] - 1:
                    t += 1

                block = self.block[t, n]
                check = 0
                if block > 5:
                    s = t + 1
                    while s < self.T[n] and (self.block[s, n] - block == 0):
                        if np.abs(self.feedback[s, n]) < 1e-10:
                            s += 1
                        else:
                            same_pair = ((self.stims[s, 0, n] == self.stims[t, 0, n]) and (
                                    self.stims[s, 1, n] == self.stims[t, 1, n])) \
                                        or ((self.stims[s, 0, n] == self.stims[t, 1, n]) and (
                                                self.stims[s, 1, n] == self.stims[t, 0, n]))

                            if same_pair:
                                check += 1
                                pair = tuple(sorted((self.stims[t, 0, n], self.stims[t, 1, n])))
                                pair_counter[pair] = pair_counter.get(pair, 0) + 1

                                # Only consider the first X occurrences of the same pair
                                if pair_counter[pair] <= threshold:
                                    if self.C[s, n] == self.C[t, n]:
                                        no_switch[int(self.o[t, n] + 1)] += 1
                                    else:
                                        switch[int(self.o[t, n] + 1)] += 1
                                break

                        s += 1
                if check > 1:
                    print("Warning: more than 1 successor taken!")
                t += 1

            pun_s_rate[i] = switch[0] / (switch[0] + no_switch[0])
            n_s_rate[i] = switch[1] / (switch[1] + no_switch[1])
            r_s_rate[i] = switch[2] / (switch[2] + no_switch[2])

            pun_sd[i] = np.sqrt(pun_s_rate[i] * (1 - pun_s_rate[i]) / (switch[0] + no_switch[0]))
            n_sd[i] = np.sqrt(n_s_rate[i] * (1 - n_s_rate[i]) / (switch[1] + no_switch[1]))
            r_sd[i] = np.sqrt(r_s_rate[i] * (1 - r_s_rate[i]) / (switch[2] + no_switch[2]))

            i += 1

        df = {'pun': pun_s_rate, 'n': n_s_rate, 'r': r_s_rate, 'subject': subject_names, 'columns': columns,
              'pun_sd': pun_sd, 'n_sd': n_sd, 'r_sd': r_sd}

        if per_subject:
            self.display_switch_rates_per_subject(df, num_subjects_per_plot=num_subjects_per_plot)
        else:
            self.display_switch_rate_all_subjects(df)

        return df

    
    
    def plot_paired_switch_rate_block_stratified(self, subject_names = None, name = 'Subjects', order_names = False, num_subjects_per_plot = 1, per_subject = True):
        """
        If stimulus i was chosen in trial t with outcome o, this function checks whether 
        stimulus i was chosen as well the next time. For each outcome, this gets averaged over
        all stimuli to compute the average switch rate after punishment, neutral and reward.
        
        Ignores the first 6 blocks and no-feedback trials. 
        
        Stratifies over blocks with at least two new stimuli and less.
        """

        subject_names, columns = self.subject_names_to_columns(subject_names, order_names)

        N = len(columns)

        # Store the switch rates per subject in here:
        pun_s_rate = np.zeros((2*N))
        n_s_rate   = np.zeros((2*N))
        r_s_rate   = np.zeros((2*N))
        
        # Store the standard deviation for the confidence intervals (normal approximation)):
        pun_sd = np.zeros((2*N))
        n_sd   = np.zeros((2*N))
        r_sd   = np.zeros((2*N))

        i = 0
        for n in columns:
            switch = np.zeros((2,3)) # The amount of times a switch occured after punishment, neutral and reward respectively
            no_switch = np.zeros((2,3))
           
            t = 0
            b0 = self.block[0,n]
            while t < self.T[n]- 1:
                while np.abs(self.feedback[t,n]) < 1e-10 and t < self.T[n]- 1:
                    t += 1
                    
                block = self.block[t,n]
                if len(self.new_stimuli[n][block - b0]) > 1:
                    u = 0
                else: 
                    u = 1
                
                c = self.C[t,n]
                check = 0
                if block > 5:
                    s = t + 1
                    while s < self.T[n]and (self.block[s,n] - block == 0):
                        if np.abs(self.feedback[s,n]) < 1e-10:
                            s+=1
                        else:
                            same_pair = ((self.stims[s,0,n] == self.stims[t,0,n]) and (self.stims[s,1,n] == self.stims[t,1,n])) \
                                    or ((self.stims[s,0,n] == self.stims[t,1,n]) and (self.stims[s,1,n] == self.stims[t,0,n]))

                            if same_pair:
                                check += 1
                                if self.C[s,n] == self.C[t,n]:
                                    no_switch[u,int(self.o[t,n] + 1)] += 1
                                else:
                                    switch[u,int(self.o[t,n] + 1)] += 1
                                break
                                
                            s += 1
                if check > 1:
                    print(f"Warning: more than 1 successor taken for t={t}, n={n}!")
                t += 1
                
            for v in range(2):
                pun_s_rate[2*i + v] = switch[v,0] / (switch[v,0] + no_switch[v,0])
                n_s_rate[2*i + v] = switch[v,1] / (switch[v,1] + no_switch[v,1])
                r_s_rate[2*i + v] = switch[v,2] / (switch[v,2] + no_switch[v,2])

                pun_sd[2*i + v] = np.sqrt( pun_s_rate[2*i + v] * (1 - pun_s_rate[2*i + v]) / (switch[v,0] + no_switch[v,0]))
                n_sd[2*i + v] = np.sqrt( n_s_rate[2*i + v] * (1 - n_s_rate[2*i + v]) / (switch[v,1] + no_switch[v,1]))
                r_sd[2*i + v] = np.sqrt( r_s_rate[2*i + v] * (1 - r_s_rate[2*i + v]) / (switch[v,2] + no_switch[v,2]))
            
            i+=1
        
        subject_names = list(itertools.chain.from_iterable([[s_name + " (>=2 new)", s_name + " (<=1 new)"] for s_name in subject_names]))
        columns = list(itertools.chain.from_iterable([[clm, clm] for clm in columns]))
        df = {'pun': pun_s_rate, 'n': n_s_rate, 'r': r_s_rate, 'subject': subject_names, 'columns': columns,
             'pun_sd':pun_sd, 'n_sd': n_sd, 'r_sd': r_sd}
        
        if per_subject:
            self.display_switch_rates_per_subject(df, num_subjects_per_plot = 2*num_subjects_per_plot)
        else:
            self.display_switch_rate_all_subjects(df)

    def plot_paired_switch_rate_trials_stratified(self, subject_names = None, name = 'Subjects', order_names = False, num_subjects_per_plot = 1, per_subject = True):
        """
        If stimulus i was chosen in trial t with outcome o, this function checks whether 
        stimulus i was chosen as well the next time. For each outcome, this gets averaged over
        all stimuli to compute the average switch rate after punishment, neutral and reward.
        
        Ignores the first 6 blocks and no-feedback trials. 
        
        Stratifies over trials where there are two new stimuli versus less than two.
        """

        subject_names, columns = self.subject_names_to_columns(subject_names, order_names)

        N = len(columns)

        # Store the switch rates per subject in here:
        pun_s_rate = np.zeros((2*N))
        n_s_rate   = np.zeros((2*N))
        r_s_rate   = np.zeros((2*N))
        
        # Store the standard deviation for the confidence intervals (normal approximation)):
        pun_sd = np.zeros((2*N))
        n_sd   = np.zeros((2*N))
        r_sd   = np.zeros((2*N))

        i = 0
        for n in columns:
            switch = np.zeros((2,3)) # The amount of times a switch occured after punishment, neutral and reward respectively
            no_switch = np.zeros((2,3))
           
            t = 0
            b0 = self.block[0,n]
            while t < self.T[n]- 1:
                while np.abs(self.feedback[t,n]) < 1e-10 and t < self.T[n]- 1:
                    t += 1
                    
                block = self.block[t,n]
                if np.abs(self.new[t,n] - 1) < 1e-10:
                    u = 0
                else: 
                    u = 1
                
                c = self.C[t,n]
                check = 0
                if block > 5:
                    s = t + 1
                    while s < self.T[n]and (self.block[s,n] - block == 0):
                        if np.abs(self.feedback[s,n]) < 1e-10:
                            s+=1
                        else:
                            same_pair = ((self.stims[s,0,n] == self.stims[t,0,n]) and (self.stims[s,1,n] == self.stims[t,1,n])) \
                                    or ((self.stims[s,0,n] == self.stims[t,1,n]) and (self.stims[s,1,n] == self.stims[t,0,n]))

                            if same_pair:
                                check += 1
                                if self.C[s,n] == self.C[t,n]:
                                    no_switch[u,int(self.o[t,n] + 1)] += 1
                                else:
                                    switch[u,int(self.o[t,n] + 1)] += 1
                                break
                                
                            s += 1
                if check > 1:
                    print(f"Warning: more than 1 successor taken for t={t}, n={n}!")
                t += 1
                
            for v in range(2):
                pun_s_rate[2*i + v] = switch[v,0] / (switch[v,0] + no_switch[v,0])
                n_s_rate[2*i + v] = switch[v,1] / (switch[v,1] + no_switch[v,1])
                r_s_rate[2*i + v] = switch[v,2] / (switch[v,2] + no_switch[v,2])

                pun_sd[2*i + v] = np.sqrt( pun_s_rate[2*i + v] * (1 - pun_s_rate[2*i + v]) / (switch[v,0] + no_switch[v,0]))
                n_sd[2*i + v] = np.sqrt( n_s_rate[2*i + v] * (1 - n_s_rate[2*i + v]) / (switch[v,1] + no_switch[v,1]))
                r_sd[2*i + v] = np.sqrt( r_s_rate[2*i + v] * (1 - r_s_rate[2*i + v]) / (switch[v,2] + no_switch[v,2]))
            
            i+=1
        
        subject_names = list(itertools.chain.from_iterable([[s_name + " (new-trial)", s_name + " (no-new-trial)"] for s_name in subject_names]))
        columns = list(itertools.chain.from_iterable([[clm, clm] for clm in columns]))
        df = {'pun': pun_s_rate, 'n': n_s_rate, 'r': r_s_rate, 'subject': subject_names, 'columns': columns,
             'pun_sd':pun_sd, 'n_sd': n_sd, 'r_sd': r_sd}
        
        if per_subject:
            self.display_switch_rates_per_subject(df, num_subjects_per_plot = 2*num_subjects_per_plot)
        else:
            self.display_switch_rate_all_subjects(df)
            
    def plot_paired_switch_rate_var_stratified(self, var_stratification = 'valence_last', subject_names = None, name = 'Subjects', order_names = False, num_subjects_per_plot = 1, per_subject = True, save_folder = None):
        """
        If stimulus i was chosen in trial t with outcome o, this function checks whether 
        stimulus i was chosen as well the next time in the same pair. For each outcome, this gets averaged over
        all stimuli to compute the average switch rate after punishment, neutral and reward.
        
        Ignores the first 6 blocks and no-feedback trials. 
        
        Stratifies over blocks where the 'var_stratification' (string) is higher or lower than the median.
        """

        subject_names, columns = self.subject_names_to_columns(subject_names, order_names)

        N = len(columns)

        # Store the switch rates per subject in here:
        pun_s_rate = np.zeros((2*N))
        n_s_rate   = np.zeros((2*N))
        r_s_rate   = np.zeros((2*N))
        
        # Store the standard deviation for the confidence intervals (normal approximation)):
        pun_sd = np.zeros((2*N))
        n_sd   = np.zeros((2*N))
        r_sd   = np.zeros((2*N))

        i = 0
        for n in columns:
            # The stratification criterion:
            var_str = self.create_subject_df(n=n)[var_stratification].values
            var_high = (var_str > np.median(var_str)).astype('int64')
            
            switch = np.zeros((2,3)) # The amount of times a switch occured after punishment, neutral and reward respectively
            no_switch = np.zeros((2,3))
           
            t = 0
            b0 = self.block[0,n]
            while t < self.T[n]- 1:
                while np.abs(self.feedback[t,n]) < 1e-10 and t < self.T[n]- 1:
                    t += 1
                    
                block = self.block[t,n]
                u = var_high[t]
                
                c = self.C[t,n]
                check = 0
                if block > 5:
                    s = t + 1
                    while s < self.T[n]and (self.block[s,n] - block == 0):
                        if np.abs(self.feedback[s,n]) < 1e-10:
                            s+=1
                        else:
                            same_pair = ((self.stims[s,0,n] == self.stims[t,0,n]) and (self.stims[s,1,n] == self.stims[t,1,n])) \
                                    or ((self.stims[s,0,n] == self.stims[t,1,n]) and (self.stims[s,1,n] == self.stims[t,0,n]))

                            if same_pair:
                                check += 1
                                if self.C[s,n] == self.C[t,n]:
                                    no_switch[u,int(self.o[t,n] + 1)] += 1
                                else:
                                    switch[u,int(self.o[t,n] + 1)] += 1
                                break
                                
                            s += 1
                if check > 1:
                    print(f"Warning: more than 1 successor taken for t={t}, n={n}!")
                t += 1
                
            for v in range(2):
                pun_s_rate[2*i + v] = switch[v,0] / (switch[v,0] + no_switch[v,0])
                n_s_rate[2*i + v] = switch[v,1] / (switch[v,1] + no_switch[v,1])
                r_s_rate[2*i + v] = switch[v,2] / (switch[v,2] + no_switch[v,2])

                pun_sd[2*i + v] = np.sqrt( pun_s_rate[2*i + v] * (1 - pun_s_rate[2*i + v]) / (switch[v,0] + no_switch[v,0]))
                n_sd[2*i + v] = np.sqrt( n_s_rate[2*i + v] * (1 - n_s_rate[2*i + v]) / (switch[v,1] + no_switch[v,1]))
                r_sd[2*i + v] = np.sqrt( r_s_rate[2*i + v] * (1 - r_s_rate[2*i + v]) / (switch[v,2] + no_switch[v,2]))
            
            i+=1
        
        subject_names = list(itertools.chain.from_iterable([[s_name + f" (low {var_stratification})", s_name + f" (high {var_stratification})"] for s_name in subject_names]))
        columns = list(itertools.chain.from_iterable([[clm, clm] for clm in columns]))
        df = {'pun': pun_s_rate, 'n': n_s_rate, 'r': r_s_rate, 'subject': subject_names, 'columns': columns,
             'pun_sd':pun_sd, 'n_sd': n_sd, 'r_sd': r_sd}
        
        if per_subject:
            self.display_switch_rates_per_subject(df, num_subjects_per_plot = 2*num_subjects_per_plot, save_folder = save_folder, spec = var_stratification)
        else:
            self.display_switch_rate_all_subjects(df, save_folder = save_folder, spec = var_stratification)

    
    def display_switch_rates_per_subject(self, df, num_subjects_per_plot = 1, switch_rate_type = 'Paired', save_folder = None, spec = ""):
        """
        Given the dict df = {'pun': pun_s_rate, 'n': n_s_rate, 'r':r_s_rate, 'subject': subject_names, 'columns': columns, 'pun_sd':pun_sd, 'n_sd': n_sd, 'r_sd': r_sd},
        displays the switch rates in a bar plot per subject, or per two subjects if per_true = True.
        """
        
        remainder = len(df['subject']) % num_subjects_per_plot
        k_lim = len(df['subject']) - remainder
        
        k = 0
        while k < k_lim:
            fig = go.Figure()
            for l in range(num_subjects_per_plot):
                print(df['subject'][k + l])
                fig.add_trace(go.Bar(name = df['subject'][k + l], x = ['pun', 'n', 'r'],
                                    y = [df['pun'][k+l], df['n'][k+l], df['r'][k+l]], 
                                    error_y =dict(type='data', array= 1.96*np.array([df['pun_sd'][k+l], df['n_sd'][k+l], df['r_sd'][k+l]])), 
                                    marker_color = self.colors[df['columns'][k+l]]))
                title = f"{switch_rate_type} switch rate" 
                if num_subjects_per_plot == 1:
                    title = f"{df['subject'][k+l]}) {switch_rate_type} switch rate" 
                fig.update_layout(barmode='group', title = title,
                                 yaxis={
                                    'title':'Switch Rate',
                                    'range': [0., .6]  
                                })
            fig.show()
            if save_folder is not None:
                save_path = os.path.join(save_folder, f'switch_rate_{spec}_{k}.jpeg')
                fig.write_image(save_path)
            
            k += num_subjects_per_plot
            
        if remainder > 0:
            fig = go.Figure()
            for l in range(remainder):
                fig.add_trace(go.Bar(name = df['subject'][k + l], x = ['pun', 'n', 'r'],
                                        y = [df['pun'][k+l], df['n'][k+l], df['r'][k+l]], 
                                        error_y =dict(type='data', array= 1.96*np.array([df['pun_sd'][k+l], df['n_sd'][k+l], df['r_sd'][k+l]])), 
                                        marker_color = self.colors[df['columns'][k+l]]))
            title = f"{switch_rate_type} switch rate"
            if num_subjects_per_plot == 1:
                title = f"{df['subject'][k+l]}) {switch_rate_type} switch rate" 
            fig.update_layout(barmode='group', title = title,
                                 yaxis={
                                    'title':'Switch Rate',
                                    'range': [0., .6]  
                                })
            fig.show()
            if save_folder is not None:
                save_path = os.path.join(save_folder, f'switch_rate_{spec}_{k_lim+1}.jpeg')
                fig.write_image(save_path)
            
            
                
    def display_switch_rate_all_subjects(self, df, name_spec = "", save_folder = None, spec = ""):
        """
        Given the dict df = {'pun': pun_s_rate, 'n': n_s_rate, 'r':r_s_rate, 'subject': subject_names, 'columns': columns, 'pun_sd':pun_sd, 'n_sd': n_sd, 'r_sd': r_sd},
        displays the switch rates in a bar plot with all subjects per 'pun', 'n', 'r'.
        """
        pun_sd = df['pun_sd']
        n_sd   = df['n_sd']
        r_sd   = df['r_sd']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df['subject'], 
            y=df['pun'], 
            error_y = dict(type='data', array=1.96 * pun_sd),
            marker_color = [self.colors[n] for n in df['columns']]))
        fig.update_layout(
            title = 'Punishment' + name_spec, 
            yaxis={
                'title':'Switch Rate',
                'range': [0., .6]  
            })
        fig.show() 
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df['subject'], 
            y=df['n'], 
            error_y = dict(type='data', array=1.96 * n_sd),
            marker_color = [self.colors[n] for n in df['columns']]))
        fig.update_layout(
            title = 'Neutral' + name_spec, 
            yaxis={
                'title':'Switch Rate',
                'range': [0., .6]  
            })
        fig.show() 
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df['subject'], 
            y=df['r'], 
            error_y = dict(type='data', array=1.96 * r_sd),
            marker_color = [self.colors[n] for n in df['columns']]))
        fig.update_layout(
            title = 'Reward' + name_spec, 
            yaxis={
                'title':'Switch Rate',
                'range': [0., .6]  
            })
        fig.show() 






class subjects_equalT:
    """
    Abstract class that stores all the data of the subjects, and allows to either simulate data or store real subject data:
    
    Requires all the state trajectories to be of equal length!
    
    Data:
    -----
    
    - T : list of ints, the time horizons of the decision making problem
    - N : int, the number of subjects.
    - Nc: number of choices, taken as the highest stimulus ID + 1.
    
    - State space information:
       1) stims:  (T, 2, N) int-numpy array, with the stim IDs at time t for subject n in [t,:,n], ordered (stim1, stim2)
       2) 'p_R'  , (T, 2, N) numpy array with the reward probabilities of the stims at time t for subject n in [t,:,n], ordered (stim1, stim2)
       3) 'p_Pun', (T, 2, N) numpy array with the punishment probabilities of the stims at time t for subject n in [t,:,n], ordered (stim1, stim2)
       4) 'feedback', (T, N) int-numpy array, indicating whether the trial is a feedback trial (= 1), or not (= 0).
       5) 'block',  (T,N) int-numpy array, the block indicator of every trial
    
    - C    : (T, N) int-numpy array, ID of the chosen stimulus.
    - C_st : (T, N) int-numpy array 0 if stim1 was chosen, 1 if stim2 was chosen.
    - o    : (T, N) numpy array, outcome of the trials.
    - feedback_time: (T, N) numpy array, time of feedback, ms since Unic Epoch.
    
    - subject_names : list of strings, names for the subjects.
    - subject_colors: list of strings, color names for all subjects. 
    
    Computes the following state info:
    - EV: (T, 2, N) numpy array, the expected value of stimulus 1 and 2 in EV[:,0,:] and EV[:,1,:] respectively
    - best_stim: (T) pandas series, the indicator of the best stimulus, 0 for stim1, 1 for stim2, nan if equal EV
    - new_stimuli, a list of lists, with self.new_stimuli[n][b] the new stimuli
        in block b for the n-th subject.
    - new, a (T, N) numpy array, with a 1 if the trial has 2 stimuli that are new to this block,
        and 0 otherwise.
    - num_new, a (num_block, N) int-numpy array, indicating the number of new stimuli per block.
    - stim_types: (T, 2, N) int-numpy array, the coin types of stim1 and stim2. 0 for A, 1 for B, 2 for C, 3 for D.
    - acc: (T, N) numpy array, 1. if the best stimulus was chosen, according to the underlying probabilities. 0. if not. np.nan if equal expected value.
    - time_since_last_rep: (T, 2, N) numpy array with the time since the last presentation of the stimulus in units of days (not rounded).
    - last_feedback_time
    - last_learning_block
    - valence_last: (T,N), the reported valence of the last mood report. Divided by 500. (range 0 to 1)
    - el_sad_last: (T,N), the reported (Elated - Sad) of the last mood report. Divided by 100. (range -1 to 1)
    
    
    To simulate data:
    - optional, Q_style: string, style of learning Q-values for the simulated subjects.
    - optional, choice_style: string, style of choice function for the simulated subjects. Currently only value-softmax.
    - optional, R_style: string, style of perceived reward. 'objective' or 'subjective'.
    - optional, P_true : (N, .) with parameter samples to generate data, with the parameter names as column names
    
    Functions:
    ----------
    - simulate_data: Function that given state trajectories, a Q_style and a choice_style, with parameters P_true,
                     simulates choice and outcome data.
    - gen_outcome  : Helper function to generate outcomes given choices and Q-values.
    - plot_avg_acc : Displays the average accuracy along all subjects, per time step t.
    
    The following code is implemented for speed, using numpy and row-major ordering for faster vectorized computations. 
    More speed gains could be made by profiling code.
    
    Would've been easier to first do everything in dfs, and then to transfer to arrays with a function call, for RL_models.
    """
    
                 
                      
        
    
    
    
    def feedback_stims_per_block(self, subject_name = "204"):
        """
        Prints the 3 feedback stims per block.
        """
        
        _, columns = self.subject_names_to_columns([subject_name], False)
        n = columns[0]
        
        t = 0
        b = 0
        while t < self.T:
            print(f"{b:3d}) {np.unique(self.stims[t:t+2, :, n])}")
            b+=1
            t+=72

    def compute_nf_acc_time_stratified(self, num_groups = 4):
        """
        Not ready yet.
        """

        for n in range(self.N):
            df = self.create_subject_df(n = n)
            df = df[df['feedback'] == 0]

            # Last feedback times for the stimuli
            df['time_since_last_feedback1'] = df['feedback_time'].values - self.last_feedback_time.loc[df['stim1'].values, self.subject_names[n]].values
            df['time_since_last_feedback2'] = df['feedback_time'].values - self.last_feedback_time.loc[df['stim2'].values, self.subject_names[n]].values

            df['min_time_since_last_feedback'] = np.minimum(df['time_since_last_feedback1'].values,df['time_since_last_feedback2'].values) / (24. * 3600. * 1000.)
            df['max_time_since_last_feedback'] = np.maximum(df['time_since_last_feedback1'].values,df['time_since_last_feedback2'].values) / (24. * 3600. * 1000.)

            df['low_stim'] = np.minimum(df['stim1'].values, df['stim2'].values)
            df['high_stim'] = np.maximum(df['stim1'].values, df['stim2'].values)


            df_stims = df.groupby(pd.cut(df['min_time_since_last_feedback'], [0., 2.5 , 4.5, 8.5, 14.5, 21.5, 28.5]))[['stim1', 'stim2','acc']]

            df_acc = df.groupby(pd.cut(df['min_time_since_last_feedback'], [0., 2.5 , 4.5, 8.5, 14.5, 21.5, 28.5]))
            df_gpd_acc = df_acc.apply(lambda x: x.groupby(['low_stim', 'high_stim'])['acc'].mean()).groupby('min_time_since_last_feedback')#.droplevel(['low_stim', 'high_stim'])

            total = df_acc['acc'].apply(lambda x: x.shape[0])
            acc_agr = df_acc['acc'].apply(lambda x: x.sum())
            total_unequal = df_acc['acc'].apply(lambda x: x.shape[0] - x.isna().sum())

            df_agr = pd.concat([total, acc_agr, total_unequal], axis = 1)
            df_agr.columns = ['total', 'acc_agr', 'total_unequal']

            total_gpd = df_gpd_acc.apply(lambda x: x.shape[0])
            acc_agr_gpd = df_gpd_acc.apply(lambda x: x.sum())
            total_unequal_gpd = df_gpd_acc.apply(lambda x: x.shape[0] - x.isna().sum())

            df_gpd = pd.concat([total_gpd, acc_agr_gpd, total_unequal_gpd], axis = 1)
            df_gpd.columns = ['total', 'acc_agr', 'total_unequal']

    #         k = 0
    #         for name, gp in df_stims:
    #             if k == 2:
    #                 with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #                     print(gp)
    #             k+=1

            display(df_agr)
            display(df_gpd)

    def compute_nf_acc_block_stratified(self, cuts = [0., 2.5 , 4.5, 6.5, 8.5, 10.5, 14.5, 20.5, 45.5, 61.5]):
        """
        Displays a few statistics relating to the accuracy of non-feedback trials over time:
        - Possibility to display a table with the number of total/unequal stim pairs stratified over the amount of blocks since 
          learning, together with the accuracy.
        - There are a lot of repetitions, so the same is done for unique pairs in each time group.
        Takes the minimum number of blocks since the last learning block of the two stimuli.
        - Displays the difference in time since last learning block: chosen - unchosen.

        TODO: better to look at repetitions.

        Arguments:
        - cuts: list, specified the cutting points for number of blocks since learning.
        """

        df_hists = []

        fig_uniqueness_rate = go.Figure()
        fig_avg_unique_acc = go.Figure()
        fig_unique_n       = go.Figure()

        for n in range(self.N):
            df = self.create_subject_df(n = n)

            df = df[df['feedback'] == 0]

            # Last feedback times for the stimuli
            df['time_since_last_lblock1'] = df['block'].values - self.last_learning_block.loc[df['stim1'].values, self.subject_names[n]].values
            df['time_since_last_lblock2'] = df['block'].values - self.last_learning_block.loc[df['stim2'].values, self.subject_names[n]].values

            df['min_blocks_since_last_lblock'] = np.minimum(df['time_since_last_lblock1'].values,df['time_since_last_lblock2'].values) 
            df['max_blocks_since_last_lblock'] = np.maximum(df['time_since_last_lblock1'].values,df['time_since_last_lblock2'].values) 

            df['low_stim'] = np.minimum(df['stim1'].values, df['stim2'].values)
            df['high_stim'] = np.maximum(df['stim1'].values, df['stim2'].values)  


            # Compute the number of unique pairs and their average accuracy per time group:
            df_acc = df.groupby(pd.cut(df['min_blocks_since_last_lblock'], cuts))
            df_gpd_acc = df_acc.apply(lambda x: x.groupby(['low_stim', 'high_stim'])['acc'].mean()).groupby('min_blocks_since_last_lblock')#.droplevel(['low_stim', 'high_stim'])

            total = df_acc['acc'].apply(lambda x: x.shape[0])
            acc_agr = df_acc['acc'].apply(lambda x: x.sum())
            total_unequal = df_acc['acc'].apply(lambda x: x.shape[0] - x.isna().sum())

            df_agr = pd.concat([total, acc_agr, total_unequal], axis = 1)
            df_agr.columns = ['total', 'acc_agr', 'total_unequal']

            total_gpd = df_gpd_acc.apply(lambda x: x.shape[0])
            acc_agr_gpd = df_gpd_acc.apply(lambda x: x.sum())
            total_unequal_gpd = df_gpd_acc.apply(lambda x: x.shape[0] - x.isna().sum())

            df_gpd = pd.concat([total_gpd, acc_agr_gpd, total_unequal_gpd], axis = 1)
            df_gpd.columns = ['total', 'acc_agr', 'total_unequal']

            if n == 0:
                display(df_agr)
                display(df_gpd)

            fig_uniqueness_rate.add_trace(go.Bar(name = self.subject_names[n],
                                                 x = [str(l) for l in df_agr.index],
                                                 y = df_gpd['total'] / df_agr['total'],
                                                marker_color = self.colors[n]))

            p = df_gpd['acc_agr'] / df_gpd['total_unequal']

            #p_std = 1.96 * np.sqrt(p * (1. - p) / df_gpd['total_unequal'])

            fig_avg_unique_acc.add_trace(go.Bar(name = self.subject_names[n],
                                               x = [str(l) for l in df_agr.index],
                                               y = p,
                                               #error_y = dict(type = 'data', array = p_std),
                                               marker_color = self.colors[n]))

            fig_unique_n.add_trace(go.Bar(name = self.subject_names[n],
                                         x = [str(l) for l in df_agr.index],
                                       y = df_gpd['total_unequal'],
                                       marker_color = self.colors[n]))


            # Compute the distribution of choice time difference of the chosen stimulus:
            #fig_choice_time_diff = go.Figure()
            df['stim_time_diff'] = df['time_since_last_lblock2'] - df['time_since_last_lblock1']
            df['choice_time_diff'] = (df['C_st'] * 2. - 1.) * df['stim_time_diff']

            #df_unique = df.groupby(['low_stim', 'high_stim'])['stim_time_diff'].first()#.apply(lambda x: x['stim_time_diff'].first()) #, 'choice_time_diff'
            #display(df_unique)

            bins  = np.concatenate((np.arange(-40.5, -4.5, 2), (-2.5, 2.5), np.arange(4.5,40.5,2.)))
            counts, bins = np.histogram(df.choice_time_diff, bins=bins)
            bins = 0.5 * (bins[:-1] + bins[1:])

            fig = go.Figure(go.Bar(x=bins, y=counts, marker_color = self.colors[n]))#, color = self.colors[n])
            fig.update_layout(title = f"{self.subject_names[n]}) Time since block(choice) - Time since block(not-choice)",
                             xaxis_title = "block_diff",
                              yaxis_title = "count")
            fig.show()

            #fig_stim_time_diff = go.Figure(x = 'choice_time_diff', bins = bins)
            #fig_stim_time_diff.show()

            print(f"{self.subject_names[n]}) (#block_diff > 2)/(#block_diff < -2): {np.sum(df['choice_time_diff'] > 2) / np.sum(df['choice_time_diff'] < -2)}")


            # Compute the number of repetitions of pairs:
            df_hists.append(df.groupby(['low_stim', 'high_stim'])['block'].count())#.hist(bins = np.arange(12) - .5))

            # Compute the aggregate type pair distribution - only displayed for n == 0 now:        
    #         df['low_type'] = np.minimum(df['type1'].values, df['type2'].values)
    #         df['high_type'] = np.maximum(df['type1'].values, df['type2'].values)
    #         df_type_pairs = df.groupby(['low_type', 'high_type'])['block'].count()     
    #         df_unordered_pairs = df.groupby(['type1', 'type2'])['feedback_time'].count()

    #         if n == 0:
    #             display(df_type_pairs)
    #             display(df_unordered_pairs)

        # Show figures:
        fig_uniqueness_rate.update_layout(
                title = "Fraction of unique non-feedback stimulus pairs in each group",
                xaxis_title = '#Blocks since last learning block, minimum of both stimuli',
                yaxis_title = 'Fraction Unique',
                yaxis={
                    'range': [0., 1.]  
                })
        fig_uniqueness_rate.show()

        fig_avg_unique_acc.update_layout(title = "Average of Unique Non-Feedback Stimuli Pair Accuracy",
                                        xaxis_title = '#Blocks since last learning block, minimum of both stimuli',
                                        yaxis_title = "Accuracy",
                                        yaxis={
                                            'range': [0.,1.]
                                        })
        fig_avg_unique_acc.show()

        fig_unique_n.update_layout(title = "Number of Unique Unequal Expected Value Pairs per group",
                                   xaxis_title = '#Blocks since last learning block, minimum of both stimuli',
                                    yaxis_title = "Number")
        fig_unique_n.show()

        df_hists = pd.concat(df_hists, axis = 1)
        df_hists.columns = self.subject_names

        for clm in df_hists.columns:
            df_hists.hist(column = clm, bins = np.arange(12) + .5)  
        return df_hists
        
    

    
    def plot_nf_switch_rate(self, save_folder = None):
        """
        For each subject, computes the within-block pair switch rate and accuracy improvement rate,
        as a way to test whether people use indirect feedback every 10 trials to switch. This statistic may need
        to be upgraded to only account for switches after indirect feedback.
        """
        switch_rate = []
        improvement_rate_within_switch = []
        
        fig = go.Figure()
        fig_num_switches = go.Figure()
        fig_acc_improve = go.Figure()
        fig_crd = go.Figure() # Choice Rate Diff
        fig_cr = go.Figure()
        
        for n in range(self.N):
            df = self.create_subject_df(n=n)
            df = df[df['feedback'] < .5]
            
            df['low_stim'] = np.minimum(df['stim1'], df['stim2'])
            df['high_stim'] = np.maximum(df['stim1'], df['stim2'])
            df = df[~df['acc'].isna()]
                        
            gps = df.groupby(['block', 'low_stim', 'high_stim'])
            
            num_potential_switches = gps['acc'].count() - 1
            
            #print(num_potential_switches[num_potential_switches > 0.5])
            
#             def count_better_choice(x):
#                 num = 0
#                 for k in range(x.shape[0] - 1):
#                     if x['C'].iloc[k+1] != x['C'].iloc[k]:
#                         if x['acc'].iloc[k+1] > .5 and x['acc'].iloc[k] < .5:
#                             num += 1
#                 return num
            
#             df_better_choice = gps.apply(count_better_choice)
            
#             #print(df_better_choice[num_potential_switches > 0.5])
            
#             def count_worse_choice(x):
#                 num = 0
#                 for k in range(x.shape[0] - 1):
#                     if x['C'].iloc[k+1] != x['C'].iloc[k]:
#                         if x['acc'].iloc[k+1] < .5 and x['acc'].iloc[k] > .5:
#                             num += 1
#                 return num
            
#             df_worse_choice = gps.apply(count_worse_choice)
            
            
            ## Count how many switches there are after different outcomes:
            def count_switch_after_pun(x):
                num = 0
                for k in range(x.shape[0] - 1):
                    if x['C'].iloc[k+1] != x['C'].iloc[k]:
                        if x['o'].iloc[k] < -.5:
                            num += 1
                return num
            
            df_switch_after_pun = gps.apply(count_switch_after_pun)
            
            def count_switch_after_n(x):
                num = 0
                for k in range(x.shape[0] - 1):
                    if x['C'].iloc[k+1] != x['C'].iloc[k]:
                        if np.abs(x['o'].iloc[k]) < .5:
                            num += 1
                return num
            
            df_switch_after_n = gps.apply(count_switch_after_n)
            
            def count_switch_after_r(x):
                num = 0
                for k in range(x.shape[0] - 1):
                    if x['C'].iloc[k+1] != x['C'].iloc[k]:
                        if x['o'].iloc[k] > .5:
                            num += 1
                return num
            
            df_switch_after_r = gps.apply(count_switch_after_r)
            
            total_r = gps['o'].apply(lambda x: np.sum(x[:-1] > .5))
            total_n = gps['o'].apply(lambda x: np.sum(np.abs(x[:-1]) < .5))
            total_pun = gps['o'].apply(lambda x: np.sum(x[:-1] < -.5))
            
#             print(df_switch_after_r[df_switch_after_r > 0])
#             print(total_r[df_switch_after_r > 0])
            
            df_total_switches = df_switch_after_pun + df_switch_after_n + df_switch_after_r
            
#             #idxs = df_total_switches > .5
            
# #             print(df_total_switches)
# #             print(df_switch_after_pun[idxs])
# #             print(df_switch_after_n[idxs])
# #             print(df_switch_after_r[idxs])
            
            switch_rates = np.array([df_switch_after_pun.sum() / total_r.sum(),
                                    df_switch_after_n.sum() / total_n.sum(),
                                    df_switch_after_r.sum() / total_pun.sum()]) 
            
            fig.add_trace(go.Bar(x = ['pun', 'n', 'r'],
                                 y = switch_rates,
                                 name = self.subject_names[n],
                                 marker_color = self.colors[n]
            ))
            fig_num_switches.add_trace(go.Bar(x=['switches'],
                                             y = [df_total_switches.sum()],
                                             name = self.subject_names[n],
                                             marker_color = self.colors[n]))
            
#             ## Pair improvement?
#             gp2 = df.groupby(['low_stim', 'high_stim'])
            
#             def compute_acc_improvement(x):
#                 if x.shape[0] > 3:
#                     acc_first_half = x['acc'].iloc[:int(x.shape[0]*.33)].mean()
#                     acc_last_half = x['acc'].iloc[(int(x.shape[0]*.66)):].mean()
#                     return acc_last_half - acc_first_half
#                 else:
#                     return np.nan
            
#             acc_improvement = gp2.apply(compute_acc_improvement)
#             #print(acc_improvement[~acc_improvement.isna()])
            
# #             if self.subject_names[n] in ['220','241']:
# #                 idxs = ~acc_improvement.isna()
# #                 print(np.sum(gp2['acc'].count() > 3))
# #                 display(gp2['acc'].mean())
# #                 display(acc_improvement[idxs])
#             fig_acc_improve.add_trace(go.Bar(x=['avg_acc_improvement'],
#                                             y = [acc_improvement.mean()],
#                                             name = self.subject_names[n],
#                                             marker_color = self.colors[n]))
            
        
#         fig_acc_improve.update_layout(title = 'Average Accuracy Improvement',
#                                      yaxis_title = 'Improvement')
#         fig_acc_improve.show()

            df = self.create_subject_df(n=n)
            df = df[df['feedback'] < .5]
            
            df['low_stim'] = np.minimum(df['stim1'], df['stim2'])
            df['high_stim'] = np.maximum(df['stim1'], df['stim2'])
        
            ## Stim improvement?
            stim_type = np.zeros(self.Nc[n]) - 1
            for st in range(self.Nc[n]):
                df_st_idx = df[df['stim1'] == st].index.union(df[df['stim2'] == st].index)
                first_idx = df_st_idx[0]
                if df.loc[first_idx,'stim1'] == st:
                    stim_type[st] = df.loc[first_idx,'type1']
                if df.loc[first_idx,'stim2'] == st:
                    stim_type[st] = df.loc[first_idx,'type2']
        
            df = df[~df['acc'].isna()]
            choice_rate_diff = np.full((self.Nc[n]), np.nan)
            choice_rate = np.full((self.Nc[n]), np.nan)
            stim_nf_num = np.zeros(self.Nc[n])
            for st in range(self.Nc[n]):
                df_st_idx = df[df['stim1'] == st].index.union(df[df['stim2'] == st].index)

                num = df_st_idx.size
                first_choice_freq = (df.loc[df_st_idx[:int(.4 * num)],'C'] == st).mean() 
                last_choice_freq = (df.loc[df_st_idx[int(.6 * num):],'C'] == st).mean()
                choice_rate_diff[st] = last_choice_freq - first_choice_freq
                choice_rate[st] = last_choice_freq
  
                stim_nf_num[st] = num 
            fig_stim_num = go.Figure()
            fig_stim_num.add_trace(go.Scatter(x = np.arange(self.Nc[n]),
                                     y = stim_nf_num,
                                             name = self.subject_names[n],
                                             marker_color = self.colors[n]))
            fig_stim_num.update_layout(title = f'{self.subject_names[n]}) Number of repetitions per stim',
                                   xaxis_title = 'Stim ID',
                                   yaxis_title = 'Num nf appearances')
            fig_stim_num.show()

            df_stims = pd.DataFrame({'type': stim_type, 'choice_rate_diff': choice_rate_diff, 'choice_rate': choice_rate})

            type_diff = df_stims.groupby('type')['choice_rate_diff'].mean()
            #type_diff_std = df_stims.groupby('type')['choice_rate_diff'].std()
            cr = df_stims.groupby('type')['choice_rate'].mean()
            
            #print(choice_rate_diff)

            fig_crd.add_trace(go.Bar(x=type_diff.index,
                                     y = type_diff,
                                     name= self.subject_names[n],
                                     marker_color = self.colors[n]))
            fig_cr.add_trace(go.Bar(x=cr.index,
                                   y= cr,
                                   name = self.subject_names[n],
                                   marker_color = self.colors[n]))
        fig.update_layout(title = 'Nf Switch Rate within Pair Repetition after Outcome',
                         xaxis_title = 'Outcome',
                         yaxis_title = 'Proportion')
        fig.show()
        fig_num_switches.update_layout(title = 'Total Number of Switches',
                                      yaxis_title = 'Count')
        fig_num_switches.show()
        fig_crd.update_layout(title = 'Mean change in choice rate between first 40% and last 40%',
                              xaxis_title = 'Stim type',
                              yaxis_title = 'Choice Rate Diff')
        fig_crd.show()
        
        fig_cr.update_layout(title = 'Choice Rate in last 40%',
                            xaxis_title = 'Stim_type',
                             yaxis_title = 'Choice Rate')
        fig_cr.show()
            
            
        
        if save_folder is not None:
            fig.write_image(os.path.join(save_folder, 'Outcome-stratified-switches.jpeg'))
            fig_num_switches.write_image(os.path.join(save_folder, 'Outcome-stratified-switches.jpeg'))

            
                
        
            
    def plot_nf_acc_learning_var_stratified(self, save_folder = None, var_stratification = 'valence_last'):
        """
        Computes the average accuracy on no-feedback trials, startified by 
        high var_stratification (above median) and low var_stratification (below median)
        combinations between stim1 and stim2.
        
        - var_stratification: string, name of the variable to stratify on.
        """
        
        fig = go.Figure()
        
        for n in range(self.N):
            df = self.create_subject_df(n=n)
            
            df['high_var1'] = (df[var_stratification + '_1'] > np.median(df[var_stratification + '_1'])).astype('int64')
            df['high_var2'] = (df[var_stratification + '_2'] > np.median(df[var_stratification + '_2'])).astype('int64')
            
            df = df[df['feedback'] < .5]
            df = df[np.logical_and(df['stim1'] > 16,  df['stim2'] > 16)]
                          
            acc = df.groupby(['high_var1', 'high_var2'])['acc'].sum()
            num_gp = df.groupby(['high_var1', 'high_var2'])['acc'].count()
            
            #display(acc)
            #display(num_gp)
            
            indxs = ["(False, False)", "(True, False)", "(True, True)"]
            vals = [acc.loc[(0,0)] / num_gp.loc[(0,0)],
                   (acc.loc[(0,1)] + acc.loc[(1,0)]) / (num_gp.loc[(0,1)] + num_gp.loc[(1,0)]),
                   acc.loc[(1,1)] / num_gp.loc[(1,1)]]
            
            fig.add_trace(go.Scatter(x =indxs, y = vals,
                                    name = self.subject_names[n],
                                    marker_color = self.colors[n],
                                    mode = 'lines+markers'))
            
        fig.update_layout(title = f"No-feedback accuracy stratified by {var_stratification}",
                         xaxis_title = f"High {var_stratification} of stimulus pair",
                         yaxis_title = "Accuracy",
                         yaxis = {'range':[.7,1.]})
        fig.show()
        
        if save_folder is not None:
            fig.write_image(os.path.join(save_folder, f'nf_acc_{var_stratification}.jpeg'))
            
            

    def plot_num_practice_trials_per_block(self, save_folder = None):
        """
        Displays the number of practice stimuli per block.
        """
        fig = go.Figure()
        for n in range(self.N):
            df = self.create_subject_df(n = n)
            
            num = df.groupby('block').apply(lambda x: np.sum(np.logical_or(x['stim1'] < 18, x['stim2'] < 18)))
            num = num.groupby(pd.cut(num.index, [5.5] + list(np.arange(6.5, 60, 4)) + [61.5])).sum()
            
            fig.add_trace(go.Bar(x=[str(interval) for interval in num.index], y=num.values, marker_color = self.colors[n], name = self.subject_names[n]))
        fig.update_layout(title = f"Number of trials with practice stimuli per block",
                             xaxis_title = "block",
                              yaxis_title = "count")#,
                         #yaxis = {'range' : [0,72*4]})
        fig.show()
        
        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
            fig.write_image(os.path.join(save_folder, 'practice_trials'))
            
#     def plot_nf_type_acc(self, save_folder = None):
#         """
#         Plots the average no-feedback accuracy per stimulus (2 per trial) startified over type.
        
#         This is the average over stimuli of the same type of average stimulus accuracy.
#         """
        
#         fig = go.Figure()
#         for n in range(self.N):
#             df = self.create_subject_df(n = n)
            
            
            
#             fig.add_trace(go.Bar(x=[str(interval) for interval in num.index], y=num.values, marker_color = self.colors[n], name = self.subject_names[n]))
#         fig.update_layout(title = f"Number of trials with practice stimuli per block",
#                              xaxis_title = "block",
#                               yaxis_title = "count")#,
#                          #yaxis = {'range' : [0,72*4]})
#         fig.show()
        
#         if save_folder is not None:
#             if not os.path.exists(save_folder):
#                 os.mkdir(save_folder)
#             fig.write_image(os.path.join(save_folder, 'practice_trials'))
        
    ### Other auxiliary functions ###
    def compute_num_learning_blocks(self):
        """        
        Computes the number of block with learning (at least one feedback trial),
        in self.num_learning_blocks, as an (N)-numpy array.
        """
        
        self.num_learning_blocks = np.zeros((self.N)).astype('int64')
        for n in range(self.N):
            df_n = pd.DataFrame({'block': self.block[:,n], 'feedback': self.feedback[:,n]})
            num_feedback_per_block_n = df_n.groupby('block').sum()#transform(lambda x: x.sum())
            self.num_learning_blocks[n] = np.sum(num_feedback_per_block_n > 0)
            
    def compute_num_outcomes_per_block(self, subject_names):
        
        subject_names, columns = self.subject_names_to_columns(subject_names, False)
        
        df_counts = []
        for idx, n in enumerate(columns):
            df = pd.DataFrame({'o': self.o[:,n], 'block': self.block[:,n]})
            df_counts.append(df.groupby('block')['o'].apply(lambda x: x.value_counts()))

        df_counts = pd.concat(df_counts, axis = 1)
        df_counts.columns = subject_names
        print(df_counts)
        
    def compute_num_outcomes_block_stratified(self, subject_names):
        subject_names, columns = self.subject_names_to_columns(subject_names, False)

        #print(subject_names)
        #print(columns)
        df_counts = []
        for idx, n in enumerate(columns):
            df = pd.DataFrame({'o': self.o[:,n], 'feedback': self.feedback[:,n], 'C': self.C[:,n],'block': self.block[:,n]})
            df = df[df['feedback'] == 1]
            df['>=2new'] = np.repeat(self.num_new[:,n] >= 2,48)
            #print(df.iloc[:100,:])
            df_counts.append(df.groupby('>=2new')['o'].apply(lambda x : x.value_counts()))
            #print(df_counts[-1])
            
            
        df_counts = pd.concat(df_counts, axis = 1)
        df_counts.columns = subject_names
        print(df_counts)
        
    def compute_num_outcomes_block_stratified_loop(self, subject_names):
        subject_names, columns = self.subject_names_to_columns(subject_names, False)

        N = len(subject_names)
        
        tuples = [('New', -1.), ('New', 0.), ('New', 1.), ('Old', -1.), ('Old', 0.), ('Old', 1.)]
        index = pd.MultiIndex.from_tuples(tuples, names=["first", "second"])
        
        df_counts = {'New': np.zeros((3,N)),
                    'Old': np.zeros((3,N))}
        
        for k,n in enumerate(columns):
            for t in range(self.T):
                if self.feedback[t,n] == 1:
                    b = self.block[t,n] - self.block[0,n]
                    if self.num_new[b,n] >= 2:
                        df_counts['New'][int(self.o[t,n]) + 1,k] += 1
                    else:
                        df_counts['Old'][int(self.o[t,n]) + 1,k] += 1
                        
        print(df_counts)
               
    def compute_num_outcomes_full_block_stratified_loop(self, subject_names):
        subject_names, columns = self.subject_names_to_columns(subject_names, False)

        N = len(subject_names)
        
        tuples = [('0 new', -1.), ('0 new', 0.), ('0 new', 1.), ('1 new', -1.), ('1 new', 0.), ('1 new', 1.),('2 new', -1.), ('2 new', 0.), ('2 new', 1.),('3 new', -1.), ('3 new', 0.), ('3 new', 1.),]
        index = pd.MultiIndex.from_tuples(tuples, names=["first", "second"])
        
        df_counts = np.zeros((12, N))
        
        for k,n in enumerate(columns):
            for t in range(self.T):
                if self.feedback[t,n] == 1:
                    b = self.block[t,n] - self.block[0,n]
                    n_n = self.num_new[b,n]
                    df_counts[int(self.o[t,n]) + 1 + 3*n_n, k] += 1
                        
        df_counts = pd.DataFrame(df_counts, index = index, columns = subject_names)
        print(df_counts)
        
        acc = self.acc       
        
        df_acc_all = []
        for k,n in enumerate(columns):
            df_acc = pd.DataFrame({'acc': acc[:,n], 'feedback': self.feedback[:,n], 'num_new': self.num_new[self.block[:,n] - self.block[0,n], n * np.ones(self.T).astype('int64')]})
            df_acc = df_acc[df_acc['feedback'] == 1]
            df_acc_all.append(df_acc.groupby('num_new')['acc'].mean())
        df_acc_all = pd.concat(df_acc_all, axis = 1)
        df_acc_all.columns = subject_names
        print(df_acc_all)
        
        fig_str = go.Figure()
        for k, s in enumerate(subject_names):
            fig_str.add_trace(go.Bar(name = s, x = ['0', '1', '2', '3'],
                                   y = df_acc_all[s],
                                   marker_color = self.colors[columns[k]]))
        fig_str.update_layout(barmode='group', title ='Accuracy per new_stim',
                                 yaxis={
                                    'title':'Acc',
                                    'range': [0.5, 1.]  
                                })
        
        avg_acc = np.zeros(N)
        for k,n in enumerate(columns):  
            avg_acc[k] = np.mean(acc[self.feedback[:,n] == 1, n])
        print(avg_acc)
        
        df_acc_all_norm = pd.DataFrame(df_acc_all)
        for k, clm in enumerate(df_acc_all.columns):
            df_acc_all_norm[clm] = df_acc_all[clm] / avg_acc[k]
        print(df_acc_all_norm)
        
        fig_avg = go.Figure()
        fig_norm = go.Figure()
        for k, s in enumerate(subject_names):
            fig_avg.add_trace(go.Bar(name = s, x = ['average accuracy'],
                                    y = [avg_acc[k]], 
                                    marker_color = self.colors[columns[k]]))
            fig_norm.add_trace(go.Bar(name = s, x = ['0', '1', '2', '3'],
                       y = df_acc_all_norm[s],
                       marker_color = self.colors[columns[k]]))
        fig_avg.update_layout(
        title = 'Average Accuracy', 
        yaxis={
            'title':'Acc',
            'range': [0.5, 1.]  
        })
        
        fig_norm.update_layout(barmode='group', title = 'Acc per new_stim, relative to overall accuracy',
                                 yaxis={
                                    'title': 'Acc',
                                    'range': [0.8, 1.20]  
                                })
        fig_avg.show()
        fig_str.show()
        fig_norm.show()
        
        df_acc_nf = []
        fig_nf = go.Figure()
        for k,n in enumerate(columns):
            df_acc = pd.DataFrame({'acc': acc[:,n], 'feedback': self.feedback[:,n], 'num_new': self.num_new[self.block[:,n] - self.block[0,n], n * np.ones(self.T).astype('int64')]})
            df_acc = df_acc[df_acc['feedback'] == 0]
            df_acc_nf.append(df_acc['acc'].mean())
            fig_nf.add_trace(go.Bar(name = subject_names[k],
                                   x = ['acc'],
                                   y = [df_acc_nf[-1]],
                                   marker_color = self.colors[k]))
        df_acc_nf = pd.DataFrame({s:[df_acc_nf[k]] for k,s in enumerate(subject_names) })
        df_acc_nf.columns = subject_names
        print(df_acc_nf)
        
        fig_nf.update_layout(
        title = 'Average Accuracy over no-feedback trials', 
        yaxis={
            'title':'Acc',
            'range': [0.5, 1.]  
        })
        fig_nf.show()
        
    def compute_num_nf_practice_stim(self):
        """
        Computes the number of appearances of practice stimuli on no-feedback trials.
        
        Prints these per subject.
        """
        
        for n in range(self.N):
            df = self.create_subject_df(n=n)
            
            df = df[df['feedback'] < .5]
            
            num_practice_stim1 = np.sum(df['stim1'] < 18)
            num_practice_stim2 = np.sum(df['stim2'] < 18)
            
            print(f"{self.subject_names[n]}) Num nf-practice stim: {num_practice_stim1 + num_practice_stim2}")
        
    def coin_checks(self, subject_names, verbose = True):
        subject_names, columns = self.subject_names_to_columns(subject_names, False)

        N = len(subject_names)
        
        # For self object, doesn't span all subjects, so this is just test code.
        
        print("\n\nStim type match-ups in whole series, on feedback trials:")
        if verbose:
            for k,n in enumerate(columns):
                df_stim_types = pd.DataFrame(self.stim_types[self.feedback[:,n]==1,:,k])
                df_stim_types.columns = ['stim1', 'stim2']
                print(f"{subject_names[k]}:\n{df_stim_types.value_counts() / np.sum(self.feedback[:,n])}\n")
        
        print(f"\n\nStimuli types per block:")
        self.stim_types_block = [] 
        self.new_stim_types_block = []
        for k,n in enumerate(columns):
            stims_b = []
            stim_type_b = []
            seen_stims = []
            new_stim_type_b = []
            
            self.stim_types_block.append([])
            self.new_stim_types_block.append([])
             
            #b0 = self.block[0,n]
            t = 0         
            while t < self.T[n]- 1:
                if self.feedback[t,n] == 1:
                    b = self.block[t,n]
                    stim1 = self.stims[t,0,n]
                    stim2 = self.stims[t,1,n]
                    if stim1 not in stims_b:
                        stims_b.append(stim1)
                        stim_type_b.append(self.stim_types[t,0,k])
                    if stim2 not in stims_b:
                        stims_b.append(stim2)
                        stim_type_b.append(self.stim_types[t,1,k])
                    if stim1 not in seen_stims:
                        seen_stims.append(stim1)
                        new_stim_type_b.append(self.stim_types[t,0,k])
                    if stim2 not in seen_stims:
                        seen_stims.append(stim2)
                        new_stim_type_b.append(self.stim_types[t,1,k])
                        

                if self.block[t+1,n] != b:
                    #print(sorted(stim_type_b))
                    self.stim_types_block[-1].append(sorted(stim_type_b))
                    stim_type_b = []
                    stims_b = []
                    
                    self.new_stim_types_block[-1].append(new_stim_type_b)
                    new_stim_type_b = []

                t+=1
                
        print("Outcomes per coin type:")
        if verbose:
            for k,n in enumerate(columns):
                type_chosen = self.stim_types[np.arange(self.T), self.C_st[:,n], k]
                df = pd.DataFrame({'o':self.o[:,n], 'type': type_chosen, 'feedback': self.feedback[:,n], 'block': self.block[:,n]})
                df = df[df['feedback'] == 1]

                coin_type_outcome = df.groupby('type')['o'].apply(lambda x:x.value_counts())

                print(f"{subject_names[k]}:\n{coin_type_outcome}\n\n")

                #self.stim_type_block[n].append(new_stimuli_b)
                
        print("Checking how frequently D was chosen:")
        frac_D = []
        fig_D = go.Figure()
        for k,n in enumerate(columns):
            stim_types_n = self.stim_types[self.feedback[:,n] == 0, :, n]
            C_st_n = self.C_st[self.feedback[:,n] == 0, n]
            #print(stim_types_n)
            
            total_D = np.sum(stim_types_n[:,:] == 3) 
            left_D  = np.sum(np.logical_and(stim_types_n[:,0] == 3, C_st_n == 0))
            right_D = np.sum(np.logical_and(stim_types_n[:,1] == 3, C_st_n == 1))
            both_D  = np.sum(np.logical_and(stim_types_n[:,0] == 3, stim_types_n[:,1] == 3))
            print(f"{subject_names[k]}: {total_D:4d} {left_D:4d} {right_D:4d} {both_D:4d}")
            
            frac_D.append((left_D + right_D - both_D) / (total_D-both_D))
            fig_D.add_trace(go.Bar(name = subject_names[k], x = ['frac'],
                                   y = [frac_D[-1]],
                                   marker_color=self.colors[columns[k]]))
            
        frac_D = pd.DataFrame({s: [frac_D[k]] for k, s in enumerate(subject_names)})
        print(frac_D)
        
        fig_D.update_layout(
        title = 'Fraction of D choices on trials with only 1 D.', 
        yaxis={
            'title':'Fraction',
            'range': [0., 1.]  
        })
        fig_D.show()
        
        # Accuracy on no-feedback D-trials
        acc = self.acc
      
        df_acc_nf = []
        fig_nf = go.Figure()
        for k,n in enumerate(columns):
            df_acc = pd.DataFrame({'acc': acc[:,n], 'feedback': self.feedback[:,n], 'D': np.logical_or(self.stim_types[:,0,n] == 3, self.stim_types[:,1,n] == 3), 'both_D': np.logical_and(self.stim_types[:,0,n] == 3, self.stim_types[:,1,n] == 3),
                                  'C':np.logical_or(self.stim_types[:,0,n] == 2, self.stim_types[:,1,n] == 2),
                                  'A': np.logical_or(self.stim_types[:,0,n] == 0, self.stim_types[:,1,n] == 0),
                                  'B': np.logical_or(self.stim_types[:,0,n] == 1, self.stim_types[:,1,n] == 1)})
            df_acc = df_acc[(df_acc['feedback'] == 0) & (df_acc['D']) & (~df_acc['both_D']) & (~df_acc['C'])]
            #print(df_acc.head(10))
            print(f"{subject_names[k]}: {df_acc.shape}")
            df_acc_nf.append(df_acc['acc'].mean())
            fig_nf.add_trace(go.Bar(name = subject_names[k],
                                   x = ['acc'],
                                   y = [df_acc_nf[-1]],
                                   marker_color = self.colors[k]))
        df_acc_nf = pd.DataFrame({s:[df_acc_nf[k]] for k,s in enumerate(subject_names) })
        df_acc_nf.columns = subject_names
        print(df_acc_nf)
        
        fig_nf.update_layout(
        title = 'Average Accuracy over no-feedback D trials (versus A or B)', 
        yaxis={
            'title':'Acc',
            'range': [0.5, 1.]  
        })
        fig_nf.show()
        
        # Accuracy over feedback D-trials where D appears the second time
        df_acc_nf = []
        fig_nf = go.Figure()
        for k,n in enumerate(columns):
            df_acc = pd.DataFrame({'acc': acc[:,n], 'feedback': self.feedback[:,n], 'D': np.logical_or(self.stim_types[:,0,n] == 3, self.stim_types[:,1,n] == 3), 'both_D': np.logical_and(self.stim_types[:,0,n] == 3, self.stim_types[:,1,n] == 3),
                                  'C':np.logical_or(self.stim_types[:,0,n] == 2, self.stim_types[:,1,n] == 2),
                                  'A': np.logical_or(self.stim_types[:,0,n] == 0, self.stim_types[:,1,n] == 0),
                                  'B': np.logical_or(self.stim_types[:,0,n] == 1, self.stim_types[:,1,n] == 1),
                                  'num_new': self.num_new[self.block[:,n] - self.block[0,n], n * np.ones(self.T).astype('int64')],
                                  'block': self.block[:,n]})
            df_acc = df_acc[(df_acc['feedback'] == 1) & (df_acc['D']) & (~df_acc['both_D']) & (~df_acc['C'])]
            if subject_names[k] in ['204', '205', '208', '210']:
                df_acc = df_acc[(df_acc['block'] > 7) & (df_acc['num_new'] == 0)]
            else:
                df_acc = df_acc[df_acc['num_new'] == 2]
            
            print(f"{subject_names[k]}: {df_acc.shape}")
            df_acc_nf.append(df_acc['acc'].mean())
            fig_nf.add_trace(go.Bar(name = subject_names[k],
                                   x = ['acc'],
                                   y = [df_acc_nf[-1]],
                                   marker_color = self.colors[k]))
        df_acc_nf = pd.DataFrame({s:[df_acc_nf[k]] for k,s in enumerate(subject_names) })
        df_acc_nf.columns = subject_names
        print(df_acc_nf)
        
        fig_nf.update_layout(
        title = 'Average Accuracy over feedback D trials, second learning block', 
        yaxis={
            'title':'Acc',
            'range': [0.5, 1.]  
        })
        fig_nf.show()

        
#         print()
        
#         for k,n in enumerate(columns):
            
#             for t in range(self.T):

def copy_spec(spec):
    
    spec_copy = {par_name: {'type': spec[par_name]['type'], 'val': spec[par_name]['val'].copy()} for par_name in spec.keys()}
    for par_name in spec.keys():
        if 'fixed_val' in spec[par_name].keys():
            spec_copy[par_name]['fixed_val'] = spec[par_name]['fixed_val']
            
    return spec_copy

# def llik(self, P, P_aux, n, plot_lik = False, T_max = None, T0 = None, llik_nf0 = None, Q0 = None, N_table0 = None, ESS_it_interval = 300, ESS_threshold = 0., ESS_only_nf_llik = False, return_Q = False, min_stim = 18, max_stim = 10000, AIS_shift = False, stim_nf_max = 0, d_delay = 0, verbose = False, P_q = None):
    # return self.llik(P, P_aux, n, plot_lik = plot_lik, T_max = T_max, T0 = T0, llik_nf0 = llik_nf0, Q0 = Q0, N_table0 = N_table0, ESS_it_interval = ESS_it_interval, ESS_threshold = ESS_threshold, ESS_only_nf_llik = ESS_only_nf_llik, return_Q = return_Q, min_stim = min_stim, max_stim = max_stim, AIS_shift = AIS_shift, stim_nf_max = stim_nf_max, d_delay = d_delay, verbose = verbose, P_q = P_q)

def process_EM_n(n, spec, S, s_dat, ESS_it_interval, ESS_threshold, m, old_fit):
    P = sampleP(spec, S) # Refers to the sampleP function outside of the RL_model

    P_aux = sampleP_aux_new(s_dat, P, spec, n)  
    
    if m.llik_style == 'advanced_template':
        llik_P, llik_f, llik_nf, ESS, S_final, ESS_scaled, _, _, _, _ = llik_adv(P, P_aux, n, m.subjects_data, m.style, return_Q = False, ESS_it_interval = ESS_it_interval)#llik(m, P, P_aux, n, ESS_it_interval = ESS_it_interval, ESS_threshold = ESS_threshold)        
        if m.llik_type == 'NF':
            llik_P = llik_nf
        elif m.llik_type == 'F':
            llik_P = llik_f
        elif llik_type != 'All':
            sys.exit('Invalid llik_type.')
    elif m.llik_style == 'simple_template':
        llik_P, llik_f, llik_nf, _, _, _ = llik_simple(P, P_aux, n, m.subjects_data, m.style, return_Q = False)#llik(m, P, P_aux, n, ESS_it_interval = ESS_it_interval, ESS_threshold = ESS_threshold)        
        if m.llik_type == 'NF':
            llik_P = llik_nf
        elif m.llik_type == 'F':
            llik_P = llik_f
        elif llik_type != 'All':
            sys.exit('Invalid llik_type.')
        ESS = None
        S_final = None
        ESS_scaled = None
    elif m.llik_style == 'custom':
        llik_P, _ = m.llik(P, P_aux, n, m.subjects_data, return_Q = False)
        llik_nf = None
        llik_f  = None
        ESS     = None
        S_final = None
        ESS_scaled = None
    
    lsumlik  = logsumexp_1D(llik_P)
    evidence = lsumlik - np.log(S)
    weights  = np.exp(llik_P - lsumlik)
    fit      = compute_estimate(m.I, weights, P, P_aux = P_aux, n = n, AIS_version = False)
    fit['evidence']    = old_fit['evidence'] + [evidence]
    if llik_f is not None:
        fit['evidence_f']  = old_fit['evidence_f']  + [logsumexp_1D(llik_f) - np.log(S)]
    else:
        fit['evidence_f']  = old_fit['evidence_f']  + [np.nan]
    if llik_nf is not None:
        fit['evidence_nf'] = old_fit['evidence_nf'] + [logsumexp_1D(llik_nf) - np.log(S)]
    else:
        fit['evidence_nf'] = old_fit['evidence_nf'] + [np.nan]
    fit['ESS']         = old_fit['ESS']         + [1. / np.sum(np.square(weights))]

    fit['N_samples']   = S
    
    return llik_P, llik_f, llik_nf, ESS, S_final, ESS_scaled, fit

def process_AIS_n(m, n, S, models_folder, q_type, min_its, num_mixtures, mean_type, var_type, min_df, max_df, num_improvement_its_stop, old_q_fit):
    return m.AIS_subject_mixture(n_subject = n, S = S, save_folder = models_folder, mixture_normalization = True, q_type = q_type, min_its = min_its, max_df = max_df, min_df = min_df, num_mixtures = num_mixtures, mean_type = mean_type, var_type = var_type, display_q_spec = 100, num_improvement_its_stop = num_improvement_its_stop, display_end_diagnostics = False, multiple_CPU = True, old_q_fit = old_q_fit)    

                    
class RL_model:
    """    
    This class defines a choice model. It contains functions to sample & fit from the model, display parameter fits,
    and perform parameter fit analyses.
    
    The main functionalities are:
    1. initialization                          : defines the model through its style, spec, subjects_data and name.
    2. sampleP, sampleP_aux                    : functions used to sample parameter values.
    3. llik                                    : defines the computations to obtain the log-likelihood of the model.
    4. fit_model, compute_estimates, fit_prior : used to fit the model using EM or EM-AIS.
    5a. save, load                             : You may not have expected this, but these save and load model fits.
    5b. input_subjects                         : Puts subjects into a loaded model.
    6a. plot_est, plot_ests_time_varying       : make figures of the estimates.
    6b. plot_true_vs_est, plot_Q               : estimates vs true and plots the Q trajectory of a given parameter setting.
    ... A few more got added. See the tutorial file.
    
    The RL_model class currently assumes the hyperprior is marginally independent. In a follow-up layer of P_aux parameter values, 
    parameters can be dependent however. (Example: the daily-varying subjective values.)
    
    Initialization data:
    - subject_data: an object of the 'subjects' class, containing the subjects' data.
        - N            : number of subjects
        - Nc           : Number of choices
        - T            : subject horizon (currently all the same)
        - subject_names: list of strings, IDs of the subjects.
    - name             : string, name of the model.
    - style: a dictionnary containing:
        - Q_style      : string, the style used for Q updates.
        - choice_style : string, the style of choice function.
        - R_style      : string, type of perceived reward. 
        - T_style      : string, specifies how time-based updates on the Q-function are performed.
        - Q_style_2    : None or string. If string, the Q_style for a second Q-table.
        - T_style_2    : None or string: If string, the T_style for a second Q-table.
    - spec             : dictionary specifying the hyperprior of the model. 
                         It has the following (key, value) pairs: {par_name : par_spec}, 
                         par_name (string) being the parameter name, 
                         par_spec being a dictionary with keys 'type' (distribution type) and 'val' (hyperprior values)
    
    While fitting, the following get updated:
    - evidence    : float, model evidence
    - bic         : float, current Bayesian Information Criterion measure for the model fit.    
    - fit_its     : int, current number of fitting iterations over all subjects, using the hierarchical Expectation Maximization algorithm.
    - S           : number of parameter samples
    - P           : (S, .) dataframe, with 'par_name' as columns. The parameters used in the hierarchical EM algorithm.
    - fit         : list of N dictionaries of fitted paramters and statistics, one per subject, with (key, value) pairs
                    'evidence' : float, log mean likeliood over all the sampled parameters 
                    'N_samples': int, number of samples with finite likelihood
                    'ESS': Effective Sample Size
                    'P' dict with 'par_name': dict with 'val' : float, mean of the 'par_name' posterior
                                                        'ci'  : (2) numpy array, 95% credibility interval of the posterior 
                                                        'samp': (S) numpy array, systematically resampled values, to reflect p('par_name'|data)
   - ESS_mean: float, the latest mean og the Effective Sample Size over all subjects.
   - num_nopractice_f: (N) array, number of feedback trials with no practice stimuli.
   - num_nopractice_nf: (N) array, number of no-feedback trials with no practice stimuli.
   
   For AIS:
   - q_fits: list of length N. q_specs[n] is either None if no q fitting has been done yet, or has a q_spec i it does. Then, it is of the form:
               {'spec': q_spec, 'type': q_type, 'num_its': num_its, 'ESS': ESS_evolution, 'evidence': evidence_evolution}
   - The fit is updated in self.fit.

   
    NOTE: - The parameters for the second Q-table are specified with 'par_name'_2. If one is missing, the same as the 'par_name' are used.
          - If there are two Q-tables and subjective rewards, the Q-tables use the same subjective value.
          - Two Q-tables can not operate with the dir-bayes Q_styles at this time.
    """
    
    ### 1. Initialization:
    def __init__(self, subjects_data = subjects_EMA(), name = "", spec = {}, empty = False, verbose = True, style_adv = None, style_simple = None, llik_custom = None):        
        if empty:
            if verbose:
                print("- Created an empty RL_model.")
        else:
            self.subjects_data = subjects_data
            self.N = subjects_data.N
            self.subject_names = subjects_data.subject_names
            self.colors = subjects_data.colors
            self.name = name
            
            # Create the llik function:
            if style_adv is not None:
                self.style = style_adv
                self.llik_style = 'advanced_template'
                if style_simple is not None or llik_custom is not None:
                    sys.exit("Must specify exactly one of 'style_adv', 'style_simple' or 'llik_custom'.")  
                self.llik_type = style_adv['llik_type']
            elif style_simple is not None:
                self.style = style_simple
                self.llik_style = 'simple_template'
                self.llik_type = style_simple['llik_type']
                if llik_custom is not None:
                    sys.exit("Must specify exactly one of 'style_adv', 'style_simple' or 'llik_custom'.")                    
            elif llik_custom is not None:
                self.llik = llik_custom
                self.llik_style = 'custom'
                self.llik_type = None
            else:
                sys.exit("Must specify exactly one of 'style_adv', 'style_simple' or 'llik_custom'.")
            
            # Extract the number of datapoints:
            if self.llik_type is not None:
                # Get the number of datapoints:               
                self.num_nopractice_f  = subjects_data.num_nopractice_f
                self.num_nopractice_nf = subjects_data.num_nopractice_nf
                if self.llik_type == 'All':
                    self.num_datapoints    = subjects_data.num_datapoints # number of datapoints per subject
                elif self.llik_type == 'NF':
                    self.num_datapoints    = subjects_data.num_nopractice_nf
                elif self.llik_type == 'F':
                    self.num_datapoints    = subjects_data.num_nopractice_f
            else:
                self.num_nopractice_f  = subjects_data.num_nopractice_f
                self.num_nopractice_nf = subjects_data.num_nopractice_nf 
                self.num_datapoints = subjects_data.num_datapoints
              
            # Decode the style:
            if self.llik_style in ['advanced_template', 'simple_template']:
                self.Q_style = self.style['Q_style']
                if 'T_style' in self.style.keys():
                    self.T_style = self.style['T_style']
                else:
                    self.T_style = None
                if 'Q_style_2' in self.style.keys():
                    self.Q_style_2 = self.style['Q_style_2']
                else:
                    self.Q_style_2 = None
                if 'T_style_2' in self.style.keys():
                    if self.Q_style_2 is None:
                        sys.exit("Cannot have Q_style_2 unspecified while having T_style_2.")
                    self.T_style_2 = self.style['T_style_2']
                else:
                    self.T_style_2 = None
                self.choice_style = self.style['choice_style']
                self.R_style = self.style['R_style']
            
            # Prior specification:
            self.spec = copy_spec(spec)
            self.spec_trace = [copy_spec(spec)]
            
            # Fits and fit statistics:
            self.evidence = []
            self.evidence_f = []
            self.evidence_nf = []
            self.bic = []
            self.fit_its = 0
            self.S = None
            self.P = None
            self.P_aux = None # Auxiliary, non-fit, parameters
            self.fit = [{'evidence': [], 'evidence_f': [], 'evidence_nf': [], 'ESS': [], 'N_samples' : np.nan, 'P': {}, 'P_aux': {}} for n in range(self.N)]
            self.q_fits = [None for n in range(self.N)]
            #{'spec_q': [], 'type': [{'q':np.nan, 'mean': np.nan, 'var': np.nan, 'num_mixtures': np.nan, 'random_init': np.nan}], 
                                              # 'S_its': [], 
                                              # 'num_its' : np.nan,
                                             # 'ESS': [np.nan],
                                             # 'evidence': [np.nan],
                                             # 'P': [],
                                             # 'P_aux': [],
                                             # 'fit_within_current_EM': np.nan,
                                             # 'evidence_mxs': [],
                                             # 'time_it': []}
            self.ESS_mean = []
            self.llik_it = np.array([np.nan])
            self.ESS = [np.nan for n in range(self.N)] # Keeps track of sequences of ESS
            self.ESS_it = np.nan
            self.runtime_it = []
            self.ev_jump = np.nan
            
            self.I = 1000 # Number of posterior samples per subject while fitting the model
            
            # AIS fits:
            self.q_fits = [None for n in range(self.N)]
            self.q_fitted_EM_it = [] # True if the given EM iteration was fit with AIS
                       
    def get_style(self):           
        return self.style
   
    ### 2. sampleP, sampleP_aux:            
    def sampleP(self, S, n):
        """
        Samples parameter values according to the parameter specifications in 'self.spec'.
        
        Parameter values that are used to fit the hyperprior are stored in P. Auxiliary parameters that are needed for the llik function are stored in P_aux.
        
        Arguments:
        - S: int, the number of parameters to be sampled.
        - n: int, the index of the subject.
        
        Creates a new (S, .) dataframe 'self.P', with S parameter values.
        """
        self.S = S
                
        self.P = sampleP(self.spec, S) # Refers to the sampleP function outside of the RL_model
                
        self.P_aux = sampleP_aux_new(self.subjects_data, self.P, self.spec, n)  
    
    ### Deprecated, a function to subsample the parameters by weight probabilities:
    def subsample(self, S_old, S_new, weights):
        # Alternatively, kill the lowest weighted ones.
        return np.random.choice(S_old, size = S_new, replace = False, p = weights)       
    
    ### Helper function to compute posterior quantities in the fitting algorithm 'fit_model()':
    def compute_estimates(self, llik, llik_f, llik_nf, n, P, P_aux):
        """
        Computes the evidence for the parameters samples on subject n, and computes the posterior mean, median, maximum weight samples,
        credibility interval and self.I posterior samples of the approximate posterior.
        
        Arguments:
        - llik    : (S) numpy array, log-likelihoods of the S parameter samples in 'self.P' for subject n.
        - llik_f  : (S) numpy array, log-likelihoods on the feedback trials of the S parameter samples.        
        - llik_nf : (S) numpy array, log-likelihoods on the no-feedback trials of the S parameter samples.           
        - n       : int, the index of the subject to compute the posterior distrbution quantities of,
                         only used to store the results in self.fit[n].
        - P       : (S,.) - DataFrame, the parameter values for the parameters to be fitted.
        - P_aux   : (S,.) - DataFrame, the parameter values for the auxiliary parameters. No self.I posterior samples will be computed on these.
        
        Updates the dictionary self.fit[n] with:
            - 'evidence': model evidence log(sum_s p(data_n| P[s,:]))
            - 'samples' : number of samples with finite log-likelihood.
            - 'P': dict with keys 'par_name' linking to a dict with
                   - 'val': posterior mean of the parameter 'par_name'
                   - 'ci' : 95% credibility interval for the parameter 'par_name'
                   - 'samp': 1000 uniform samples of the posterior
                   - 'median': posterior median
                   - 'max'   : list of the 5 samples with maximal weight
            - 'P_aux': dict with keys 'par_name' linking to a dict with 
                   - 'val': posterior mean of the parameter 'par_name'
                   - 'ci' : 95% credibility interval for the parameter 'par_name'
                   - 'median': posterior median
                   - 'max'   : list of the 5 samples with maximal weight
        """
        
        fit = {}
        # Find the lines where there is numerical overflow or underflow:
        inc = ~((np.abs(llik) < 1e-12) | (llik == np.nan))
        if(np.sum(~inc) > 0):
            sys.exit(f'\n==> subject {n}) llik underflow or overflow\n')
        self.fit[n]['N_samples'] = llik[inc].shape[0]
        
        # Compute the evidence estimate (log mean likelihood):
        lsumlik = logsumexp_1D(llik)
        self.fit[n]['evidence'].append(lsumlik - np.log(self.fit[n]['N_samples']))
        
        # Compute the evidence estimate on feedback and no-feedback trials:
        if llik_f is not None and llik_nf is not None:
            lsumlik_f = logsumexp_1D(llik_f); lsumlik_nf = logsumexp_1D(llik_nf)
            self.fit[n]['evidence_f'].append(lsumlik_f - np.log(self.fit[n]['N_samples']))
            self.fit[n]['evidence_nf'].append(lsumlik_nf - np.log(self.fit[n]['N_samples']))
        else:
            self.fit[n]['evidence_f'].append(np.nan)
            self.fit[n]['evidence_nf'].append(np.nan)
        
        # Compute weights for resampling:
        weights = np.exp(llik - lsumlik)
        
        # Compute the Effective Sample Size:
        self.fit[n]['ESS'].append(1. /  np.sum(np.square(weights)))
        
        ## TEMP: print 10 highest unnormalized weights, ESS, log(S), 
#         ranks = np.argsort(llik_nf)[::-1][:10]
#         lprior_m = compute_lpdf_prior(self.P, self.P_aux, self.spec, self.subjects_data, self)
        
#         print(f"Evidence: {self.fit[n]['evidence'][-1]}") 
#         print(f"Evidence_nf: {self.fit[n]['evidence_nf'][-1]}")
#         print(f'10 highest weights: {llik_nf[ranks]}')
#         print(f'prior: {lprior_m[ranks]}')
#         print(f'log(S): {np.log(llik_nf.shape)[0]}')
#         print(f"log(N_samples): {self.fit[n]['N_samples']}")
        
        # Compute Approximate Posterior quantities:
        for par_name in P.columns:
            par_values = P[par_name].values
            param_fit = {}
       
            # Order the parameter values from small to large
            ranks = np.argsort(par_values)
            oparams = par_values[ranks]
            oweights = weights[ranks]
                 
            # Mean of the posterior:
            param_fit['val'] = np.sum(oweights * oparams)
            param_fit['mean'] = np.sum(oweights * oparams)
            
            # Maximum weights:
            maxweights = np.argsort(weights)[::-1][:5]
            param_fit['max'] = par_values[maxweights]            
            
            # Credibility interval:
            cdf = np.cumsum(oweights)
            ci_thresh = np.array([.025, .5, .975])
            param_fit['ci'] = np.zeros(2) 
            param_fit['median'] = np.nan
            i = 0; j = 0
            while i < 3:
                if cdf[j] > ci_thresh[i]:
                    if i == 0:
                        param_fit['ci'][0] = oparams[j]
                    elif i == 1:
                        param_fit['median'] = oparams[j]
                    elif i == 2:
                        param_fit['ci'][1] = oparams[j]
                    i += 1
                else:
                    j += 1
                   
            # 1000 equally spaced quantiles:
            I = self.I
            sampp = np.arange(1. / (2.*I), 1., 1. / I)
            samp = np.zeros(I)
            i = 0; j = 0
            ci_i = 0
            while i < I:
                if cdf[j] > sampp[i]:
                    samp[i] = oparams[j]
                    i += 1
                else:
                    j += 1
            param_fit['samp'] = samp
            
            self.fit[n]['P'][par_name] = param_fit
            
        if P_aux is not None:
            for par_name in P_aux.columns:
                par_values = P_aux[par_name].values
                param_fit = {}

                # Order the parameter values from small to large
                ranks = np.argsort(par_values)
                oparams = par_values[ranks]
                oweights = weights[ranks]

                # Mean of the posterior:
                param_fit['val'] = np.sum(oweights * oparams)
                param_fit['mean'] = np.sum(oweights * oparams)
                
                # Maximum weights:
                maxweights = np.argsort(weights)[::-1][:5]
                param_fit['max'] = par_values[maxweights]               
                
                # Credibility interval:
                cdf = np.cumsum(oweights)
                ci_thresh = np.array([.025, .5, .975])
                param_fit['ci'] = np.zeros(2) 
                param_fit['median'] = np.nan
                i = 0; j = 0
                while i < 3:
                    if cdf[j] > ci_thresh[i]:
                        if i == 0:
                            param_fit['ci'][0] = oparams[j]
                        elif i == 1:
                            param_fit['median'] = oparams[j]
                        elif i == 2:
                            param_fit['ci'][1] = oparams[j]
                        i += 1
                    else:
                        j += 1
                        
                self.fit[n]['P_aux'][par_name] = param_fit
            
    
    ### A helper function to fit a distribution to the approximate posterior samples, used in 'fit_model()':    
    def fit_prior(self):
        """
        Given the model and posterior samples in the self.fit[:]['P'][:]['samp'],
        fits the prior to the samples of all subjects combined. The space of prior models
        is specified in 'self.spec'.
        
        Updates 'self.spec' with the new values.
        """
        
        for p, par_name in enumerate(self.spec.keys()):
            
            # The current prior specification
            old_prior_val = self.spec[par_name]['val'].copy()
            h_names = []
            
            prior_type = self.spec[par_name]['type']
            
            # Collect all the samples of the subject posteriors:
            if prior_type not in ['logBB_stat', 'min_logBB_stat','BB_stat','logitBB_stat', 'OU', 'logOU', 'min_logOU', 'logitOU', 'OU_fixed_mean', 'logOU_fixed_mean', 'min_logOU_fixed_mean', 'logitOU_fixed_mean']:
                samp = np.concatenate([self.fit[n]['P'][par_name]['samp'] for n in range(self.N)])
                        
            if prior_type == 'normal':
                self.spec[par_name]['val'] = list(scipy.stats.norm.fit(samp))
                h_names = ['mean  ', 'stddev']
            elif prior_type == 'beta':
                a, b, _, _ = scipy.stats.beta.fit(samp, floc = 0, fscale = 1)
                self.spec[par_name]['val'] = [a, b]
                h_names = ['alpha', 'beta ']
            elif prior_type == 'gamma':
                loc, _, scale = scipy.stats.gamma.fit(samp, floc = 0)
                self.spec[par_name]['val'] = [loc, scale]
                h_names = ['shape', 'scale']
            elif prior_type == 'min_gamma':
                loc, _, scale = scipy.stats.gamma.fit(-samp, floc = 0)
                self.spec[par_name]['val'] = [loc, scale]
                h_names = ['shape', 'scale']
            elif prior_type == 'binom':
                self.spec[par_name]['val'] = [np.mean(samp)]
                h_names = ['p']
            elif prior_type == 'lognormal':
                self.spec[par_name]['val'] = list(scipy.stats.norm.fit(np.log(samp)))  
                h_names = ['mean  ', 'stddev']               
            elif prior_type == 'min_lognormal':
                self.spec[par_name]['val'] = list(scipy.stats.norm.fit(np.log(-samp)))  
                h_names = ['mean  ', 'stddev']
            elif prior_type == 'logitnormal':
                self.spec[par_name]['val'] = list(scipy.stats.norm.fit(logit(samp))) 
                h_names = ['mean  ', 'stddev']
            elif prior_type == 'logBB_stat' or prior_type == 'min_logBB_stat':            
                h_names = [f'log_{par_name}_0      mean  ', f'log_{par_name}_0      stddev', f'log_{par_name}_sigma  mean  ', f'log_{par_name}_sigma  stddev']
                # var0:
                samp0 = np.concatenate([self.fit[n]['P'][f'{par_name}_0']['samp'] for n in range(self.N)])
                if prior_type == 'logBB_stat':
                    vals0 = list(scipy.stats.norm.fit(np.log(samp0)))
                elif prior_type == 'min_logBB_stat':
                    vals0 = list(scipy.stats.norm.fit(np.log(-samp0)))
                self.spec[par_name]['val'][0] = vals0[0]
                self.spec[par_name]['val'][1] = vals0[1]
                
                # sigma:
                samp_sigma = np.concatenate([self.fit[n]['P'][f'{par_name}_sigma']['samp'] for n in range(self.N)])
                vals_sigma = list(scipy.stats.norm.fit(np.log(samp_sigma)))
                self.spec[par_name]['val'][2] = vals_sigma[0]
                self.spec[par_name]['val'][3] = vals_sigma[1]
                
                # step:
                samp_s = []
                for n in range(self.N):
                    for P_name in self.fit[n]['P'].keys():
                        # Account for variable length:
                        if f'{par_name}_step' in P_name:
                            samp_s.append(self.fit[n]['P'][P_name]['samp'])
                samp_s = np.concatenate(samp_s)
                vals_s = list(scipy.stats.norm.fit(samp_s, floc = 0.))
                self.spec[par_name]['val'][4]   = vals_s[1]
                h_names += [f'{par_name}_step       stddev']
            elif prior_type == 'BB_stat':            
                h_names = [f'{par_name}_0          mean  ', f'{par_name}_0          stddev', f'log_{par_name}_sigma  mean  ', f'log_{par_name}_sigma  stddev']
                # var0:
                samp0 = np.concatenate([self.fit[n]['P'][f'{par_name}_0']['samp'] for n in range(self.N)])
                vals0 = list(scipy.stats.norm.fit(samp0))
                self.spec[par_name]['val'][0] = vals0[0]
                self.spec[par_name]['val'][1] = vals0[1]
                
                # sigma:
                samp_sigma = np.concatenate([self.fit[n]['P'][f'{par_name}_sigma']['samp'] for n in range(self.N)])
                vals_sigma = list(scipy.stats.norm.fit(np.log(samp_sigma)))
                self.spec[par_name]['val'][2] = vals_sigma[0]
                self.spec[par_name]['val'][3] = vals_sigma[1]
                
                # step:
                samp_s = []
                for n in range(self.N):
                    for P_name in self.fit[n]['P'].keys():
                        # Account for variable length:
                        if f'{par_name}_step' in P_name:
                            samp_s.append(self.fit[n]['P'][P_name]['samp'])
                samp_s = np.concatenate(samp_s)
                vals_s = list(scipy.stats.norm.fit(samp_s, floc = 0.))
                self.spec[par_name]['val'][4]   = vals_s[1]
                h_names += [f'{par_name}_step       stddev']
            elif prior_type == 'logitBB_stat':            
                h_names = [f'logit{par_name}_0     mean  ', f'logit{par_name}_0     stddev', f'log_{par_name}_sigma  mean  ', f'log_{par_name}_sigma  stddev']
                # var0:
                samp0 = np.concatenate([self.fit[n]['P'][f'{par_name}_0']['samp'] for n in range(self.N)])
                vals0 = list(scipy.stats.norm.fit(logit(samp0)))
                self.spec[par_name]['val'][0] = vals0[0]
                self.spec[par_name]['val'][1] = vals0[1]
                
                # sigma:
                samp_sigma = np.concatenate([self.fit[n]['P'][f'{par_name}_sigma']['samp'] for n in range(self.N)])
                vals_sigma = list(scipy.stats.norm.fit(np.log(samp_sigma)))
                self.spec[par_name]['val'][2] = vals_sigma[0]
                self.spec[par_name]['val'][3] = vals_sigma[1]
                
                # step:
                samp_s = []
                for n in range(self.N):
                    for P_name in self.fit[n]['P'].keys():
                        # Account for variable length:
                        if f'{par_name}_step' in P_name:
                            samp_s.append(self.fit[n]['P'][P_name]['samp'])
                samp_s = np.concatenate(samp_s)
                vals_s = list(scipy.stats.norm.fit(samp_s, floc = 0.))
                self.spec[par_name]['val'][4]   = vals_s[1]
                h_names += [f'{par_name}_step       stddev']

            elif prior_type in ['OU', 'logOU', 'min_logOU', 'logitOU']:
                h_names = [f'{par_name}_mu        mean  ', f'{par_name}_mu        stddev', f'{par_name}_rate      mean  ', f'{par_name}_rate      stddev', f'{par_name}_stat_std  mean  ', f'{par_name}_stat_std  stddev']
                
                # mu:
                samp0 = np.concatenate([self.fit[n]['P'][f'{par_name}_mu']['samp'] for n in range(self.N)])
                vals0 = list(scipy.stats.norm.fit(samp0))
                self.spec[par_name]['val'][0] = vals0[0]
                self.spec[par_name]['val'][1] = vals0[1]
                
                # rate:
                samp_rate = np.concatenate([self.fit[n]['P'][f'{par_name}_rate']['samp'] for n in range(self.N)])
                vals_rate = list(scipy.stats.norm.fit(np.log(samp_rate)))
                self.spec[par_name]['val'][2] = vals_rate[0]
                self.spec[par_name]['val'][3] = vals_rate[1]
                
                # stat_std:
                samp_sigma = np.concatenate([self.fit[n]['P'][f'{par_name}_stat_std']['samp'] for n in range(self.N)])
                vals_sigma = list(scipy.stats.norm.fit(np.log(samp_sigma)))
                self.spec[par_name]['val'][4] = vals_sigma[0]
                self.spec[par_name]['val'][5] = vals_sigma[1]
                
            elif prior_type in ['OU_fixed_mean', 'logOU_fixed_mean', 'min_logOU_fixed_mean', 'logitOU_fixed_mean']:
                h_names = [f'{par_name}_rate      mean  ', f'{par_name}_rate      stddev', f'{par_name}_stat_std  mean  ', f'{par_name}_stat_std  stddev']
                
                # mu:
                # stays the same
                
                # rate:
                samp_rate = np.concatenate([self.fit[n]['P'][f'{par_name}_rate']['samp'] for n in range(self.N)])
                vals_rate = list(scipy.stats.norm.fit(np.log(samp_rate)))
                self.spec[par_name]['val'][0] = vals_rate[0]
                self.spec[par_name]['val'][1] = vals_rate[1]
                
                # stat_std:
                samp_sigma = np.concatenate([self.fit[n]['P'][f'{par_name}_stat_std']['samp'] for n in range(self.N)])
                vals_sigma = list(scipy.stats.norm.fit(np.log(samp_sigma)))
                self.spec[par_name]['val'][2] = vals_sigma[0]
                self.spec[par_name]['val'][3] = vals_sigma[1]
                    
            elif prior_type == 'fixed':
                h_names = []
                print(f"{par_name:10s} - fixed_val: {self.spec[par_name]['fixed_val']}")
                    
            elif prior_type != 'fixed':
                sys.exit(f"{par_name} does not have a valid 'spec['{param_name}']['type']' ('{prior_type}' given)."+ \
             "\nTry one of the following:\n- 'normal' with val (loc, scale)\n- 'beta' with val (a, b)" + \
             "\n- 'gamma' with val (loc, scale)\n- 'binom' with val (p)\n- 'lognormal' with val (loc, scale)\n- the OU/BB variants")
        
            for l, h_name in enumerate(h_names):
                if l == 0:            
                    print(f"{par_name:10s} - {h_name} -  {old_prior_val[l]:.4f} --> {self.spec[par_name]['val'][l]:.4f}.")
                else:
                    print(" " * 10 + f" - {h_name} -  {old_prior_val[l]:.4f} --> {self.spec[par_name]['val'][l]:.4f}.")
        print("")
            

    
    ### 4. The main workhorse function to fit models, either using the EM algorithm, or EM-AIS, or EM-IBIS, or EM-AIS-IBIS    
    def fit_model(self, S, models_folder, verbose = True, debug_verbose = False, min_improvement = .5, ESS_it_interval = 300, plot_ESS = False, ESS_threshold = 0., plot_lik = False, AIS_dict = None, n_start = 0, num_CPU = None):
        """
        Function to fit the given model with the Hierarchical Expectation-Maximization algorithm or Hierarchical EM-AIS: it performs
        Type 2 Maximal Likelihood Estimation / Empirical Bayes to fit the hyperprior that the subject parameters get drawn from.        
        
        The fitting procedure keeps finding new hyperprior fits until there is no improvement in overal BIC_int.
        
        Arguments:
        - S              : int, the number of parameter samples per subject to approximate the subject posteriors.
        - models_folder  : string, the name of the folder to save model fits in. This happens every EM iteration and after every AIS subject fit.
        - n_start        : int, the index of the subject to start fitting from. Mostly used when the algorithm crashes and you want to start from where it crashed.
        - num_CPU        : None, or int. If None, runs unparallellized fitting. Else uses num_CPU cpus, speeding up code, at the cost of fewer diagnostics and less clear debugging.
        - AIS_dict       : None, or dictionary. If None, uses vanilla EM. If not, uses the dictionary to specify the EM-AIS algorithm.
        
        Igonore for now:
        - IBIS_dict: specifications to use the EM-IBIS algorithm.
        - ESS_it_interval: int, estimates the ESS every 'ESS_it_interval' steps.
        - plot_ESS       : boolean, if True, plots ESS progessions.
        - plot_lik       : plots parameter likelihoods if True.
            
        Iteratively updates self.spec with new hyperprior values, self.fit/self.q_fit with posterior fits, and computes the BIC_int of the corresponding model.
        """
           
        improvement = np.nan
        it = 1 # Iterations in the current algorithm call
        #self.ESS_it = [np.arange(0, self.T[n], ESS_it_interval) for n in range(self.N)] # ESS trajectories that get computed while computing the llik of subjects.
        llik_it = [np.nan for n in range(self.N)]       # Storage for all llik
        P_it = [np.nan for n in range(self.N)]          # Storage for all final parameters, used for EM only to plot parameter likelihoods.
        
        while improvement > min_improvement or np.isnan(improvement):
            print(f"Iteration {self.fit_its:2d}:\n-------------")
            tm = time.time()
            if len(self.bic) == 0:
                oldbic = np.nan
            else:
                oldbic = self.bic[-1]
            
            # Non-parallellized version:
            if num_CPU is None:
                for n in range(n_start, self.N):                
                    # # Option 2: EM-IBIS: ignore for now
                    # if IBIS_dict is not None:
                        # IBIS_style = IBIS_dict['style']
                        # IBIS_day_style = IBIS_dict['day_style']
                        # IBIS(self, n, S, models_folder, only_nf_llik = only_nf_llik, IBIS_style = IBIS_style, day_style = IBIS_day_style, verbose = debug_verbose, diagn = (50000,2000), collapse_rejections = False, df = 10)
                    
                    # Option 3: EM-AIS
                    if AIS_dict is not None:
                        # Load the EM-AIS specifications from the dictionary:
                        q_type = AIS_dict['q_type']            # Type of approximators: 'robust' (strongly preferred) or 'standard'
                        if len(self.q_fitted_EM_it) == 0:      # Check whether tha EM algorithm has been run before.
                            min_its = AIS_dict['min_its_init'] # If so, uses a higher number of minimum iterations, as given by min_its_init.
                            num_improvement_its_stop = 1
                        else:
                            if self.q_fitted_EM_it[-1]:
                                min_its = AIS_dict['min_its_ctd'] # After the first EM-AIS iteration, use a lower number of minimum iterations, min_its_ctd.
                                num_improvement_its_stop = 2      # But need at least two consecutive iterations that have lower evidence than two ago to stop.
                            else:
                                min_its = AIS_dict['min_its_init'] # In the first EM-AIS iteration after EM, use min_its_init as well.
                                num_improvement_its_stop = 1
                        mean_type = AIS_dict['mean_type']         # Method to update the mean of the distributions.
                        var_type  = AIS_dict['var_type']          # Method to update the variance of the distributions.
                        num_mixtures = AIS_dict['num_mixtures']   # The number of fixed mixtures (fixed mixtures don't work so well, better to explore adaptive mixtures)
                        if 'keep_q_spec' in AIS_dict:
                            keep_q_spec = AIS_dict['keep_q_spec']
                        else:
                            keep_q_spec = False
                        if 'max_df' in AIS_dict:                  # The maximum degrees of freedom of the robust approximators.
                            max_df = AIS_dict['max_df']
                        else:
                            max_df = 8
                        if 'min_df' in AIS_dict:                  # The minimum degrees of freedom of the robust approximators.
                            min_df = AIS_dict['min_df']
                        else:
                            min_df = 8
                            
                        # Perform the fits for one EM iteration:    
                        print(f"{self.subject_names[n]})")
                        self.AIS_subject_mixture(n_subject = n, S = S, save_folder = models_folder, mixture_normalization = True, q_type = q_type, min_its = min_its, max_df = max_df, min_df = min_df, num_mixtures = num_mixtures, mean_type = mean_type, var_type = var_type, display_q_spec = 100, num_improvement_its_stop = num_improvement_its_stop, display_end_diagnostics = False, keep_q_spec = keep_q_spec)    
                        # Transfer the AIS fits to EM fits:
                        self.transfer_AIS_to_fit(n)
                        # Save the model:
                        self.save(models_folder)    
                        print("") 
                    
                    # Option 1: vanilla EM
                    else:
                        # Sample the hyperprior:
                        self.sampleP(S, n)
                        
                        # Compute llik:
                        if self.llik_style == 'advanced_template':
                            llik_P, llik_f, llik_nf, ESS, _, _, _, _, _, _ = llik_adv(self.P, self.P_aux, n, self.subjects_data, self.style, return_Q = False, ESS_it_interval = ESS_it_interval)
                            if self.llik_type == 'NF':
                                llik_P = llik_nf   
                            elif self.llik_type == 'F':
                                llik_P = llik_f
                            elif self.llik_type != 'All':
                                sys.exit(f"Incorrect llik_type! '{self.llik_type}' given but it should be in ('All', 'F', 'NF').") 
                            self.ESS[n] = ESS
                        elif self.llik_style == 'simple_template':
                            llik_P, llik_f, llik_nf, _, _, _ = llik_simple(self.P, self.P_aux, n, self.subjects_data, self.style, return_Q = False)
                            if self.llik_type == 'NF':
                                llik_P = llik_nf   
                            elif self.llik_type == 'F':
                                llik_P = llik_f
                            elif self.llik_type != 'All':
                                sys.exit(f"Incorrect llik_type! '{self.llik_type}' given but it should be in ('All', 'F', 'NF').")
                        elif self.llik_style == 'custom':
                            llik_P, _ = self.llik(self.P, self.P_aux, n, self.subjects_data, return_Q = False)
                            llik_f = None; llik_nf = None
                        else:
                            sys.exit("Found no llik function!")
                                                       
                        llik_it[n] = llik_P
                        P_it[n] = self.P
                        
                        # Compute Posterior Quantities:
                        self.compute_estimates(llik_P, llik_f, llik_nf, n, self.P, self.P_aux)                
            else:
                pool = mp.Pool(num_CPU)
                
                if AIS_dict is None:                    
                    results = pool.starmap(process_EM_n, [(n, self.spec, S, self.subjects_data, ESS_it_interval, ESS_threshold, self, self.fit[n]) for n in np.arange(n_start,self.N).astype('int32')])                   
                    
                    for n in range(n_start, self.N):
                        llik_it[n] = results[n - n_start][0]
                        P_it[n] = np.nan
                        self.ESS[n] = results[n - n_start][3]; 
                        self.fit[n] = results[n - n_start][6]
                        
                else:
                    q_type = AIS_dict['q_type']
                    # To initialize the algorithm, use a higher number of AIS iterations:
                    if len(self.q_fitted_EM_it) == 0:
                        min_its = AIS_dict['min_its_init']
                        num_improvement_its_stop = 1
                    # Takes another number in consecutive iterations:
                    else:
                        if self.q_fitted_EM_it[-1]:
                            min_its = AIS_dict['min_its_ctd']
                            num_improvement_its_stop = 2
                        else:
                            min_its = AIS_dict['min_its_init']
                            num_improvement_its_stop = 1
                    mean_type = AIS_dict['mean_type']
                    var_type  = AIS_dict['var_type']
                    num_mixtures = AIS_dict['num_mixtures']
                    if 'max_df' in AIS_dict:                  # The maximum degrees of freedom of the robust approximators.
                        max_df = AIS_dict['max_df']
                    else:
                        max_df = 8
                    if 'min_df' in AIS_dict:                  # The minimum degrees of freedom of the robust approximators.
                        min_df = AIS_dict['min_df']
                    else:
                        min_df = 8
                    
                    results = pool.starmap(process_AIS_n, [(self, n, S, models_folder, q_type, min_its, num_mixtures, mean_type, var_type, min_df, max_df, num_improvement_its_stop, self.q_fits[n]) for n in np.arange(n_start,self.N).astype('int32')])
                        
                    
                    for n in range(n_start, self.N):
                        self.q_fits[n] = results[n]
                        self.transfer_AIS_to_fit(n)
                        self.plot_ESS_and_Evidence_AIS(n, save_folder = models_folder, show_figs = False) 
                    self.save(models_folder)                          
                    
            n_start = 0
            
            # Plot the ESS if desired:
            if plot_ESS:
                self.plot_ESS()
            
            # Compute the BIC_int, evidences etc.:
            N_hyperparams = np.sum([len(self.spec[par_name]['val']) for par_name in self.spec])
            self.evidence.append(np.sum([self.fit[n]['evidence'][-1] for n in range(self.N)]))
            self.evidence_f.append(np.sum([self.fit[n]['evidence_f'][-1] for n in range(self.N)]))
            self.evidence_nf.append(np.sum([self.fit[n]['evidence_nf'][-1] for n in range(self.N)]))  

            self.bic.append(- 2. * self.evidence[-1] + N_hyperparams * np.log(np.sum(self.num_datapoints)))
            self.ESS_mean.append(np.mean([self.fit[n]['ESS'][-1] for n in range(self.N)]))          
            # Compute the improvement in BIC_int:
            improvement = oldbic - self.bic[-1]
            it = it + 1
            self.fit_its += 1
            # A boolean for whether the current iteration was EM-AIS fit or not:
            if AIS_dict is not None:
                self.q_fitted_EM_it.append(True)
            else:
                self.q_fitted_EM_it.append(False)
                          
            if improvement > min_improvement or np.isnan(improvement):
                # If there's improvement, save the model. 
                # Note that if there's no improvement, the model will not be saved (only for vanilla EM). So if you load it, you will have the model with the best iBIC.
                self.save(models_folder)
                
                # Fit the hyperprior based on the posterior samples of all subjects:
                self.fit_prior()
                self.spec_trace.append(copy_spec(self.spec))
            else:
                # Always save when doing EM-IBIS or EM-AIS, because those algorithms always save after approximating one subject posterior.
                # So, the final, worse approximation gets saved.
                if AIS_dict is not None:
                    self.save(models_folder)
                
            if verbose:
                if AIS_dict is not None:
                    S_sampled = S * AIS_dict['num_mixtures']
                else:
                    S_sampled = S
                print(f"==> BIC_int   -    old: {oldbic:.2f},    new: {self.bic[-1]:.2f}.")
                print(f"    Evidence  - {self.evidence[-1]:.2f} ({self.evidence_f[-1]:.2f} / {self.evidence_nf[-1]:.2f}).")
                print(f"    Mean ESS  - {self.ESS_mean[-1]:.2f} / {S_sampled}.")
                print(f"    Stdev ESS - {np.std([self.fit[n]['ESS'][-1] for n in range(self.N)]):.2f}.")
                print(f"    Time      - {time.time() - tm:.2f}s.\n")
                
            self.runtime_it.append(time.time() - tm)
            if len(self.bic) > 1:
                self.ev_jump = self.evidence[-1] - self.evidence[-2]
            
            # If plotting the likelihood is desired: (TODO: review)
            if plot_lik:
                if AIS_dict is not None:
                    sys.exit("fit_model > No implementation yet to show the estimated lik for the AIS fits.")
                results_folder = os.path.join(model_folder, self.name)
                if not os.path.exists(results_folder):
                    os.mkdir(results_folder)
               
                figs = [go.Figure() for par_name in self.P]
                for n in range(self.N):
                    weights = np.exp(llik_it[n] -  logsumexp_1D(llik_it[n]))

                    # To speed up fitting:
                    idxs = weights > 1. / (10. * weights.shape[0])
                    
                    for k, par_name in enumerate(self.P):
                        weighted = sm.nonparametric.KDEUnivariate(P_it[n][par_name].iloc[idxs])
                        weighted.fit(fft=False, weights= weights[idxs])

                        figs[k].add_trace(go.Scatter(x = weighted.support, y = weighted.density, marker_color= self.colors[n], name= self.subject_names[n], mode = 'lines'))
                for k, par_name in enumerate(self.P):
                    save_path = os.path.join(model_folder, self.name, f"lik_{par_name}.jpeg")
                
                    figs[k].update_layout(title = par_name)
                    figs[k].show()
                    figs[k].write_image(save_path)
        
        # Plot the ESS and Evidence estimates per EM iteration:
        for n in range(self.N):
            self.plot_ESS_and_Evidence(n, save_folder = models_folder, show_figs = False)
                
    # def plot_ESS(self):
        # """
        # Plots the ESS lines.
        # """
        # fig = go.Figure()
        # for n in range(self.N):
            # fig.add_trace(go.Scatter(x = self.ESS_it[n], y = self.ESS[n], mode = 'lines', name = self.subject_names[n], line = {'color': self.colors[n]}))
        # fig.update_layout(title = "ESS", xaxis_title = "iteration #")
        # fig.show()
        
        # fig = go.Figure()
        # for n in range(self.N):
            # fig.add_trace(go.Scatter(x = self.ESS_it[n], y = self.ESS_scaled[n], mode = 'lines', name = self.subject_names[n], line = {'color': self.colors[n]}))
        # fig.update_layout(title = "ESS / S_final", xaxis_title = "iteration #")
        # fig.show()
        
    ### AIS ###
    def AIS_subject_mixture(self, n_subject = 0, num_mixtures = 5, mixture_normalization = True, min_its = 1, llik_type = 'All', q_type = 'robust', mean_type = 'ESS', var_type = 'ESS', S = 10000, max_df = 8, min_df = 4, display_q_spec = 50, display_weighted_trajs = 50, display_par_names = ['R_mean', 'R_asym'], display_end_diagnostics = True, random_init = True, global_variation_inflation = True, save_folder = None, max_day = 1000, max_stim = 1000, num_improvement_its_stop = 3, multiple_CPU = False, old_q_fit = None, keep_q_spec = False):
        """
        Performs Adaptive Importance Sampling on one subject.
        
        - S_its: list of length total_its, specifiying the number of samples at each iteration
        - display_q_spec: int. The q_spec will be displayed after every 'display_q_spec' iterations.
        - display_weighted_trajs: int. Shows the 7 sampled trajectories with heighest weights every 'display_weighted_trajs' trajectories.
        - display_par_names: list, names of the temporal parameters to display.
        """
        s_dat = self.subjects_data
        
        if self.llik_style in ['advanced_template', 'simple_template']:
            style = self.get_style()
                              
        ESS = []
        evidence = []
        last_ev = np.nan
        it = 0
        improvement_in_last_3 = True # Can now specify how many improvement its are needed before stopping in 'num_improvement_its_stop' (default 3)
        old_evidence = None
            
        tm = time.time()
        while improvement_in_last_3 or it < min_its:
            time_it = time.time()
            print(f'\r- Iteration {it:2d}, elapsed time: {time.time() - tm:5.2f}s, evidence estimate: {last_ev:.2f}.', end = '')
                        
            # 0) Init: First sampler specification
            if (not multiple_CPU and self.q_fits[n_subject] is None) or (multiple_CPU and old_q_fit is None):
                # Construct an initial sampler:
                spec_q = construct_initial_sampler_EIS_mixture_new(self, n_subject, q_type = q_type, num_mixtures = num_mixtures, random_init = random_init) 
                if display_q_spec == 1:
                    spec_q_df = [{} for mx in spec_q]
                    for mx, spec_q_mx in enumerate(spec_q):
                        for par_name in spec_q_mx.keys():
                            vals = list(spec_q_mx[par_name]['val'])
                            if len(vals) == 2:
                                vals.append(np.nan)
                            spec_q_df[mx][par_name] = vals
                        display(pd.DataFrame(spec_q_df[mx]))
            else:
                # Load the previous spec:
                if not multiple_CPU:
                    spec_q = self.q_fits[n_subject]['spec_q'][-1]
                else:
                    spec_q = old_q_fit['spec_q'][-1]
                                                       
            # 1) sample q:
            P_q = []
            for mx, spec_q_mx in enumerate(spec_q):
                P_q.append(sampleP(spec_q_mx, S))
            
            # 2) Transform the parameters of q to parameters required for the llik:
            P_m = []; P_aux = []
            for mx, P_q_mx in enumerate(P_q):
                P_m_mx, P_aux_mx = P_q_to_m_new(P_q_mx, self, n_subject)
                P_m.append(P_m_mx)
                P_aux.append(P_aux_mx)
                                       
            llik_P = np.empty((num_mixtures, S))
            lprior_q = np.empty((num_mixtures, S))
            lprior_m = np.empty((num_mixtures, S))
            lweights_u = np.empty((num_mixtures, S))
            evidence_mxs = np.empty(num_mixtures)
            
            for mx, P_m_mx in enumerate(P_m):                
                # 3) Compute llik:
                # self.P_aux = P_aux[mx]
                # self.P = P_m[mx]
                if self.llik_style == 'advanced_template':
                    llik, llik_f, llik_nf, _, _, _, _, _, _, _ = llik_adv(P_m[mx], P_aux[mx], n_subject, self.subjects_data, self.style, return_Q = False, ESS_it_interval = 10000)#self.llik(P_m[mx], P_aux[mx], n_subject, False, 10000)

                    if self.llik_type == 'NF':
                        llik_P[mx, :] = llik_nf   
                    elif self.llik_type == 'F':
                        llik_P[mx, :] = llik_f
                    elif self.llik_type == 'All':
                        llik_P[mx, :] = llik
                    elif self.llik_type != 'All':
                        sys.exit(f"Incorrect llik_type! '{self.llik_type}' given but it should be in ('All', 'F', 'NF').")     
                elif self.llik_style == 'simple_template':
                    llik, llik_f, llik_nf, _, _, _ = llik_simple(P_m[mx], P_aux[mx], n_subject, self.subjects_data, self.style, return_Q = False)#self.llik(P_m[mx], P_aux[mx], n_subject, False, 10000)

                    if self.llik_type == 'NF':
                        llik_P[mx, :] = llik_nf   
                    elif self.llik_type == 'F':
                        llik_P[mx, :] = llik_f
                    elif self.llik_type == 'All':
                        llik_P[mx, :] = llik
                    elif self.llik_type != 'All':
                        sys.exit(f"Incorrect llik_type! '{self.llik_type}' given but it should be in ('All', 'F', 'NF').")     
                elif self.llik_style == 'custom':
                    llik_P[mx, :], _ = self.llik(P_m[mx], P_aux[mx], n_subject, self.subjects_data, return_Q = False)
                    llik_f = np.nan; llik_nf = np.nan
                else:
                    sys.exit("Found no llik function!") 
                       
                # 4) Compute q-pdf:
                if mixture_normalization:
                    lprior_q_mx = np.zeros((num_mixtures, S))
                    for mx_2 in range(num_mixtures):
                        lprior_q_mx[mx_2,:] = compute_lpdf_q_new(P_q[mx], spec_q[mx_2], self.spec, self.subjects_data, n_subject, max_day = max_day, q_type = q_type)
                    lprior_q[mx, :] = logsumexp_dim0(lprior_q_mx) - np.log(num_mixtures)
                else:
                    lprior_q[mx, :] = compute_lpdf_q_new(P_q[mx], spec_q[mx], self.spec, self.subjects_data, n_subject, max_day = max_day, q_type = q_type)                    

                # 5) Compute model prior pdf:
                lprior_m[mx, :] = compute_lpdf_prior_new(P_m[mx], P_aux[mx], self.spec, self.subjects_data, self, n_subject, max_day = max_day)                
                
            
            # 6) Compute local weights, evidence and ESS:
            lweights_u = llik_P + lprior_m - lprior_q # Unnormalized log-weights
            for mx in range(num_mixtures):
                evidence_mxs[mx] = logsumexp_1D(lweights_u[mx, :]) - np.log(S)
                      
            
            lsumlik = logsumexp_1D(lweights_u.flatten())
            evidence.append(lsumlik - np.log(S*num_mixtures)) ##
            last_ev = evidence[-1]
            weights = np.exp(lweights_u - lsumlik) # (num_mixtures, S)
            ESS.append(1. / np.sum(np.square(weights))) ## 
            
            ## Maximal lweights:
            # print(lweights_u.shape)
            # maxw = np.argsort(lweights_u[0,:])[::-1][:5]
            # print(f"\n\nAIS:")
            # print(lweights_u[0,maxw])
            # print(lprior_m[0,maxw])
            # print(lprior_q[0,maxw])
            # print(P_aux[0]['R_asym_27'][maxw.flatten()])
            # print(P_aux[0]['R_mean_27'][maxw.flatten()])
            # print(P_aux[0]['R_asym_26'][maxw.flatten()])
            # print("\n")
                                 
            # 6b) Compute the current parameter estimates:
            fit = compute_estimate(self.I, weights, P_m, P_aux = P_aux, n = n_subject)
            
            # 7) Fit q:
            spec_q, gamma = fit_q_mixture_new(self, spec_q, P_q, n_subject, lweights_u = lweights_u, max_df = max_df, min_df = min_df, mean_type = mean_type, var_type = var_type, global_variation_inflation = global_variation_inflation)                
            if (it % display_q_spec == 0 and it != 0) or display_q_spec == 1:
                spec_q_df = [{} for mx in spec_q]
                for mx, spec_q_mx in enumerate(spec_q):
                    for par_name in spec_q_mx.keys():
                        vals = list(spec_q_mx[par_name]['val'])
                        if len(vals) == 2:
                            vals.append(np.nan)
                        spec_q_df[mx][par_name] = vals
                    display(pd.DataFrame(spec_q_df[mx]))
                    
            # 8) Store the q specifications and metrics:
            if not multiple_CPU:
                if self.q_fits[n_subject] is None:
                    self.q_fits[n_subject] = {'spec_q': [spec_q], 
                                              'type': [{'q':q_type, 'mean': mean_type, 'var': var_type, 'num_mixtures': num_mixtures, 'random_init': random_init}], 
                                              'S_its': [S], 
                                              'num_its' : 1,
                                             'ESS': [ESS[-1]],
                                             'evidence': [evidence[-1]],
                                             'P': [fit['P']],
                                             'P_aux': [fit['P_aux']],
                                             'fit_within_current_EM': True,
                                             'evidence_mxs': [evidence_mxs],
                                             'time_it': [time.time() - time_it]}

                else:
                    self.q_fits[n_subject]['S_its'].append(S)
                    self.q_fits[n_subject]['num_its'] += 1
                    self.q_fits[n_subject]['ESS'].append(ESS[-1])
                    self.q_fits[n_subject]['evidence'].append(evidence[-1])
                    self.q_fits[n_subject]['fit_within_current_EM'] = True
                    self.q_fits[n_subject]['evidence_mxs'].append(evidence_mxs)
                    self.q_fits[n_subject]['time_it'].append(time.time() - time_it)

                    
                    spec_q_trace = self.q_fits[n_subject]['spec_q']
                    P_trace     = self.q_fits[n_subject]['P']
                    P_aux_trace = self.q_fits[n_subject]['P_aux']
                    type_trace  = self.q_fits[n_subject]['type']
                                    
                    # Only keep the last 2:
                    while len(P_trace) >= 2:
                        P_trace.pop(0)
                        P_aux_trace.pop(0)
                        type_trace.pop(0)
                        
                    if not keep_q_spec:
                        while len(spec_q_trace) >= 2:
                            spec_q_trace.pop(0)
                        
                        
                    spec_q_trace.append(spec_q)
                    P_trace.append(fit['P'])
                    P_aux_trace.append(fit['P_aux'])
                    type_trace.append({'q':q_type, 'mean': mean_type, 'var': var_type, 'num_mixtures': num_mixtures, 'random_init': random_init})
                        
                    self.q_fits[n_subject]['spec_q'] = spec_q_trace
                    self.q_fits[n_subject]['P'] = P_trace
                    self.q_fits[n_subject]['P_aux'] = P_aux_trace
                    self.q_fits[n_subject]['type'] = type_trace   
            else:
                if old_q_fit is None:
                    old_q_fit = {'spec_q': [spec_q], 
                                              'type': [{'q':q_type, 'mean': mean_type, 'var': var_type, 'num_mixtures': num_mixtures, 'random_init': random_init}], 
                                              'S_its': [S], 
                                              'num_its' : 1,
                                             'ESS': [ESS[-1]],
                                             'evidence': [evidence[-1]],
                                             'P': [fit['P']],
                                             'P_aux': [fit['P_aux']],
                                             'fit_within_current_EM': True,
                                             'evidence_mxs': [evidence_mxs],
                                             'time_it': [time.time() - time_it]}

                else:
                    old_q_fit['S_its'].append(S)
                    old_q_fit['num_its'] += 1
                    old_q_fit['ESS'].append(ESS[-1])
                    old_q_fit['evidence'].append(evidence[-1])
                    old_q_fit['fit_within_current_EM'] = True
                    old_q_fit['evidence_mxs'].append(evidence_mxs)
                    old_q_fit['time_it'].append(time.time() - time_it)

                    
                    spec_q_trace = old_q_fit['spec_q']
                    P_trace     = old_q_fit['P']
                    P_aux_trace = old_q_fit['P_aux']
                    type_trace  = old_q_fit['type']
                                    
                    # Only keep the last 2:
                    while len(P_trace) >= 2:
                        P_trace.pop(0)
                        P_aux_trace.pop(0)
                        type_trace.pop(0)
                        
                    if not keep_q_spec:
                        while len(spec_q_trace) >= 2:
                            spec_q_trace.pop(0)
                        
                    spec_q_trace.append(spec_q)
                    P_trace.append(fit['P'])
                    P_aux_trace.append(fit['P_aux'])
                    type_trace.append({'q':q_type, 'mean': mean_type, 'var': var_type, 'num_mixtures': num_mixtures, 'random_init': random_init})
                        
                    old_q_fit['spec_q'] = spec_q_trace
                    old_q_fit['P'] = P_trace
                    old_q_fit['P_aux'] = P_aux_trace
                    old_q_fit['type'] = type_trace   
                
            if it > num_improvement_its_stop and it >= min_its - 1:
                if not multiple_CPU:
                    evidence_loaded = self.q_fits[n_subject]['evidence'][-(num_improvement_its_stop+1):]
                else:
                    evidence_loaded = old_q_fit['evidence'][-(num_improvement_its_stop+1):]
                if np.max(evidence_loaded[1:]) < evidence_loaded[0] + .5:
                    improvement_in_last_3 = False
                
            it += 1
            
        
        print(f'\r- Iteration {it-1:2d}, elapsed time: {time.time() - tm:5.2f}s, evidence estimate: {last_ev:.2f}.', end = '')
        
        if not multiple_CPU:
            self.plot_ESS_and_Evidence_AIS(n_subject, save_folder = save_folder, show_figs = display_end_diagnostics) 
            return None
        else:
            return old_q_fit
            
    
    # A helper function to transfer the q_fit of subject n to the normal fit.
    def transfer_AIS_to_fit(self, n):
        
        for e in ['evidence', 'ESS']:
            self.fit[n][e].append(self.q_fits[n][e][-1])
        for e in  ['evidence_f', 'evidence_nf']:
            self.fit[n][e].append(np.nan)
            
        self.fit[n]['P'] = self.q_fits[n]['P'][-1] 
        if not 'dummy' in self.q_fits[n]['P_aux'][-1].keys():         
            self.fit[n]['P_aux'] = self.q_fits[n]['P_aux'][-1]
            
    ### A helper function to display the ESS and Evidence after fitting:
    def plot_ESS_and_Evidence_AIS(self, n_subject, save_folder = None, show_figs = True):
        # Plot ESS per iteration, evidence estimate per iteration
        ESS = self.q_fits[n_subject]['ESS']
        evidence = self.q_fits[n_subject]['evidence']

        fig_ESS = go.Figure(go.Scatter(x= np.arange(len(ESS)),
                                      y = np.array(ESS),
                                      mode = 'lines'))
        fig_ESS.update_layout(title = f'{self.subject_names[n_subject]}) ESS per twisting iteration',
                             xaxis_title = 'Iteration #',
                             yaxis_title = 'ESS')



        fig_ev = go.Figure(go.Scatter(x= np.arange(len(evidence)),
                                      y = np.array(evidence),
                                      mode = 'lines'))
        fig_ev.update_layout(title = f'{self.subject_names[n_subject]}) Evidence per twisting iteration',
                             xaxis_title = 'Iteration #',
                             yaxis_title = 'Evidence')
        
        if self.llik_style in ['advanced_template', 'simple_template']:
            if self.llik_type == 'NF':
                norm_ct = self.subjects_data.num_nopractice_nf[n_subject]
            elif self.llik_type == 'All':
                norm_ct = self.subjects_data.num_nopractice_nf[n_subject] + self.subjects_data.num_nopractice_f[n_subject]
            elif self.llik_type == 'F':
                norm_ct = self.subjects_data.num_nopractice_f[n_subject]
            else:
                norm_ct = self.num_datapoints[n_subject]
        else:
            norm_ct = self.num_datapoints[n_subject]
                             
        fig_ev_norm = go.Figure(go.Scatter(x= np.arange(len(evidence)),
                                      y = np.exp(np.array(evidence) / norm_ct),
                                      mode = 'lines'))
        fig_ev_norm.update_layout(title = f'{self.subject_names[n_subject]}) exp(Evidence / number nopractice trials)',
                             xaxis_title = 'Iteration #',
                             yaxis_title = 'Evidence')
            
            
        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)

            model_folder = os.path.join(save_folder, self.name)
            if not os.path.exists(model_folder):
                os.mkdir(model_folder)
                
            fig_ESS.write_image(os.path.join(model_folder, f'{self.subject_names[n_subject]}_AIS_ESS.jpeg'))
            fig_ev.write_image(os.path.join(model_folder, f'{self.subject_names[n_subject]}_AIS_Evidence.jpeg'))
            fig_ev_norm.write_image(os.path.join(model_folder, f'{self.subject_names[n_subject]}_AIS_Evidence_norm.jpeg'))
            self.save(save_folder)
            
        if show_figs:
            fig_ESS.show()
            fig_ev.show()
            fig_ev_norm.show()
            
    def plot_ESS_and_Evidence(self, n_subject, save_folder = None, show_figs = True):
        # Plot ESS per iteration, evidence estimate per iteration
        ESS = self.fit[n_subject]['ESS']
        evidence = self.fit[n_subject]['evidence']

        fig_ESS = go.Figure(go.Scatter(x= np.arange(len(ESS)),
                                      y = np.array(ESS),
                                      mode = 'lines'))
        fig_ESS.update_layout(title = f'{self.subject_names[n_subject]}) ESS per twisting iteration',
                             xaxis_title = 'Iteration #',
                             yaxis_title = 'ESS')



        fig_ev = go.Figure(go.Scatter(x= np.arange(len(evidence)),
                                      y = np.array(evidence),
                                      mode = 'lines'))
        fig_ev.update_layout(title = f'{self.subject_names[n_subject]}) Evidence per twisting iteration',
                             xaxis_title = 'Iteration #',
                             yaxis_title = 'Evidence')
                             
        if self.llik_style in ['advanced_template', 'simple_template']:
            if self.llik_type == 'NF':
                norm_ct = self.subjects_data.num_nopractice_nf[n_subject]
            elif self.llik_type == 'All':
                norm_ct = self.subjects_data.num_nopractice_nf[n_subject] + self.subjects_data.num_nopractice_f[n_subject]
            elif self.llik_type == 'F':
                norm_ct = self.subjects_data.num_nopractice_f[n_subject]
            else:
                norm_ct = self.num_datapoints[n_subject]
        else:
            norm_ct = self.num_datapoints[n_subject]
                             
        fig_ev_norm = go.Figure(go.Scatter(x= np.arange(len(evidence)),
                                      y = np.exp(np.array(evidence) / norm_ct),
                                      mode = 'lines'))
        fig_ev_norm.update_layout(title = f'{self.subject_names[n_subject]}) exp(Evidence / number nopractice trials)',
                             xaxis_title = 'Iteration #',
                             yaxis_title = 'Evidence')
            
            
        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)

            model_folder = os.path.join(save_folder, self.name)
            if not os.path.exists(model_folder):
                os.mkdir(model_folder)
                
            fig_ESS.write_image(os.path.join(model_folder, f'{self.subject_names[n_subject]}_ESS.jpeg'))
            fig_ev.write_image(os.path.join(model_folder, f'{self.subject_names[n_subject]}_Evidence.jpeg'))
            fig_ev_norm.write_image(os.path.join(model_folder, f'{self.subject_names[n_subject]}_Evidence_norm.jpeg'))
            self.save(save_folder)
            
        if show_figs:
            fig_ESS.show()
            fig_ev.show()
            fig_ev_norm.show()


    ### Function to save fit results:
    def save(self, models_folder, name_spec = ""):
        """
        Saves the 'style', 'spec', 'fit' and 'q_fit' objects to a hdf5 file.
        
        The model is saved as '{models_folder}/"name""name_spec".h5' where name is self.name, and spec is supplied in the kwargs.
        """
        
        if not os.path.exists(models_folder):
            os.mkdir(models_folder)
        
        filename = os.path.join(models_folder, f"{self.name}{name_spec}.h5")
        with h5py.File(filename, 'w') as hf:
            
            hf.attrs['name'] = self.name

            hf.attrs['llik_style'] = self.llik_style
            if self.llik_style in ['advanced_template', 'simple_template']:
                hf.attrs['Q_style'] = self.Q_style
                hf.attrs['choice_style'] = self.choice_style
                hf.attrs['R_style'] = self.R_style
                if self.T_style is not None:
                    hf.attrs['T_style'] = self.T_style
                if self.Q_style_2 is not None:
                    hf.attrs['Q_style_2'] = self.Q_style_2
                if self.T_style_2 is not None:
                    hf.attrs['T_style_2'] = self.T_style_2
                hf.attrs['llik_type'] = self.llik_type                  

            hf.create_dataset('bic', data = self.bic)
            hf.create_dataset('evidence', data = self.evidence)
            hf.create_dataset('evidence_f', data = self.evidence_f)
            hf.create_dataset('evidence_nf', data = self.evidence_nf)
            hf.attrs['fit_its'] = self.fit_its
            hf.attrs['I'] = self.I
            hf.create_dataset('ESS_mean', data = self.ESS_mean)
            hf.attrs['runtime_it'] = self.runtime_it
            hf.attrs['ev_jump'] = self.ev_jump
            
            hf.attrs['num_nopractice_f']  = self.num_nopractice_f
            hf.attrs['num_nopractice_nf'] = self.num_nopractice_nf
            hf.attrs['num_datapoints']    = self.num_datapoints
            
            hf.attrs['q_fitted_EM_it'] = self.q_fitted_EM_it

            # Save spec:
            hf_spec = hf.create_group('spec')
            for par_name in self.spec.keys():
                gp = hf_spec.create_group(par_name)
                gp.create_dataset('val', data = self.spec[par_name]['val'])
                gp.attrs['type'] = self.spec[par_name]['type']
                if 'fixed_val' in self.spec[par_name].keys():
                    gp.attrs['fixed_val'] = self.spec[par_name]['fixed_val']
                    #gp.create_dataset('fixed_val', data = self.spec[par_name]['fixed_val'])
                    
            # Save spec trace:
            hf_spec_trace = hf.create_group('spec_trace')
            for l in range(len(self.spec_trace)):
                gp_l = hf_spec_trace.create_group(str(l))
                for par_name in self.spec.keys():
                    gp = gp_l.create_group(par_name)
                    gp.create_dataset('val', data = self.spec[par_name]['val'])
                    gp.attrs['type'] = self.spec[par_name]['type']
                    if 'fixed_val' in self.spec[par_name].keys():
                        gp.attrs['fixed_val'] = self.spec[par_name]['fixed_val']               

            # Save fits:
            fits = hf.create_group('fits')
            for n, s_name in enumerate(self.subject_names):
                gp = fits.create_group(s_name)
                #gp.attrs['ESS_it'] = self.ESS_it[n]
                gp.attrs['color'] = self.colors[n]
                gp.create_dataset('evidence', data = self.fit[n]['evidence'])
                gp.create_dataset('evidence_f', data = self.fit[n]['evidence_f'])
                gp.create_dataset('evidence_nf', data = self.fit[n]['evidence_nf'])
                gp.attrs['N_samples'] = self.fit[n]['N_samples']
                gp.create_dataset('ESS', data = self.fit[n]['ESS'])
                P = self.fit[n]['P']
                pgp = gp.create_group('P')
                for par_name in P:
                    ppgp = pgp.create_group(par_name)
                    ppgp.attrs['val'] = P[par_name]['val']
                    ppgp.attrs['mean'] = P[par_name]['mean']
                    ppgp.attrs['median'] = P[par_name]['median']
                    ppgp.attrs['max'] = P[par_name]['max']
                    ppgp.create_dataset('ci', data = P[par_name]['ci'])
                    ppgp.create_dataset('samp', data = P[par_name]['samp'])
                    
                # Optionally, save auxiliary fits:
                if 'P_aux' in self.fit[n].keys() and not 'dummy' in self.fit[n]['P_aux'].keys():
                    pgp = gp.create_group('P_aux')
                    P_aux = self.fit[n]['P_aux']
                    for par_name in P_aux:
                        ppgp = pgp.create_group(par_name)
                        ppgp.attrs['val'] = P_aux[par_name]['val']
                        ppgp.attrs['mean'] = P_aux[par_name]['mean']
                        ppgp.attrs['median'] = P_aux[par_name]['median']
                        ppgp.attrs['max'] = P_aux[par_name]['max']
                        ppgp.create_dataset('ci', data = P_aux[par_name]['ci'])                  
                        
            # save q-fits:
            q = hf.create_group('q_fits')
            for n in range(self.N):
                if self.q_fits[n] is not None:
                    gp = q.create_group(str(n))
                    gp.attrs['num_its'] = self.q_fits[n]['num_its']
                    gp.attrs['fit_within_current_EM'] = self.q_fits[n]['fit_within_current_EM']
                    gp.create_dataset('S_its', data = self.q_fits[n]['S_its'])
                    gp.create_dataset('ESS', data = self.q_fits[n]['ESS'])
                    gp.create_dataset('evidence', data = self.q_fits[n]['evidence'])
                    if 'max_day' in self.q_fits[n].keys():
                        gp.create_dataset('max_day', data = self.q_fits[n]['max_day'])
                    gp.attrs['time_it'] = self.q_fits[n]['time_it']
                    
                    AIS_type = gp.create_group('type')
                    for it in range(len(self.q_fits[n]['type'])):
                        AIS_type_it = AIS_type.create_group(str(it))
                        for AIS_spec in ['q', 'mean', 'var', 'num_mixtures', 'random_init']:
                            AIS_type_it.attrs[AIS_spec] = self.q_fits[n]['type'][it][AIS_spec]
                    
                    spec_q = gp.create_group('spec_q')
                    for it in range(len(self.q_fits[n]['spec_q'])):
                        spec_q_it = spec_q.create_group(str(it))
                        for mx in range(len(self.q_fits[n]['spec_q'][it])):
                            spec_q_mx = spec_q_it.create_group(str(mx))
                            spec_q_data = self.q_fits[n]['spec_q'][it][mx]
                            if not 'var_list' in spec_q_data:
                                for par_name in spec_q_data:
                                    ppgp = spec_q_mx.create_group(par_name)
                                    ppgp.attrs['type'] = spec_q_data[par_name]['type']
                                    ppgp.create_dataset('val', data = spec_q_data[par_name]['val'])
                                    if 'fixed_val' in spec_q_data[par_name].keys():
                                        ppgp.attrs['fixed_val'] = spec_q_data[par_name]['fixed_val']
                            else:
                                spec_q_mx.attrs['var_list'] = spec_q_data['var_list']
                                spec_q_mx.attrs['ct_vars'] = spec_q_data['ct_vars']
                                spec_q_mx.attrs['fit_vars'] = spec_q_data['fit_vars']
                                spec_q_mx.attrs['tf'] = spec_q_data['tf']
                                spec_q_mx.attrs['df'] = spec_q_data['df']
                                spec_q_mx.create_dataset('mean', data = spec_q_data['mean'])
                                spec_q_mx.create_dataset('covar', data = spec_q_data['covar'])
                    
                        
                    P_fit = gp.create_group('P')
                    for it in range(len(self.q_fits[n]['P'])):
                        P_fit_it = P_fit.create_group(str(it))
                        P_fit_data = self.q_fits[n]['P'][it]
                        for par_name in P_fit_data:
                            ppgp = P_fit_it.create_group(par_name)
                            ppgp.attrs['val'] = P_fit_data[par_name]['val']
                            if 'mean' in P_fit_data[par_name].keys():
                                ppgp.attrs['mean'] = P_fit_data[par_name]['mean']
                                ppgp.attrs['median'] = P_fit_data[par_name]['median']
                                ppgp.attrs['max'] = P_fit_data[par_name]['max']
                            else:
                                ppgp.attrs['mean'] = np.nan
                                ppgp.attrs['median'] = np.nan
                                ppgp.attrs['max'] = [np.nan for j in range(5)]
                            ppgp.create_dataset('ci', data = P_fit_data[par_name]['ci'])
                            ppgp.create_dataset('samp', data = P_fit_data[par_name]['samp'])
                    
                    P_aux_fit = gp.create_group('P_aux')
                    for it in range(len(self.q_fits[n]['P_aux'])):
                        P_aux_fit_data = self.q_fits[n]['P_aux'][it]
                        P_aux_fit_it = P_aux_fit.create_group(str(it))
                        for par_name in P_aux_fit_data:
                            ppgp = P_aux_fit_it.create_group(par_name)
                            ppgp.attrs['val'] = P_aux_fit_data[par_name]['val']
                            if 'mean' in P_aux_fit_data[par_name].keys():
                                ppgp.attrs['mean'] = P_aux_fit_data[par_name]['mean']
                                ppgp.attrs['median'] = P_aux_fit_data[par_name]['median']
                                ppgp.attrs['max'] = P_aux_fit_data[par_name]['max']
                            else:
                                ppgp.attrs['mean'] = np.nan
                                ppgp.attrs['median'] = np.nan
                                ppgp.attrs['max'] = [np.nan for j in range(5)]
                            ppgp.create_dataset('ci', data = P_aux_fit_data[par_name]['ci'])
                            
    def check_saved_model_subjects(self, filename):
        """
        For a saved model, checks what the fit subjects are.
        """
        
        with h5py.File(filename, 'r') as hf:
            gp = hf['fits']
            
            subject_names = list(gp.keys())
                       
        return subject_names
                
    
    ### Function to load saved results:    
    def load(self, filename, s_dat, llik_custom = None, old_version = False, verbose = True):
        """
        Opens a saved model (using model.save()), given the filename (of the '.h5' format). 
        """
        
        self.input_subjects(s_dat)
        
        with h5py.File(filename, 'r') as hf:
        
            self.name = hf.attrs['name']
            
            if 'llik_style' in hf.attrs:
                self.llik_style = hf.attrs['llik_style']
            else:
                self.llik_style = 'advanced_template'
                
            if self.llik_style in ['advanced_template', 'simple_template']:            
                self.Q_style = hf.attrs['Q_style']
                self.choice_style = hf.attrs['choice_style']
                self.R_style = hf.attrs['R_style'] 
                if 'llik_type' in hf.attrs:
                    self.llik_type = hf.attrs['llik_type']
                    self.style = {'Q_style'     : self.Q_style,
                                  'choice_style': self.choice_style,
                                  'R_style'     : self.R_style,
                                  'llik_style'  : self.llik_style}
                else:
                    self.llik_type = 'NF'
                    self.style = {'Q_style'     : self.Q_style,
                                  'choice_style': self.choice_style,
                                  'R_style'     : self.R_style,
                                  'llik_style'  : self.llik_style}
                    
                if 'T_style' in hf.attrs:
                    self.T_style = hf.attrs['T_style']
                    self.style['T_style'] = self.T_style
                else:
                    self.T_style = None
                if 'Q_style_2' in hf.attrs:
                    self.Q_style_2 = hf.attrs['Q_style_2']
                    self.style['Q_style_2'] = self.Q_style_2
                else:
                    self.Q_style_2 = None    
                if 'T_style_2' in hf.attrs:
                    self.T_style_2 = hf.attrs['T_style_2']
                    self.style['T_style_2'] = self.T_style_2
                else:
                    self.T_style_2 = None
            elif self.llik_style == 'custom':
                if llik_custom is None:
                    sys.exit("Load > Provide llik_custom = ")
                else:
                    self.llik = llik_custom
                    
                # if self.llik_style == 'advanced_template':
                    # self.llik = create_llik_adv(self, s_dat, self.style)       
                # else:
                    # 1 + 1 # Nada

            self.bic = list(hf.get('bic'))
            self.evidence    = list(hf.get('evidence'))
            self.evidence_f  = list(hf.get('evidence_f'))
            self.evidence_nf = list(hf.get('evidence_nf'))

            self.fit_its = hf.attrs['fit_its']
            self.I       = hf.attrs['I']
            if 'q_fitted_EM_it' in hf.attrs:
                self.q_fitted_EM_it = list(hf.attrs['q_fitted_EM_it'])
            else:
                self.q_fitted_EM_it = []
            
            self.ESS_mean = list(hf.get('ESS_mean'))
            self.llik_it = np.array([np.nan])
            
            if 'runtime_it' in hf.attrs:
                self.runtime_it = list(hf.attrs['runtime_it'])
            else:
                self.runtime_it = [np.nan for bic in self.bic]
                
            if 'ev_jump' in hf.attrs:
                self.ev_jump = hf.attrs['ev_jump']
            else:
                self.ev_jump = np.nan
            
            if self.llik_style in ['advanced_template', 'simple_template']:
                self.num_nopractice_f  = s_dat.num_nopractice_f
                self.num_nopractice_nf = s_dat.num_nopractice_nf
                if self.llik_type == 'All':
                    self.num_datapoints    = s_dat.num_datapoints # number of datapoints per subject
                elif self.llik_type == 'NF': 
                    self.num_datapoints    = s_dat.num_nopractice_nf
                elif self.llik_type == 'F':
                    self.num_datapoints    = s_dat.num_nopractice_f
            else:
                self.num_nopractice_f  = s_dat.num_nopractice_f
                self.num_nopractice_nf = s_dat.num_nopractice_nf
                self.num_datapoints = s_dat.num_datapoints
                
            if 'num_datapoints' in hf.attrs:
                self.num_nopractice_f_old  = hf.attrs['num_nopractice_f']
                self.num_nopractice_nf_old = hf.attrs['num_nopractice_nf']
                self.num_datapoints_old    = hf.attrs['num_datapoints']
            
            # Load spec:
            self.spec = {}
            for par_name in hf['spec'].keys():
                gp = hf['spec'][par_name]
                val = gp.get('val')
                self.spec[par_name] = {'type': gp.attrs['type'],
                                      'val': list(np.array(val))}
                if 'fixed_val' in gp.attrs:
                    self.spec[par_name]['fixed_val'] = gp.attrs['fixed_val']
                    
            # Load spec trace:
            self.spec_trace = []
            if 'spec_trace' in hf:
                for l in range(len(hf['spec_trace'])):
                    gp_l = hf['spec_trace'][str(l)]
                    spec = {}
                    for par_name in gp_l.keys():
                        gp = gp_l[par_name]
                        val = gp.get('val')
                        spec[par_name] = {'type': gp.attrs['type'],
                                              'val': list(np.array(val))}
                        if 'fixed_val' in gp.attrs:
                            spec[par_name]['fixed_val'] = gp.attrs['fixed_val']
                    self.spec_trace.append(spec)            
                
            # Load fits:
            #self.subject_names = []; self.colors = []
            self.fit = []; self.fit_aux = None
            #self.N = 0
            self.ESS = []
            self.ESS_it = []
            for n, subject_name in enumerate(self.subject_names):
                #self.N += 1
                #self.subject_names.append(subject_name)
                
                gp = hf[f'fits/{subject_name}']
                #self.colors.append(gp.attrs['color'])
                self.fit.append({'evidence': list(gp.get('evidence')),
                                 'evidence_f': list(gp.get('evidence_f')),
                                 'evidence_nf': list(gp.get('evidence_nf')),
                                 'ESS': list(gp.get('ESS')),
                                 'N_samples': gp.attrs['N_samples'],
                                 'P': {},
                                'P_aux': {}})
                if 'ESS' in gp.attrs:
                    self.ESS.append(gp.attrs['ESS'])
                else:
                    self.ESS.append(np.nan)
                if 'ESS_it' in gp.attrs:
                    self.ESS_it.append(gp.attrs['ESS_it'])
                else:
                    self.ESS_it.append(np.nan)
                pgp = gp['P']
                for par_name in pgp.keys():
                    ppgp = pgp[par_name]
                    if 'mean' in ppgp.attrs:
                        self.fit[-1]['P'][par_name] = {
                            'val': ppgp.attrs['val'],
                            'mean': ppgp.attrs['mean'],
                            'median': ppgp.attrs['median'],
                            'max': ppgp.attrs['max'],
                            'ci': np.array(ppgp.get('ci')),
                            'samp': np.array(ppgp.get('samp'))
                        }
                    else:
                        self.fit[-1]['P'][par_name] = {
                        'val': ppgp.attrs['val'],
                        'mean': np.nan,
                        'median': np.nan,
                        'max': [np.nan for j in range(5)],
                        'ci': np.array(ppgp.get('ci')),
                        'samp': np.array(ppgp.get('samp'))
                    }
                if 'P_aux' in gp:
                    pgp = gp['P_aux']
                    for par_name in pgp.keys():
                        ppgp = pgp[par_name]
                        if 'mean' in ppgp.attrs:
                            self.fit[-1]['P_aux'][par_name] = {'val': ppgp.attrs['val'],
                            'mean': ppgp.attrs['mean'],
                            'median': ppgp.attrs['median'],
                            'max': ppgp.attrs['max'],
                            'ci': np.array(ppgp.get('ci'))
                            }
                        else:
                            self.fit[-1]['P_aux'][par_name] = {
                                'val': ppgp.attrs['val'],
                                'mean': np.nan,
                                'median': np.nan,
                                'max': [np.nan for j in range(5)],
                                'ci': np.array(ppgp.get('ci'))
                            }       
                    
            # Load q fits:
            self.q_fits = [None for n in range(self.N)]
            if 'q_fits' in hf:
                for n_str in hf['q_fits'].keys():
                    n = int(n_str)
                    gp = hf[f'q_fits/{n}']

                    fit = {}               
                    fit['num_its'] = gp.attrs['num_its']
                    
                    fit['S_its'] = list(gp.get('S_its'))
                    fit['ESS'] = list(gp.get('ESS'))
                    fit['evidence'] = list(gp.get('evidence'))
                    if 'fit_within_current_EM' in gp.attrs:
                        fit['fit_within_current_EM'] = gp.attrs['fit_within_current_EM']
                    else:
                        fit['fit_within_current_EM'] = False
                    if 'max_day' in gp.keys():
                        fit['max_day'] = list(gp.get('max_day'))
                    fit['evidence_mxs'] = [] # TODO: save and load
                    if 'time_it' in gp.attrs:
                        fit['time_it'] = list(gp.attrs['time_it'])
                    else:
                        fit['time_it'] = []                
                        
                    if old_version:
                        fit['q_type'] = gp.attrs['q_type'][0]
                    else:
                        fit['type'] = []
                        num_saved = min(fit['num_its'], 2)
                        for it in range(num_saved):
                            AIS_type_it = gp['type'][str(it)]
                            data_it = {}
                            for AIS_spec in ['q', 'mean', 'var', 'num_mixtures', 'random_init']:
                                data_it[AIS_spec] = AIS_type_it.attrs[AIS_spec]
                            fit['type'].append(data_it)                          

                    if old_version:
                        fit['spec_q'] = [{} for mx in gp['spec_q'].keys()]
                        for mx in gp['spec_q'].keys():
                            spec_q_mx = gp['spec_q'][mx]
                            for par_name in gp['spec_q'].keys():
                                fit['spec_q'][int(mx)][par_name] = {}
                                ppgp = spec_q_mix[par_name]
                                fit['spec_q'][int(mx)][par_name]['type'] = ppgp.attrs['type']
                                fit['spec_q'][int(mx)][par_name]['val']  = list(ppgp.get('val'))

                        fit['P'] = {}
                        for par_name in gp['P'].keys():
                            fit['P'][par_name] = {}
                            ppgp = gp[f'P/{par_name}']
                            fit['P'][par_name]['val'] = ppgp.attrs['val']
                            fit['P'][par_name]['ci']  = np.array(ppgp.get('ci'))
                            fit['P'][par_name]['samp']  = np.array(ppgp.get('samp'))

                        fit['P_aux'] = {}
                        for par_name in gp['P_aux'].keys():
                            fit['P_aux'][par_name] = {}
                            ppgp = gp[f'P_aux/{par_name}']
                            fit['P_aux'][par_name]['val'] = ppgp.attrs['val']
                            fit['P_aux'][par_name]['ci']  = np.array(ppgp.get('ci')) 
                    else:
                        num_saved = min(fit['num_its'], 2)
                        
                        fit['spec_q'] = []
                        num_specs = len(gp['spec_q'])
                        for it in range(num_specs):
                            spec_it = [{} for mx in gp['spec_q'][str(it)].keys()]
                            for mx in gp['spec_q'][str(it)].keys():
                                spec_q_mx = gp['spec_q'][str(it)][mx]
                                if not 'var_list' in spec_q_mx.attrs:
                                    for par_name in spec_q_mx.keys():
                                        spec_it[int(mx)][par_name] = {}
                                        ppgp = spec_q_mx[par_name]
                                        spec_it[int(mx)][par_name]['type'] = ppgp.attrs['type']
                                        spec_it[int(mx)][par_name]['val']  = list(ppgp.get('val'))
                                        if 'fixed_val' in ppgp.attrs:
                                            spec_it[int(mx)][par_name]['fixed_val'] = ppgp.attrs['fixed_val']
                                else:
                                    spec_it[int(mx)]['var_list'] = list(spec_q_mx.attrs['var_list'])
                                    spec_it[int(mx)]['ct_vars'] = list(spec_q_mx.attrs['ct_vars']) 
                                    spec_it[int(mx)]['fit_vars'] = list(spec_q_mx.attrs['fit_vars']) 
                                    spec_it[int(mx)]['tf'] = list(spec_q_mx.attrs['tf'])
                                    spec_it[int(mx)]['df'] = spec_q_mx.attrs['df']
                                    spec_it[int(mx)]['mean'] = np.array(spec_q_mx.get('mean'))
                                    spec_it[int(mx)]['covar'] = np.array(spec_q_mx.get('covar'))
                            fit['spec_q'].append(spec_it)

                        fit['P'] = []
                        for it in range(num_saved):
                            P_it = {}
                            for par_name in gp['P'][str(it)].keys():
                                P_it[par_name] = {}
                                ppgp = gp['P'][str(it)][par_name]
                                P_it[par_name]['val'] = ppgp.attrs['val']
                                if 'mean' in ppgp.attrs:
                                    P_it[par_name]['mean'] = ppgp.attrs['mean']
                                    P_it[par_name]['median'] = ppgp.attrs['median']
                                    P_it[par_name]['max'] = ppgp.attrs['max']
                                else:
                                    P_it[par_name]['mean'] = np.nan
                                    P_it[par_name]['median'] = np.nan
                                    P_it[par_name]['max'] = [np.nan for j in range(5)]
                                P_it[par_name]['ci']  = np.array(ppgp.get('ci'))
                                P_it[par_name]['samp']  = np.array(ppgp.get('samp'))
                            fit['P'].append(P_it)

                        fit['P_aux'] = []
                        for it in range(num_saved):
                            P_aux_it = {}
                            for par_name in gp['P_aux'][str(it)].keys():
                                P_aux_it[par_name] = {}
                                ppgp = gp['P_aux'][str(it)][par_name]
                                P_aux_it[par_name]['val'] = ppgp.attrs['val']
                                if 'mean' in ppgp.attrs:
                                    P_aux_it[par_name]['mean'] = ppgp.attrs['mean']
                                    P_aux_it[par_name]['median'] = ppgp.attrs['median']
                                    P_aux_it[par_name]['max'] = ppgp.attrs['max']
                                else:
                                    P_aux_it[par_name]['mean'] = np.nan
                                    P_aux_it[par_name]['median'] = np.nan
                                    P_aux_it[par_name]['max'] = [np.nan for j in range(5)]
                                P_aux_it[par_name]['ci']  = np.array(ppgp.get('ci')) 
                            fit['P_aux'].append(P_aux_it)
                            

                    self.q_fits[n] = fit
                    
            if 'IBIS_fit_times' in hf.attrs:
                self.IBIS_fit_times = hf.attrs['IBIS_fit_times']
            else:
                self.IBIS_fit_times = [np.nan for n in range(self.N)]

                        
        self.P = None
        self.P_aux = None
                
        if verbose:            
            print(f"- Loaded '{self.name}'.")
        
        return self
               
    def extract_param_values(self, with_names = False, subject_names = None, order_names = False, only_PM = True, mode = 'mean', use_AIS_fits = False):
        """
        
        Arguments:
        - only_PM: True or False, if True returns fitted parameters with confidence intervals. If False, returns the whole fit object.
        - mode: String, the type of fitted parameters to return. Posterior means if 'mean', Posterior Median if 'median', Posterior Maximum if 'max'.

        Returns a Pandas DataFrame with the posterior mean estimates of the subjects paramters:
        (N, num_par) - dataframe
        """

        subject_names, columns = self.subjects_data.subject_names_to_columns(subject_names, order_names)

        if only_PM:
            par_vals = []
            par_ci   = []
        
            for k,n in enumerate(columns):
                par_n = {}
                ci_n  = {}
                
                fits = self.fit[n]['P']
                
                for par_name in fits.keys():
                    if mode == 'mean':
                        par_n[par_name] = fits[par_name]['val']
                    elif mode == 'median':
                        par_n[par_name] = fits[par_name]['median']
                    elif mode == 'max':
                        par_n[par_name] = fits[par_name]['max'][0]
                    ci_n[par_name]  = fits[par_name]['ci']
                               
                par_vals.append(pd.DataFrame(par_n, index = [subject_names[k]]))
                par_ci.append(ci_n)

            par_vals = pd.concat(par_vals, axis = 0)
            if with_names:
                par_vals['subject'] = subject_names
            
            # Determine whether to extract aux fits:
            no_aux_fits = False
            if 'P_aux' not in self.fit[0].keys():
                no_aux_fits = True
            elif len(self.fit[0]['P_aux'].keys()) == 0:
                no_aux_fits = True

            if no_aux_fits:
                return (par_vals, None, par_ci, None)
            else:
                P_aux_vals = []
                P_aux_ci = []
                
                for k,n in enumerate(columns):
                    par_n = {}
                    ci_n  = {}
                    
                    if use_AIS_fits:
                        fits = self.q_fits[n]['P_aux'][AIS_fit_idxs[k]]
                    else:
                        fits = self.fit[n]['P_aux']
                    
                    for par_name in fits.keys():
                        if mode == 'mean':
                            par_n[par_name] = fits[par_name]['val']
                        elif mode == 'median':
                            par_n[par_name] = fits[par_name]['median']
                        elif mode == 'max':
                            par_n[par_name] = fits[par_name]['max'][0]
                        ci_n[par_name]  = fits[par_name]['ci']
                    
                    P_aux_vals.append(pd.DataFrame(par_n, index = [subject_names[k]]))
                    P_aux_ci.append(ci_n)
                
                P_aux_vals = pd.concat(P_aux_vals, axis = 0)                
                if with_names:
                    P_aux_vals['subject'] = subject_names

                return (par_vals, P_aux_vals, par_ci, P_aux_ci)

        else:
            if use_AIS_fits:
                q_fits = []
                for k, n in enumerate(columns):
                    q_fits_n = {}
                    q_fits_n['P'] = self.q_fits[n]['P'][AIS_fit_idxs[k]]
                    q_fits_n['P_aux'] = self.q_fits[n]['P_aux'][AIS_fit_idxs[k]]
                    q_fits_n['subject_name'] = subject_names[k]
                    q_fits_n['evidence'] = self.q_fits[n]['evidence']
                    q_fits.append(q_fits_n)

                return q_fits
            else:
                fits = []
                for k,n in enumerate(columns):
                    fits.append(self.fit[n])
                return fits

    ### A function that creates subjects data consisting of the real data and a simulated trajectory with the Posterior Mean of each subject: ###
    ### Currently does not work with two process models or T_styles:
    def create_posterior_subjects(self, copy_function = None):
        """
        Returns a 'subjects' objects with simulated data using the Posterior Mean estimates of the model.
        
        The simulated subjects have names 'subject_name'_fit.
        
        Returns a subjects object with N' = 2*N.
        """
        s_dat = self.subjects_data
        P, P_aux, _, _ = self.extract_param_values()
        
        if copy_function is None:
            copy_s_dat = False
            if 'subjects_EMA' in globals():
                if type(s_dat) == subjects_EMA:
                    copy_s_dat = True
            if copy_s_dat:
                s_dat_n = subjects_EMA().copy_subject_states(s_dat)
            else:
                s_dat_n = s_dat
        else:
            s_dat_n = copy_function(s_dat)

        tm = time.time()
        for n in range(s_dat.N):
            
            P_n = P.iloc[[n],:]
            if P_aux is None:
                P_aux_n = None
            else:
                P_aux_n = P_aux.iloc[[n],:]
                
            if self.llik_style == 'advanced_template':
                llik_adv(P_n, P_aux_n, n, s_dat_n, self.style, simulate_data = True)
            elif self.llik_style == 'simple_template':
                llik_simple(P_n, P_aux_n, n, s_dat_n, self.style, simulate_data = True)
            elif self.llik_style == 'custom':
                self.llik(P_n, P_aux_n, n, s_dat_n, simulate_data = True)
            else:
                sys.exit("generate_PM_datasets > Not a valid llik_style.")
        print(f"Time to simulate {s_dat_n.N} subjects over {s_dat_n.T} timesteps:\n==> {time.time() - tm:.3f}s.")  
                
        s_dat_n.P_true = P
        s_dat_n.P_aux_true = P_aux
        
        s_dat_n.subject_names = [f'{s_name} (sim)' for s_name in s_dat_n.subject_names]

        return s_dat_n        
        
    ### Plot estimates vs true parameters: ###
    def plot_true_vs_est(self, save_folder = None, use_AIS_fits = False, AIS_best_idx = -1, subject_names = None, order_names = False):
        """
        Displays subject parameter estimates vs their true value that was used to generate their trajectories, 
        in a scatter plot with 95% credibility intervals.
        """
        
        subject_names, columns = self.subjects_data.subject_names_to_columns(subject_names, order_names)
        
        spec = self.spec
        s_dat = self.subjects_data
        if s_dat.P_true is None:
            sys.exit("No true parameters found!")
        P_true = s_dat.P_true
        
        for par_name in spec.keys():
            true_values = P_true[par_name].values[columns]
            if use_AIS_fits:
                ests = np.array([self.q_fits[n]['P'][AIS_best_idx][par_name]['val'] for n in columns])
                ci = np.array([self.q_fits[n]['P'][AIS_best_idx][par_name]['ci'] for n in columns])
            else:
                ests = np.array([self.fit[n]['P'][par_name]['val'] for n in columns])
                ci = np.array([self.fit[n]['P'][par_name]['ci'] for n in columns])
            
            # Create the first diagonal, x = y, to display on the figure:
            minx = np.min(ci); maxx = np.max(ci)
            cover_x = np.linspace(minx, maxx)
            
            # Create a Linear Regression line
            lm = LinearRegression()
            lm.fit(true_values.reshape(-1,1), ests)
            cover_y = lm.predict(cover_x.reshape(-1, 1))     
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                    x=true_values,
                    y=ests,
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array= ci[:, 1] - ests,
                        arrayminus= ests - ci[:,0],
                        thickness = 1.),
                    mode = 'markers',
                    name = 'Estimates'
                    ))
            fig.add_trace(go.Scatter(
                x = cover_x,
                y = cover_x,
                mode = 'lines',
                line = dict(color='red'),
                name = 'x = y'
            ))
            fig.add_trace(go.Scatter(
                x = cover_x,
                y = cover_y,
                mode = 'lines',
                line = dict(color='blue'),
                name = 'Linear Regression'
            ))
            fig.update_layout(
                    title = self.name +": "+par_name,
                xaxis_title = 'True Values',
                yaxis_title = 'Estimates')
            
            fig.show()
            
            if save_folder is not None:
                results_folder = os.path.join(save_folder, self.name)
                if not os.path.exists(results_folder):
                    os.mkdir(results_folder)
                fig.write_image(os.path.join(results_folder, f"consistency_{par_name}.jpeg"))
    
    ### A helper function that prints the credibility interval lengths for each parameter:
    def ci_length(self):
        
        spec = self.spec
        P_true = self.subjects_data.P_true
        for par_name in spec.keys():
            true_values = P_true[par_name].values
            ests = np.array([self.fit[n]['P'][par_name]['val'] for n in range(self.N)])
            ci = np.array([self.fit[n]['P'][par_name]['ci'] for n in range(self.N)])
            
            print(f"{par_name:10s} - mean(ci length) = {np.mean(ci[:,1] - ci[:,0])}\n             std(ci length) = {np.std(ci[:,1] - ci[:,0])}")
    
    ### A funtcion that displays the parameter fits per subject:
    def plot_est(self, save_folder = None, show_figs=True, use_AIS_fits = False):
        """
        Plots the posterior means with their credibility interval, for all subjects.
        
        - save_figs: bool, if True, saves the figures in "model_ests\\{self.name}\\{par_name}.jpeg"
        
        May be deprecated. Use repfit_summary_fixed_data instead.
        """
        
        spec = self.spec
        N = self.N
        
        for par_name in self.fit[0]['P'].keys():            
            if use_AIS_fits:
                q_fit_subjects = []
                for n in range(self.N):
                    if self.q_fits is not None:
                        q_fit_subjects.append(n)
                ests = np.array([self.q_fits[n]['P'][par_name]['val'] for n in q_fit_subjects])
                ci = np.array([self.q_fits[n]['P'][par_name]['ci'] for n in q_fit_subjects])
            else:
                q_fit_subjects = list(range(self.N))
                ests = np.array([self.fit[n]['P'][par_name]['val'] for n in range(self.N)])
                ci = np.array([self.fit[n]['P'][par_name]['ci'] for n in range(self.N)])

            xticks = np.arange(len(q_fit_subjects))
            
            fig = go.Figure()
            for n in q_fit_subjects:
                fig.add_trace(go.Scatter(
                        x=[xticks[n]],
                        y=[ests[n]],
                        error_y=dict(
                            type='data',
                            symmetric=False,
                            array= [ci[n, 1] - ests[n]],
                            arrayminus= [ests[n] - ci[n,0]],
                            thickness = 1.),
                        mode = 'markers',
                        name = str(self.subject_names[n]),
                        line = {'color': self.colors[n]}
                        ))
                fig.update_layout(
                    title = self.name +": "+par_name,
                    xaxis_title = 'Subjects',
                    yaxis_title = 'Estimates')
            if show_figs:
                fig.show()
            
            if save_folder is not None:
                if not os.path.exists(save_folder):
                    os.mkdir(save_folder)
                results_folder = os.path.join(save_folder, self.name)
                if not os.path.exists(results_folder):
                    os.mkdir(results_folder)
                save_path = os.path.join(save_folder, self.name, f"{par_name}.jpeg")
                fig.write_image(save_path)
    
    ### Display daily-varying parameters and correlate them with correlation variables (such as moods):    
    def plot_est_time_varying(self, par_names = [], P_ref = None, save_folder = None, show_figs = True, use_AIS_fits = False, AIS_best_idxs = None):
        """
        For models that use day varying parameters, plots the parameter estimates as a time series
        
        - par_names: list of strongs, base names of the parameters (e.g. 'R_r' for 'R_r_0' to 'R_r_28')
        - P_ref: reference trajectories per subject. These could be the parameters used to generate the data.
        
        May be deprecated. Use repfit_summary_fixed_data instead.
         """
        
        ## Determine which subjects are fit:
        if use_AIS_fits:
            q_fit_subjects = []
            for n in range(self.N):
                if self.q_fits is not None:
                    q_fit_subjects.append(n)
        else:
            q_fit_subjects = list(range(self.N))
        num_subjects = len(q_fit_subjects)
        print(q_fit_subjects)
            
        ## Determine which fit object to look in:
        if use_AIS_fits:
            fits = self.q_fits # AIS
            fit_type = 'AIS'
            if AIS_best_idxs is None:
                AIS_best_idxs = [-1 for n in q_fit_subjects]
        else:
            fits = self.fit # EM
            fit_type = 'EM'
            
        s_dat = self.subjects_data
         
        ## Make a figure of the time-varying estimates for each parameter, comprising all subjects:         
        for par_name in par_names:
            fig = go.Figure()

            # Determine whether to look in 'P' or 'P_aux'
            if par_name + '_1' in self.spec.keys():
                P_frame = 'P'
            else:
                P_frame = 'P_aux'
                
            # Find the trajectory for each subject and add it to the Figure:
            for n in q_fit_subjects:
                par_T = int((np.max(s_dat.block[:s_dat.T[n],n]) - np.min(s_dat.block[:s_dat.T[n],n])) / 2) + 1
                if use_AIS_fits:
                    ests = np.array([fits[n][P_frame][AIS_best_idxs[n]][par_name + '_' + str(d)]['val'] for d in range(par_T)])
                    ci = np.array([fits[n][P_frame][AIS_best_idxs[n]][par_name + '_' + str(d)]['ci'] for d in range(par_T)])    
                else:
                    ests = np.array([fits[n][P_frame][par_name + '_' + str(d)]['val'] for d in range(par_T)])
                    ci = np.array([fits[n][P_frame][par_name + '_' + str(d)]['ci'] for d in range(par_T)])
                
                par_name_title = par_name                
                if par_name == 'R_asym_inv':
                    ests = 1. / ests
                    ci[:,0] = 1. / ests
                    ci[:,1] = 1. / ests
                    par_name_title = "1/" + par_name
                                   
                xticks = np.arange(par_T)

                fig.add_trace(go.Scatter(
                        x=xticks,
                        y=ests,
                        error_y=dict(
                            type='data',
                            symmetric=False,
                            array= ci[:, 1] - ests,
                            arrayminus= ests - ci[:,0],
                            thickness = 1.),
                        mode = 'markers+lines',
                        name = str(self.subject_names[n]),
                        line = {'color': self.colors[n]}
                        ))
                
            if P_ref is not None:
                ests = np.array([P_ref[par_name + '_' + str(d)].iloc[0] for d in range(par_T)])
                fig.add_trace(go.Scatter(
                        x=xticks,
                        y=ests,
                        mode = 'markers+lines',
                        name = 'True',
                        line = {'color': 'blue', 'dash': 'dash'}
                        ))
               
            fig.update_layout(
                    title = self.name +": "+par_name_title,
                    xaxis_title = f'd: {par_name_title}_d',
                    yaxis_title = 'Estimates')
            if show_figs:
                fig.show()
            
            if save_folder is not None:
                results_folder = os.path.join(save_folder, self.name)
                if not os.path.exists(results_folder):
                    os.mkdir(results_folder)
                save_path = os.path.join(save_folder, self.name, f"{par_name}_trajectory_{fit_type}.jpeg")
                fig.write_image(save_path)     
                    
               

    def plot_true_vs_est_time_varying(self, par_names = [], par_T = 28, save_folder = None):
        """
        For models that use day varying parameters, plots the parameter estimates as a time series vs the true underlying values.
        
        - par_names: list of strongs, base names of the parameters (e.g. 'R_r' for 'R_r_0' to 'R_r_28')
        - par_T: int, the number of time steps the parameter varies over, to extract par_name_0 to par_name_par_T.
        
        May be deprecated. Use repfit_summary_fixed_data instead.
        """
        s_dat = self.subjects_data
        
        for par_name in par_names:
            fig = go.Figure()
            fig_lin = go.Figure()
            
            all_min = None; all_max = None
            for n in range(self.N):
                if par_name + '_1' in self.spec.keys():
                    ests = np.array([self.fit[n]['P'][par_name + '_' + str(d)]['val'] for d in range(par_T)])
                    ci = np.array([self.fit[n]['P'][par_name + '_' + str(d)]['ci'] for d in range(par_T)])
                else:
                    ests = np.array([self.fit[n]['P_aux'][par_name + '_' + str(d)]['val'] for d in range(par_T)])
                    ci = np.array([self.fit[n]['P_aux'][par_name + '_' + str(d)]['ci'] for d in range(par_T)])
                  
                xticks = np.arange(par_T)

                fig.add_trace(go.Scatter(
                        x=xticks,
                        y=ests,
                        error_y=dict(
                            type='data',
                            symmetric=False,
                            array= ci[:, 1] - ests,
                            arrayminus= ests - ci[:,0],
                            thickness = 1.),
                        mode = 'markers+lines',
                        name = str(self.subject_names[n]),
                        line = {'color': self.colors[n]}
                        ))
                
                true_vals = s_dat.P_aux_true[[par_name + '_' + str(d) for d in range(par_T)]].iloc[n,:].values
                
                fig.add_trace(go.Scatter(
                    x = xticks,
                    y = true_vals,
                    mode = 'markers+lines',
                    name = f"{self.subject_names[n]} (true)",
                    line = {'color': self.colors[n], 'dash': 'dash'}
                ))
                
                fig_lin.add_trace(go.Scatter(x = true_vals,
                                            y = ests,
                                            error_y = dict(
                                                        type='data',
                                                        symmetric=False,
                                                        array= ci[:, 1] - ests,
                                                        arrayminus= ests - ci[:,0],
                                                        thickness = 1.),
                                            mode = 'markers',
                                            name = str(self.subject_names[n]),
                                            line = {'color': self.colors[n]}))
                
                # Create the first diagonal, x = y, to display on the figure:
                minx = np.min(ci); maxx = np.max(ci)
                cover_x = np.linspace(minx, maxx)
                
                # min/max over all subjects:
                if all_min is not None:
                    all_min = min(all_min, minx)
                    all_max = max(all_max, maxx)
                else:
                    all_min = minx
                    all_max = maxx

                # Create a Linear Regression line
                lm = LinearRegression()
                lm.fit(true_vals.reshape(-1,1), ests)
                cover_y = lm.predict(cover_x.reshape(-1, 1)) 
                
                fig_lin.add_trace(go.Scatter(x = cover_x,
                                            y = cover_y,
                                            mode = 'lines',
                                            line = {'color': self.colors[n]}))
                        
     
            fig.update_layout(
                    title = self.name +": "+par_name,
                    xaxis_title = f'd: {par_name}_d',
                    yaxis_title = 'Estimates')
            fig.show()
            
            fig_lin.add_trace(go.Scatter(x = np.linspace(all_min, all_max),
                                        y = np.linspace(all_min, all_max),
                                        line = dict(color='firebrick', dash='dash')))
            fig_lin.update_layout(title = f"{self.name}: {par_name}",
                                 xaxis_title = 'True Values',
                                 yaxis_title = 'Estimates')
            fig_lin.show()
            
            if save_folder is not None:
                results_folder = os.path.join(save_folder, self.name)
                if not os.path.exists(results_folder):
                    os.mkdir(results_folder)
                fig.write_image(os.path.join(results_folder, f'true_vs_est_{par_name}_series.jpeg'))
                fig_lin.write_image(os.path.join(results_folder, f'true_vs_est_{par_name}_lin.jpeg'))
    
    ### A function to input subjects data into a given model:
    def input_subjects(self, s_dat):
        """
        After loading a model, subjects_data can be put in in this way.
        
        - s_dat: subjects data object.
        """
        self.subjects_data = s_dat
        self.N = s_dat.N
        self.colors = s_dat.colors
        self.subject_names = s_dat.subject_names
        
        return self
    
    ### Functions to display Q, RPE, and probability per trial for a given setting of parameters:
    def compute_Q_RPE_and_p(self, P = None, P_aux = None, return_dfs = False, display_P = False):
        """
        - P: a dataframe of parameters with number of columns corresponding to the number of subjects. If None given, the Posterior Means will be used.
        - P_aux: a dataframe of auxiliary parameters with number of columns corresponding to the number of subjects. If None given, the Posterior Means will be used.
        
        Computes the following per subject:
        1. The Q trajectory corresponding to the parameters.
        2. The Q_trajectory of an optional second process, corresponding to the parameters.
        3. The average probability per trial.
        4. The Reward Prediction Error of Q or Q_1.
        5. The Reward Prediction Error of Q_2.       
        """
        if P is None:
            self.P = self.extract_param_values(only_PM = True)[0]
        else:
            self.P = P
        if display_P:
            print('Parameters used to compute Q, RPE and p_t:')
            display(self.P)
        
        if P_aux is None and self.extract_param_values(only_PM = True)[1] is not None:
            self.P_aux = self.extract_param_values(only_PM = True)[1]
        else:
            self.P_aux = P_aux
        if display_P:
            print('Auxiliary parameters used to compute Q, RPE and p_t:')
            display(self.P_aux)
        
        self.Q_traj = []
        self.Q_traj_2 = []
        self.p_t    = []
        self.RPE    = []
        self.RPE_2  = []
        self.delta_Q_traj  = []
        self.Q_old  = []
        self.Q_new  = []
        self.Q_unchosen  = []
        for n in range(self.subjects_data.N):
            P_n = self.P.iloc[[n],:]
            if self.P_aux is None:
                P_aux_n = None
            else:
                P_aux_n = self.P_aux.iloc[[n],:]
            if self.llik_style == 'advanced_template':
                _,_,_,_,_, Q_trajs, _, _, _, _ = llik_adv(P_n, P_aux_n, n, self.subjects_data, self.style, return_Q = True)
            elif self.llik_style == 'simple_template':
                _,_,_, Q_trajs, _, _ = llik_simple(P_n, P_aux_n, n, self.subjects_data, self.style, return_Q = True)
            elif self.llik_style == 'custom':
                _, Q_trajs = self.llik(P_n, P_aux_n, n, self.subjects_data, return_Q = True)
            self.Q_traj.append(Q_trajs[0])
            self.Q_traj_2.append(Q_trajs[1])
            self.p_t.append(Q_trajs[3])
            self.RPE.append(Q_trajs[4])
            self.RPE_2.append(Q_trajs[5])
            self.delta_Q_traj.append(Q_trajs[6])
            self.Q_old.append(Q_trajs[7])
            self.Q_new.append(Q_trajs[8])
            self.Q_unchosen.append(Q_trajs[9])
            
        if return_dfs:
            dfs = []
            for n in range(self.subjects_data.N):
                df = self.subjects_data.create_subject_df(n=n)
                df['Q_chosen'] = self.Q_old[n]
                df['Q_chosen_new'] = self.Q_new[n]
                df['Q_unchosen'] = self.Q_unchosen[n]                
                df['RPE'] = self.RPE[n]
                df['delta_Q'] = self.delta_Q_traj[n]                
                dfs.append(df)
            return dfs
            
    def plot_p_t_and_RPE(self, save_folder = None, subject_names = None, show_figs = False, feedback_version = False):
        """
        Displays the probability per trial and RPE computed in self.compute_Q_RPE_and_p
        """
        s_dat = self.subjects_data
        _, columns = s_dat.subject_names_to_columns(subject_names, False)
        
        if not save_folder is None:
            model_folder = os.path.join(save_folder, self.name)
            if not os.path.exist(model_folder):
                os.mkdir(model_folder)
            p_t_folder = os.path.join(model_folder, 'p_t')    
            if not os.path.exist(p_t_folder):
                os.mkdir(p_t_folder)
            RPE_folder = os.path.join(model_folder, 'RPE')    
            if not os.path.exist(RPE_folder):
                os.mkdir(RPE_folder)
                
        if not hasattr(self, 'p_t'):
            sys.exit("Run 'compute_Q_RPE_and_p' first.")
                
        for n in columns:
            fig = go.Figure()
            fig_RPE = go.Figure()
            if feedback_version:
                fig.add_trace(go.Scatter(x = np.arange(s_dat.T[n])[s_dat.feedback[:,n] > .5],
                                        y = self.p_t[n][s_dat.feedback[:,n] > .5],
                                        marker_color = 'blue',
                                        mode = 'markers',
                                        name = 'feedback'))
                fig.add_trace(go.Scatter(x = np.arange(s_dat.T[n])[s_dat.feedback[:,n] < .5],
                                        y = self.p_t[n][s_dat.feedback[:,n] < .5],
                                        marker_color = 'red',
                                        mode = 'markers',
                                        name = 'no feedback'))      
                fig_RPE.add_trace(go.Scatter(x = np.arange(s_dat.T[n])[s_dat.feedback[:,n] > .5],
                                        y = self.RPE[n][s_dat.feedback[:,n] > .5],
                                        marker_color = 'blue',
                                        mode = 'markers',
                                        name = 'feedback'))
            else:
                fig.add_trace(go.Scatter(x = np.arange(s_dat.T[n]),
                                        y = self.p_t[n],
                                        marker_color = 'blue',
                                        mode = 'markers'))   
                fig_RPE.add_trace(go.Scatter(x = np.arange(s_dat.T[n]),
                                        y = self.RPE[n],
                                        marker_color = 'blue',
                                        mode = 'markers',
                                        name = 'feedback'))
            fig.update_layout(title = f'{s_dat.subject_names[n]}) Average Probability per trial',
                    yaxis = {'range' : [-.05,1.05], 'title' : 'probability'},
                    xaxis_title = 'trial index')
            fig_RPE.update_layout(title = f'{s_dat.subject_names[n]}) RPE per trial',
                                yaxis = {'title' : 'RPE'},
                                xaxis_title = 'trial index')
                                
            if not save_folder is None:
                fig.write_image(os.path.join(p_t_folder, f'{s_dat.subject_names[n]}.jpeg'))
                fig_RPE.write_image(os.path.join(RPE_folder, f'{s_dat.subject_names[n]}.jpeg'))
            if show_figs:
                fig.show()
                fig_RPE.show()
        
    def plot_Q(self, subject_name = '219', stims = [20], save_folder = None, show_figs = False, feedback_version = True):
        """
        Requires "compute_Q_RPE_and_p" to be run first.
        
        - n: int, the number of the subject to plot the Q on.
        - stims: list of int, the stim IDs to display.
        - display_end_Q: boolean, if True, displays the Q at the end of learning vs stim type.
        """
        s_dat = self.subjects_data
        _, columns = s_dat.subject_names_to_columns([subject_name], True)
        n = columns[0]
        
        if not hasattr(self, 'Q_traj'):
            sys.exit("Run 'compute_Q_RPE_and_p' first.")
        
        if not save_folder is None:
            model_folder = os.path.join(save_folder, self.name)
            if not os.path.exist(model_folder):
                os.mkdir(model_folder)
            Q_folder = os.path.join(model_folder, 'Q')    
            if not os.path.exist(Q_folder):
                os.mkdir(Q_folder)
        
        
        df = s_dat.create_subject_df(n=n).reset_index()
        
        for stim in stims:
            fig = go.Figure()
            
            if feedback_version:
                df_stim = df[np.logical_and(df['feedback'] > .5, df['C'] == stim)]
            else:
                df_stim = df[df['C'] == stim]
            
            
            # Find the day transition, if any:
            day_transition = np.nan
            for t in range(1, df_stim.shape[0]):
                if df_stim['feedback_time'].iloc[t] - df_stim['feedback_time'].iloc[t-1] > .5 * (1000. * 3600. * 16.):
                    day_transition = df_stim.index[t]
            
            # Display the Q trajectories for the given stimulus
            names = ['Q_1 (alpha_Q scaled)', 'Q_2 (alpha_Q scaled)']#, 'P_trace (invtemp_P scaled)']
            colors = ['blue', 'red']#, 'orange']
            for k, Q in enumerate([self.Q_traj[n], self.Q_traj_2[n]]):
                if Q is not None:
                    fig.add_trace(go.Scatter(x = df_stim.index,
                                            y = Q[stim, df_stim.index],
                                            mode = 'lines',
                                            name = names[k],
                                            marker_color = colors[k]))
                
            # Add the outcomes of the trials to the figure:
            r_trials = df_stim.loc[df_stim['o'] > .5].index
            fig.add_trace(go.Scatter(x = r_trials,
                                    y = np.zeros_like(r_trials),
                                     marker_color = 'green',
                                    name = 'Reward',
                                    mode = 'markers'))
            n_trials = df_stim.loc[df_stim['o'] < .5].index
            fig.add_trace(go.Scatter(x = n_trials,
                                    y = np.zeros_like(n_trials),
                                     marker_color = 'purple',
                                    name = 'Neutral',
                                    mode = 'markers'))
            pun_trials = df_stim.loc[df_stim['o'] < -.5].index
            fig.add_trace(go.Scatter(x = pun_trials,
                                    y = np.zeros_like(pun_trials),
                                     marker_color = 'red',
                                    name = 'Punishment',
                                    mode = 'markers'))
            
            if day_transition > 0:
                fig.add_vline(x=day_transition)
            fig.update_layout(title = f'Q during learning, subject {self.subject_names[n]}, stim {stim}',
                             xaxis_title = 'trial index',
                             yaxis_title = 'Q')
            if show_figs:
                fig.show()
            
            if save_folder is not None:
                fig.write_image(os.path.join(Q_folder, f'{self.subject_names[n]}_stim_{stim}_Q.jpeg'))
        
        # if display_end_Q:
            # for n in columns:
                # Q_traj = self.Q_traj[n]

                # stim_last_learning_idx = np.zeros(s_dat.Nc[n]).astype('int64')
                # stim_type = np.zeros(s_dat.Nc[n]) -1
                # for t in range(self.T[n]):
                    # if s_dat.feedback[t,n] > .5:
                        # stim_last_learning_idx[s_dat.stims[t,:,n]] = t
                    # stim_type[s_dat.stims[t,:,n]] = s_dat.stim_types[t,:,n] 
                # stim_last_learning_Q = Q_traj[0][stim_last_learning_idx[20:]]

                # #print(stim_last_learning_idx)
                # ## TODO: perhaps first keep track of the stim type table.

                # fig_end_Q = go.Figure()
                # fig_end_Q.add_trace(go.Scatter(x=stim_type[20:],
                                              # y = stim_last_learning_Q,
                                              # mode = 'markers',
                                              # name = 'Q1',
                                              # marker_color = self.colors[n]))
                # if self.Q_traj_2[n] is not None:
                    # Q_traj2 = self.Q_traj_2[n]
                    # stim_last_learning_Q2 = Q_traj2[stim_last_learning_idx[20:]]
                    # fig_end_Q.add_trace(go.Scatter(x=stim_type[20:],
                                              # y = stim_last_learning_Q2,
                                              # mode = 'markers',
                                              # name = 'Q2',
                                              # marker_color = self.colors[n]))
                # fig_end_Q.update_layout(title = f'{self.subject_names[n]}) Q at end of learning versus stimulus type',
                                       # xaxis_title = 'Stim Type',
                                       # yaxis_title = 'Q')
                # if show_figs:
                    # fig_end_Q.show()
                # if save_folder is not None:
                    # fig_end_Q.write_image(os.path.join(Q_folder, f'{self.subject_names[n]}_end_Q.jpeg'))
    
    # Deprecated:    
    def spec_q_summary(self, n = 0, save_folder = None, show_figs = True):
        
        if self.q_fits[n] is None:
            sys.exit('No AIS fit is present yet!')
            
        fits = self.q_fits[n]
        num_mixtures = len(fits[0])
        
        for mx in num_mixtures:
            fig = go.Figure()
            
            for par_name in fits[0][0].keys():
                avg_loc_displacement = 0.
                avg_std = fits[-6][mx][par_name]['val'][1]
                for k in range(-5,0):
                    avg_loc_displacement += np.abs(fits[k][mx][par_name]['val'][0] - fits[k-1][mx][par_name]['val'][0]) 
                    avg_std += fits[k][mx][par_name]['val'][1]
                avg_loc_displacement = avg_loc_displacement / 5.
                avg_std = avg_std / 6.
                
                fig.add_trace(go.Scatter(x = [avg_std],
                                        y = [avg_loc_displacement],
                                        name = par_name,
                                        color = 'blue'))
            fig.update_layout(title = f'{self.subject_names[n]}) Mixture {mx}, avg location displacement vs avg stdev',
                            xaxis_title = 'Average Standard deviation over the last 6 iterations',
                            yaxis_title = 'Average Absolute Location Change over the last 6 iterations')
            if save_folder is not None:
                if not os.path.exists(save_folder):
                    os.mkdir(save_folder)
                model_folder = os.path.join(save_folder, self.name)
                if not os.path.exists(model_folder):
                    os.mkdir(model_folder)
                subject_AIS_folder = os.path.join(model_folder, f'{self.subject_names[n]}_AIS')
                if not os.path.exists(subject_AIS_folder):
                    os.mkdir(subject_AIS_folder)
                fig.write_image(os.path.join(subject_AIS_folder, f'spec_q_{mx}_summary.jpeg'))
            if show_figs:
                fig.show()
            
def plot_custom_est(self, S = 10000, subjects_data = None, save_folder = None):
    """
    This is a function that given a loaded or actual model, and subjects_data if it is a loaded model,
    computes the posterior of subject-specific custom parameters. This function will need to be edited to create the custom parameters.
    
    - save_folder: None or string, if not None, saves figures to "{save_folder}\\{self.name}\\{custom_par_name}.jpeg".
    
    """
    if subjects_data is not None:
        subject_names = subjects_data.subject_names
        for k in range(len(subject_names)):
            if subject_names[k] != self.subject_names[k]:
                sys.exit("Not the same (order of) subjects!")

        self.subjects_data = subjects_data
        self.T = subjects_data.T
        self.Nc = subjects_data.Nc
        self.colors = subjects_data.colors
        
    self.P = None
    self.S = S
    
    custom_fit = [{'evidence': np.nan, 'N_samples' : None, 'P': {}} for n in range(self.N)]
    tm = time.time()
    
    for n in range(self.N):
        self.sampleP(S, n)
        llik, llik_f, llik_nf, ESS, S_final, ESS_scaled, _ = llik_adv(self.P, self.P_aux, n, self.subjects_data, self.style, return_Q = False, ESS_it_interval = 10000)#self.llik(self.P, self.P_aux, n)
        
        # Find the lines where there is numerical overflow:
        inc = ~((np.abs(llik) < 1e-10) | (llik == np.nan))
        custom_fit[n]['N_samples'] = llik[inc].shape[0]
        llik_inc = llik[inc]
        
        # Compute the evidence (log mean likelihood):
        lsumlik = logsumexp_1D(llik_inc)
        custom_fit[n]['evidence'] = lsumlik - np.log(custom_fit[n]['N_samples'])
        
        # Compute weights for resampling:
        weights = np.exp(llik_inc - lsumlik)
        
        # Compute the Effective Sample Size:
        custom_fit[n]['ESS'] = 1. /  np.sum(np.square(weights))
        
        
        #######################
        ## Customizable part ##
        #######################
        
        custom_params =   pd.DataFrame({
            'UCB_diff_20': self.P['UCB_weight'].values / np.sqrt(self.P['invsigma2_0'].values + 20 ) - self.P['UCB_weight'].values / np.sqrt(self.P['invsigma2_0'].values),           
            'UCB_diff_10': self.P['UCB_weight'].values / np.sqrt(self.P['invsigma2_0'].values + 10 ) - self.P['UCB_weight'].values / np.sqrt(self.P['invsigma2_0'].values),          
            'UCB_diff_3': self.P['UCB_weight'].values / np.sqrt(self.P['invsigma2_0'].values + 3 ) - self.P['UCB_weight'].values / np.sqrt(self.P['invsigma2_0'].values),
            'invtemp_delta': self.P['invtemp'].values * self.P['delta_Q'].values
        })
        
#         pd.DataFrame({
#             'plus_transform': (self.P['gamma_Q_plus'] * (1. - self.P['delta_Q_plus'])).values,
#             'min_transform': (self.P['gamma_Q_min'] * (-1. - self.P['delta_Q_min'])).values,
#             'plus_transform_small': (self.P['gamma_Q_plus'] * (.1 - self.P['delta_Q_plus'])).values,
#             'min_transform_small': (self.P['gamma_Q_min'] * (-.1 - self.P['delta_Q_min'])).values,
            
#         })      
        
#         pd.DataFrame({
#                         'plus_transform': (self.P['gamma_Q'] * (1. - self.P['delta_Q'])).values,
#                         'min_transform': (self.P['gamma_Q'] * (-1. - self.P['delta_Q'])).values,
#                         'plus_transform_small': (self.P['gamma_Q'] * (.1 - self.P['delta_Q'])).values,
#                         'min_transform_small': (self.P['gamma_Q'] * (-.1 - self.P['delta_Q'])).values,
#         })
    
        
        
#         pd.DataFrame({'R_pun - R_n': (self.P['R_pun'] - self.P['R_n']).values,
#                          'R_r - R_n': (self.P['R_r'] - self.P['R_n']).values
#                          #'R_pun_sc': (self.P['R_pun'] * self.P['invtemp']).values
#                         })
        
#         pd.DataFrame({#'early_zeta': (self.P['invtemp'] * self.P['zeta'] * (self.P['lrate'] + self.P['alpha_bayes'] / (self.P['eps'] + 2))).values,
#                                       #'late_zeta': (self.P['invtemp'] * self.P['zeta'] * (self.P['lrate'] + self.P['alpha_bayes'] / (self.P['eps'] + 40))).values,
#                                       'early_reward': (self.P['invtemp'] * (self.P['lrate'] + self.P['alpha_bayes'] / (self.P['eps'] + 2))).values,
#                                       'late_reward': (self.P['invtemp'] * (self.P['lrate'] + self.P['alpha_bayes'] / (self.P['eps'] + 40))).values,
#         })#'alpha': self.P['alpha'].values})
        
        
#         pd.DataFrame({'alpha_bayes_div_(eps + 40)': (self.P['alpha_bayes']/(self.P['eps'] + 40)).values,
#                                      'alpha_bayes_div_(eps + 2)': (self.P['alpha_bayes']/(self.P['eps'] + 2)).values,
#                                      })
            
            
        
        #######################
        #######################
        
        for par_name in custom_params.columns:
            par_values = custom_params[par_name].values[inc]
            param_fit = {}
       
            # Order the parameter values from small to large
            ranks = np.argsort(par_values)
            oparams = par_values[ranks]
            oweights = weights[ranks]
                 
            # Mean of the posterior:
            param_fit['val'] = np.sum(oweights * oparams)
            
            # Credibility interval:
            cdf = np.cumsum(oweights)
            ci_thresh = np.array([.025, .975])
            param_fit['ci'] = np.zeros(2) 
            i = 0; j = 0
            while i < 2:
                if cdf[j] > ci_thresh[i]:
                    param_fit['ci'][i] = oparams[j]
                    i += 1
                else:
                    j += 1
                    
            custom_fit[n]['P'][par_name] = param_fit

                 
            # 1000 equally spaced quantiles:
#             I = self.I
#             sampp = np.arange(1. / (2.*I), 1, 1. / I)
#             samp = np.zeros(I)
#             i = 0; j = 0
#             ci_i = 0
#             while i < I:
#                 if cdf[j] > sampp[i]:
#                     samp[i] = oparams[j]
#                     i += 1
#                 else:
#                     j += 1
#             param_fit['samp'] = samp
            
     
    N = self.N
        
    for par_name in custom_params.columns:
        ests = np.array([custom_fit[n]['P'][par_name]['val'] for n in range(self.N)])
        ci = np.array([custom_fit[n]['P'][par_name]['ci'] for n in range(self.N)])

        xticks = np.arange(N)

        fig = go.Figure()
        for n in range(N):
            fig.add_trace(go.Scatter(
                    x=[xticks[n]],
                    y=[ests[n]],
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array= [ci[n, 1] - ests[n]],
                        arrayminus= [ests[n] - ci[n,0]],
                        thickness = 1.),
                    mode = 'markers',
                    name = str(self.subject_names[n]),
                    line = {'color': self.colors[n]}
                    ))
            fig.update_layout(
                title = self.name +": "+par_name,
                xaxis_title = 'Subjects',
                yaxis_title = 'Estimates')
        fig.show()
        
        if save_folder is not None:
            results_folder = os.path.join(save_folder, self.name)
            if not os.path.exists(results_folder):
                os.mkdir(results_folder)
            save_path = os.path.join(save_folder, self.name, f"{par_name}.jpeg")
            fig.write_image(save_path)
        
    print(f"==> BIC_int   - {self.bic:.2f}.")
    print(f"    Evidence  - {self.evidence:.2f}.")
    print(f"    Mean ESS  - {np.mean([custom_fit[n]['ESS'] for n in range(self.N)])} / {S}.")
    print(f"    Stdev ESS - {np.std([custom_fit[n]['ESS'] for n in range(self.N)])}.")
    print(f"    Time      - {time.time() - tm}s.\n")
    
def plot_custom_series_est(self, S = 10000, subjects_data = None, save_folder = None):
    """
    This is a function that given a loaded or actual model, and subjects_data if it is a loaded model,
    computes the posterior of subject-specific custom parameters. This function will need to be edited to create the custom parameters.
    
    - save_folder: None or string, if not None, saves figures to "{save_folder}\\{self.name}\\{custom_par_name}.jpeg".
    
    """
    if subjects_data is not None:
        subject_names = subjects_data.subject_names
        for k in range(len(subject_names)):
            if subject_names[k] != self.subject_names[k]:
                sys.exit("Not the same (order of) subjects!")

        self.subjects_data = subjects_data
        self.T = subjects_data.T
        self.Nc = subjects_data.Nc
        self.colors = subjects_data.colors
        
    self.P = None
    self.S = S
    
    custom_fit = [{'evidence': np.nan, 'N_samples' : None, 'P': {}} for n in range(self.N)]
    tm = time.time()
    
    MAP_custom_params = [] # N subjects, for each subject list of 4 custom_params series
    
    for n in range(self.N):
        self.sampleP(S, n)
        llik, llik_f, llik_nf, ESS, S_final, ESS_scaled, _ = llik_adv(self.P, self.P_aux, n, self.subjects_data, self.style, return_Q = False, ESS_it_interval = 10000)#self.llik(self.P, self.P_aux, n)
        
        # Find the lines where there is numerical overflow:
        inc = ~((np.abs(llik) < 1e-10) | (np.isnan(llik)))
        custom_fit[n]['N_samples'] = llik[inc].shape[0]
        llik_inc = llik[inc]
        
        # Compute the evidence (log mean likelihood):
        lsumlik = logsumexp_1D(llik_inc)
        custom_fit[n]['evidence'] = lsumlik - np.log(custom_fit[n]['N_samples'])
        
        # Compute weights for resampling:
        weights = np.exp(llik_inc - lsumlik)
        
        # Compute the Effective Sample Size:
        custom_fit[n]['ESS'] = 1. /  np.sum(np.square(weights))
        
        
        #######################
        ## Customizable part ##
        #######################
        el_sad_sd = np.std(subjects_data.el_sad_last[:,n])
        P = self.P
        
        K = 10
        custom_params_1 =  np.zeros((K, S))
        custom_params_2 =  np.zeros((K, S))
        custom_params_3 =  np.zeros((K, S))
        custom_params_4 =  np.zeros((K, S))
        for k in range(K): # Bit inefficient
            a_plus = 1. #P['a_plus_0'].values 
            eps_plus = P['eps_plus_0'].values 
            a_min = 1. #P['a_min_0'].values 
            eps_min = P['eps_min_0'].values 
            custom_params_1[k,:] = 1. / (a_plus * k + eps_plus)
            custom_params_3[k,:] = 1. / (a_min * k + eps_min)
            
            # P['a_plus_0'].values *
            # P['a_min_0'].values *
            a_plus =  np.exp(P['a_plus_el_sad'].values * el_sad_sd)
            eps_plus = P['eps_plus_0'].values * np.exp(P['eps_plus_el_sad'].values * el_sad_sd)
            a_min =  np.exp(P['a_min_el_sad'].values * el_sad_sd)
            eps_min = P['eps_min_0'].values * np.exp(P['eps_min_el_sad'].values * el_sad_sd)
            custom_params_2[k,:] = 1. / (a_plus * k + eps_plus)
            custom_params_4[k,:] = 1. / (a_min * k + eps_min)
      
        custom_params = [custom_params_1, custom_params_2, custom_params_3, custom_params_4]
        
        MAP = []
        for params in custom_params:          
            MAP.append(np.sum(params * weights[None,:], axis = 1))        
        MAP_custom_params.append(MAP)
        
    N = self.N
        
    fig_plus = go.Figure()
    for n in range(N):
        fig_plus.add_trace(go.Scatter(
                x=np.arange(K),
                y=MAP_custom_params[n][0],
                mode = 'lines+markers',
                name = str(self.subject_names[n]) + ' (mean mood)',
                line = {'color': self.colors[n]}
                ))
        fig_plus.add_trace(go.Scatter(
                x=np.arange(K),
                y=MAP_custom_params[n][1],
                mode = 'lines+markers',
                name = str(self.subject_names[n]) + ' (mean + sd mood)',
                line = {'color': self.colors[n], 'dash' : 'dash'}
                ))
    fig_plus.update_layout(
        title = self.name +": learning rate pos RPE",
        xaxis_title = 'N = k',
        yaxis_title = 'Estimates')
    fig_plus.show()
    
    fig_min = go.Figure()
    for n in range(N):
        fig_min.add_trace(go.Scatter(
                x=np.arange(K),
                y=MAP_custom_params[n][2],
                mode = 'lines+markers',
                name = str(self.subject_names[n]) + ' (mean mood)',
                line = {'color': self.colors[n]}
                ))
        fig_min.add_trace(go.Scatter(
                x=np.arange(K),
                y=MAP_custom_params[n][3],
                mode = 'lines+markers',
                name = str(self.subject_names[n]) + ' (mean + sd mood)',
                line = {'color': self.colors[n], 'dash' : 'dash'}
                ))
    fig_min.update_layout(
        title = self.name +": learning rate neg RPE",
        xaxis_title = 'N = k',
        yaxis_title = 'Estimates')
    fig_min.show()

    if save_folder is not None:
        results_folder = os.path.join(save_folder, self.name)
        if not os.path.exists(results_folder):
            os.mkdir(results_folder)
        save_path = os.path.join(save_folder, self.name, f"lrate_pos_RPE.jpeg")
        fig_plus.write_image(save_path)
        save_path = os.path.join(save_folder, self.name, f"lrate_neg_RPE.jpeg")
        fig_min.write_image(save_path)
        
    print(f"==> BIC_int   - {self.bic:.2f}.")
    print(f"    Evidence  - {self.evidence:.2f}.")
    print(f"    Mean ESS  - {np.mean([custom_fit[n]['ESS'] for n in range(self.N)])} / {S}.")
    print(f"    Stdev ESS - {np.std([custom_fit[n]['ESS'] for n in range(self.N)])}.")
    print(f"    Time      - {time.time() - tm}s.\n")

def plot_multiple_model_ests_time_varying(model_name, s_dat, fits_models = [], P_ref = None, par_names = [], save_folder = None, names = None, colors = None, show_figs = False, use_AIS_fits = False, AIS_fit_idxs = None, subject_names = None, with_evidence = True, par_T_minus = 0):
    """
    For models that use day varying parameters, plots the parameter estimates as a time series vs the true underlying values.

    - par_names: list of strongs, base names of the parameters (e.g. 'R_r' for 'R_r_0' to 'R_r_28')
    - par_T: int, the number of time steps the parameter varies over, to extract par_name_0 to par_name_par_T.
    - P_aux_models: list of P_aux extracted from the different models.
    - s_dat: subjects_data object that the models were fit to.
    - model_name: string, name of the model
    - par_T_minus: int, the number of end days not to display for subjects

    """
    _, columns = s_dat.subject_names_to_columns(subject_names, True)
    subject_names = []

    num_models = len(fits_models)
    fitted_subjects = []
    for n, fits in enumerate(fits_models[0]):
        if fits is not None and n in columns:
            fitted_subjects.append(n)
            subject_names.append(s_dat.subject_names[n])
    num_subjects = len(fitted_subjects)
            
    if names is None:
        names = [f"{k}" for k in range(num_models)]

    for l, n in enumerate(fitted_subjects):
        if colors is None:
            colors = [px.colors.qualitative.Plotly[nm % 10] for nm in range(num_models)]
            
        par_T = int(np.max(s_dat.session[:s_dat.T[n],n])) + 1 - par_T_minus
        
        for par_name in par_names:
            fig = go.Figure()

            for m, fits in enumerate(fits_models):
                xticks = np.arange(par_T)
                
                ests = np.array([fits[n]['P_aux'][par_name + '_' + str(d)]['val'] for d in range(par_T)])
                ci = np.array([fits[n]['P_aux'][par_name + '_' + str(d)]['ci'] for d in range(par_T)])
                
                par_name_title = par_name                
                if par_name == 'R_asym_inv':
                    ests = 1. / ests
                    ci[:,0] = ests
                    ci[:,1] = ests
                    par_name_title = "1/" + par_name
                
                if with_evidence:
                    name = f"{names[m]}: {fits[n]['evidence'][-1]:.1f}"
                fig.add_trace(go.Scatter(
                        x=xticks,
                        y=ests,
                        error_y=dict(
                            type='data',
                            symmetric=False,
                            array= ci[:,1] - ests,
                            arrayminus= ests - ci[:,0],
                            thickness = 1.),
                        mode = 'markers+lines',
                        name = name,
                        line = {'color': colors[m]}
                        ))
                
            if P_ref is not None:
                if P_ref.shape[0] == 1:
                    ests = np.array([P_ref[par_name + '_' + str(d)].iloc[0] for d in range(par_T)])
                    if par_name == 'R_asym_inv':
                        ests = 1. / ests
                    fig.add_trace(go.Scatter(
                            x=xticks,
                            y=ests,
                            mode = 'markers+lines',
                            name = 'True',
                            line = {'color': 'blue', 'dash': 'dash'}
                            ))
                else:
                    ests = np.array([P_ref[par_name + '_' + str(d)].iloc[n] for d in range(par_T)])
                    
                    if par_name == 'R_asym_inv':
                        ests = 1. / ests
                    fig.add_trace(go.Scatter(
                            x=xticks,
                            y=ests,
                            mode = 'markers+lines',
                            name = 'True',
                            line = {'color': 'blue', 'dash': 'dash'}
                            ))

            fig.update_layout(
                    title = f"{subject_names[l]}: {par_name_title}",
                    xaxis_title = f'd: {par_name_title}_d',
                    yaxis_title = 'Estimates')
            if show_figs:
                fig.show()

            if save_folder is not None:
                results_folder = os.path.join(save_folder, model_name)
                if not os.path.exists(results_folder):
                    os.mkdir(results_folder)
                fig.write_image(os.path.join(results_folder, f'{subject_names[l]}_{par_name}.jpeg'))

def plot_multiple_model_ests(model_name, s_dat, subject_names = None, P_ref = None, fits_models = [], save_folder = None, show_figs=True, model_names = None, with_evidence = False, par_names = None):
    """
    Plots the posterior means with their credibility interval, for all subjects.

    - save_figs: bool, if True, saves the figures in "model_ests\\{self.name}\\{par_name}.jpeg"
    """
    _, columns = s_dat.subject_names_to_columns(subject_names, True)
    subject_names = []
    
    M = len(fits_models)
    fitted_subjects = []
    for n, fits in enumerate(fits_models[0]):
        if fits is not None and n in columns:
            fitted_subjects.append(n)
            subject_names.append(s_dat.subject_names[n])
    N = len(fitted_subjects)
    
    if M == 1:
        M_displacement = np.array([0.1])
    else:
        M_displacement = np.linspace(0.1,.5,num=M)
        
    if model_names is None:
        model_names = list(range(M))
        
    if par_names is None:
        par_names = fits_models[0][fitted_subjects[0]]['P'].keys()

    for par_name in par_names:
        fig = go.Figure()
        
        m_ests = []

        for m in range(M):
            P = fits_models[m]
            ests = np.array([P[n]['P'][par_name]['val'] for n in fitted_subjects])
            m_ests.append(ests)
            ci = np.array([P[n]['P'][par_name]['ci'] for n in fitted_subjects])
 
            xticks_ref = np.arange(N)
            xticks = np.arange(N) + M_displacement[m]
        
            if P_ref is not None:
                ests_ref = np.array([P_ref[par_name].iloc[n] for n in fitted_subjects])

            for n in fitted_subjects:
                fig.add_trace(go.Scatter(
                        x=[xticks[n]],
                        y=[ests[n]],
                        error_y=dict(
                            type='data',
                            symmetric=False,
                            array= [ci[n, 1] - ests[n]],
                            arrayminus= [ests[n] - ci[n,0]],
                            thickness = 1.),
                        mode = 'markers',
                        name = f'{s_dat.subject_names[n]}) {model_names[m]}: {P[n]["evidence"][-1]:.1f}',
                        line = {'color': s_dat.colors[n]}
                        ))
                if P_ref is not None:
                    fig.add_trace(go.Scatter(
                            x= [n],
                            y= [ests_ref[n]],
                            mode = 'markers',
                            name = f'{s_dat.subject_names[n]}) True',
                            line = {'color': s_dat.colors[n], 'dash': 'dash'}))
        m_ests = np.array(m_ests)
        avg_ests = np.mean(m_ests, axis = 0)
        if P_ref is not None:
            for n in fitted_subjects:
                fig.add_trace(go.Scatter(
                        x= [n, n + .5],
                        y= [avg_ests[n], avg_ests[n]],
                        mode = 'lines',
                        name = f'{s_dat.subject_names[n]}) avg',
                        line = {'color': s_dat.colors[n], 'dash': 'dash'}
                        ))       
        fig.update_layout(
            title = model_name +": "+par_name,
            xaxis_title = 'Subjects',
            yaxis_title = 'Estimates')
        if show_figs:
            fig.show()

        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
            results_folder = os.path.join(save_folder, model_name)
            if not os.path.exists(results_folder):
                os.mkdir(results_folder)
            save_path = os.path.join(save_folder, model_name, f"{par_name}.jpeg")
            fig.write_image(save_path)


def plot_est_time_varying(self, par_names = [], P_ref = None, par_T = 28, save_folder = None, show_figs = True, corr_vars = None, lm_mode = 'OLS', plot_sum = False, plot_diff = False, use_AIS_fits = False, log_transform = True, AIS_best_idxs = None):
    """
    For models that use day varying parameters, plots the parameter estimates as a time series
    
    - par_names: list of strongs, base names of the parameters (e.g. 'R_r' for 'R_r_0' to 'R_r_28')
    - par_T: int, the number of time steps the parameter varies over , to extract par_name_0 to par_name_par_T.
    - corr_vars: if not None, list of mood variables (string) to correlate the day-varying parameters with. Takes the first of each day.
    - plot_sum: if True, plots the sum of the first two variables against the corr_vars as well, plus the trajectories of the sum and differences. This is used to asses the asymmetry its dependency on corr_vars.
    """
    
    ## Determine which subjects are fit:
    if use_AIS_fits:
        q_fit_subjects = []
        for n in range(self.N):
            if self.q_fits is not None:
                if n < 4:
                    q_fit_subjects.append(n)
    else:
        q_fit_subjects = list(range(self.N))
    num_subjects = len(q_fit_subjects)
    print(q_fit_subjects)
        
    ## Determine which fit object to look in:
    if use_AIS_fits:
        fits = self.q_fits # AIS
        fit_type = 'AIS'
    else:
        fits = self.fit # EM
        fit_type = 'EM'
     
    ## Make a figure of the time-varying estimates for each parameter, comprising all subjects:         
    for par_name in par_names:
        fig = go.Figure()

        # Determine whether to look in 'P' or 'P_aux'
        if par_name + '_1' in self.spec.keys():
            P_frame = 'P'
        else:
            P_frame = 'P_aux'
            
        # Find the trajectory for each subject and add it to the Figure:
        for n in q_fit_subjects:
            if use_AIS_fits:
                ests = np.array([fits[n][P_frame][AIS_best_idxs[n]][par_name + '_' + str(d)]['val'] for d in range(par_T)])
                ci = np.array([fits[n][P_frame][AIS_best_idxs[n]][par_name + '_' + str(d)]['ci'] for d in range(par_T)])    
            else:
                ests = np.array([fits[n][P_frame][par_name + '_' + str(d)]['val'] for d in range(par_T)])
                ci = np.array([fits[n][P_frame][par_name + '_' + str(d)]['ci'] for d in range(par_T)])
            
                
            xticks = np.arange(par_T)

            fig.add_trace(go.Scatter(
                    x=xticks,
                    y=ests,
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array= ci[:, 1] - ests,
                        arrayminus= ests - ci[:,0],
                        thickness = 1.),
                    mode = 'markers+lines',
                    name = str(self.subject_names[n]),
                    line = {'color': self.colors[n]}
                    ))
            
        if P_ref is not None:
            ests = np.array([P_ref[par_name + '_' + str(d)].iloc[0] for d in range(par_T)])
            fig.add_trace(go.Scatter(
                    x=xticks,
                    y=ests,
                    mode = 'markers+lines',
                    name = 'True',
                    line = {'color': 'blue', 'dash': 'dash'}
                    ))

            
        fig.update_layout(
                title = self.name +": "+par_name,
                xaxis_title = f'd: {par_name}_d',
                yaxis_title = 'Estimates')
        if show_figs:
            fig.show()
        
        if save_folder is not None:
            results_folder = os.path.join(save_folder, self.name)
            if not os.path.exists(results_folder):
                os.mkdir(results_folder)
            save_path = os.path.join(save_folder, self.name, f"{par_name}_trajectory_{fit_type}.jpeg")
            fig.write_image(save_path)
            
    # corr_vars will be mood variables. 
    # For each subject, we want to regress m_{t + 24h} ~ m_t + params.
    # And then display the parameter estimates for each subject.
    
    
    


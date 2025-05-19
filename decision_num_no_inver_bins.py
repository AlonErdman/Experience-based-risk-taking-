import pandas as pd
import numpy as np
from scipy.special import expit, logit

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
    
def gen_outcome(p_pun, p_r):
    """
    Generate outcomes according to the EMA environment.
    
    - R_st  : float, the reward probabilities of the current stimuli, order (stim1, stim2).
    - Pun_st: float, the punishment probabilities of the current stimuli, order (stim1, stim2).
    
    Returns:
    - float, -1 for punishment, 0 for neutral, 1 for reward
    """
    p_cum = np.array([p_pun, 1. -  p_r, 1.])    
    rv = np.random.uniform(size = 1)    
    idx = 0    
    for i in range(3):
        idx += (rv > p_cum[i])
    
    return idx - 1
    
def llik_decision_num_no_inver_nf_bins(P, P_aux, n, s_dat_dec, return_Q = False, simulate_data = False):
    """
    This is an example of a complete llik function to be used as a custom llik function in the fitting algorithm. 
    In general, custom llik functions need to take as:
    
    Arguments:
    1. P: (S, .) DataFrame of parameter values
    2. P_aux: (S, .) DataFrame of auxiliary parameter values.
    3. n: int, the subject index to fit on
    4. s_dat_dec: subjects data class object, an object of the specific class created to store your data in. You use state information from this class, preferably used as numpy arrays for speed (much faster than DataFrames), 
              and including the following information: 
                - .N (int, number of subjects),
                - .T (list of ints, number of trials per subject),
                - .Nc (list of ints, number of stimuli per subject)
                - .num_datapoints (the number of fitted datapoints),
                - .subject_names (list of strings, the subject names to be used in Figures),
                - .colors (list of color codings, to be used in Figures).
    
    Keywords:
    - return_Q: boolean, if True, return a full Q-trajectory
    - simulate_data: boolean, if True, simulates data with the first parameter combination.    

    
    Returns:
    - llik:    (S)-numpy array of log-likelihoods corresponding to the parameter samples.
    - Q_traj:  A list or tuple with in the corresponding positions, if return_Q = True - if these are not desired, just return None:
                1. Q_1: a (T)-numpy array of Q-values, potentially rescaled.
                2. Q_2: a (T)-numpy array of Q-values of the second process, potentially rescaled.
                3. None or an optional extra trace, (T)-numpy array, sucj as a perseverance trace.
                4. p_t, a (T)-numpy array of probabilities per trial of the chosen stimuli (not on the log scale)
                5. RPE, a (T)-numpy array of Reward Prediction Errors
                6. RPE2, a (T)-numpy array of Reward Prediction Errors of the second process.
    """
    
    S  = P.shape[0]
    Nc = s_dat_dec.Nc[n]
    T  =  s_dat_dec.T[n]
    min_stim = s_dat_dec.min_stim
    
    ## Initialization of the latent process:
    Q = np.zeros((Nc, S))
    
    ## Initialization of the Q trajectory to return of Q_traj = True:
    Q_traj = np.zeros((Nc, T))
    p_t    = np.zeros((T))
    RPE    = np.full((T), np.nan)
    
    #Initialization:
    llik = np.zeros((S))
    old_session = -0.5 #sessions without bins equal minus 1, so this step assure only sessions=>0 will get paramters.
    session     = -0.5
    R_r         = np.nan
    R_pun       = np.nan
    
    if simulate_data:
        T = s_dat_dec.T[n];
        s_dat_dec.C[:,n]      = s_dat_dec.C[:,n] 
        s_dat_dec.C_st[:,n]   = s_dat_dec.C_st[:,n]
        s_dat_dec.o[:,n]      = s_dat_dec.o[:,n]
        
    counter = np.zeros((3, Nc, S))
    for t in range(T): 
        # Extract the relevant state information:
        st = s_dat_dec.stims[t,:,n]
        if not simulate_data:
            C  = s_dat_dec.C[t,n]; C_st = s_dat_dec.C_st[t,n]
            o  = s_dat_dec.o[t,n]
        else:
            C_st = 1
            C = st[C_st]
        feedback = s_dat_dec.feedback[t,n]
        stim_types = s_dat_dec.stim_types[t, : , n]   

        # Update the parameters if necessary:
        session = s_dat_dec.session[t,n]
        if (session > old_session): 
            R_r         = P_aux[f'R_r_{session}'].values
            R_pun       = P_aux[f'R_pun_{session}'].values           
            old_session = session          
          
        #apply weights on both stimuli (before decision) if the choice is between two no-pratice stimuli and this is a no-feedback trial: 
        if (st[0] >= min_stim) and (st[1] >= min_stim) and (feedback < .5):
            reward_minus_pun = R_r * counter[0, st, :] + R_pun * counter[1, st, :]
            sum_all = counter[0, st, :] + counter[1, st, :] + counter[2, st, :]
            if np.sum(sum_all[0]) > 0: #if denominator eqauls zero (i.e. the stimulus was never chosen during feedback trials) then Q[st] will stay 0 
                Q[st[0],:] = np.divide(reward_minus_pun[0], sum_all[0])
            if np.sum(sum_all[1]) >0: #if denominator eqauls zero (i.e. the stimulus was never chosen during feedback trials) then Q[st] will stay 0 
                Q[st[1],:] = np.divide(reward_minus_pun[1], sum_all[1])
        
        if not simulate_data:
            # Compute the logprobabilityies of the choice                
            if (st[0] >= min_stim) and (st[1] >= min_stim): # Only use no practice trials
                logp = Q[C,:] - logsumexp_dim0(Q[st,:]) 
                llik = llik + logp * (1. - feedback)   # Only add on NF trials    
            else:
                logp = np.nan
        else: #simulation of choices that are within the bin range
            if (st[0] >= min_stim) and (st[1] >= min_stim) and(feedback < .5):
                logp = Q[C,:] - logsumexp_dim0(Q[st,:]) 
        
                C_st = np.random.binomial(1, np.exp(logp[0]))
                C = st[C_st]           
                o = gen_outcome(s_dat_dec.p_Pun[t,C_st,n], s_dat_dec.p_R[t,C_st,n])
            
                s_dat_dec.C_st[t,n] = C_st
                s_dat_dec.C[t,n]    = C
                s_dat_dec.o[t,n]    = o
            else:
                C = s_dat_dec.C[t,n]
                o = s_dat_dec.o[t,n]
           
            
        # outcome counter:
        
        if o > .5:
            counter[0, C, :] = counter[0, C, :] + feedback  #counts rewards
        elif o < -.5:
            counter[1, C, :] = counter[1, C, :] + feedback #counts punishment
        else:
            counter[2, C, :] = counter[2, C, :] + feedback #counts neutral   
        
        # If return_Q, store the RPE:
        if return_Q and feedback > .5:
            RPE[t] = o - Q[C,0]               
        
       
        # Store relevant quantities for return_Q = True, for the first sample/particle:
        if return_Q:
            Q_traj[:,t] = Q[:,0]
            p_t[t]      = np.exp(logp)
            
    if simulate_data:
        s_dat_dec.C = s_dat_dec.C.astype('int32')
        s_dat_dec.C_st = s_dat_dec.C_st.astype('int32')
        s_dat_dec.o    = s_dat_dec.o.astype('int32')
        s_dat_dec.compute_acc()
    
    if return_Q:
        return llik, (Q_traj, None, None, p_t, RPE, None)
    else:
        return llik, None
            
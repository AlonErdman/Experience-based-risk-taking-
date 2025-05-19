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
    
def llik_learning_num_no_inver_nf_bins_zero_counters_fmri(P, P_aux, n, s_dat_lea, return_Q = False, simulate_data = False):
    """
    This is an example of a complete llik function to be used as a custom llik function in the fitting algorithm. 
    In general, custom llik functions need to take as:
    
    Arguments:
    1. P: (S, .) DataFrame of parameter values
    2. P_aux: (S, .) DataFrame of auxiliary parameter values.
    3. n: int, the subject index to fit on
    4. s_dat_lea: subjects data class object, an object of the specific class created to store your data in. You use state information from this class, preferably used as numpy arrays for speed (much faster than DataFrames), 
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
    Nc = s_dat_lea.Nc[n] #Number of stimuli. If there is an fmri block, then the index of stim=-1000 in Q the last one (Nc)
    T  =  s_dat_lea.T[n] #number of trials
    min_stim = s_dat_lea.min_stim #the first non-pratice stimulus 
        
    ## Initialization of the latent process:
    Q = np.zeros((Nc, S)) #Q values of each stimulus
    
    ## Initialization of the Q trajectory to return of Q_traj = True:
    Q_traj = np.zeros((Nc, T))
    p_t    = np.zeros((T))
    RPE    = np.full((T), np.nan)
    Q_old = np.full((T), np.nan)
    Q_unchosen = np.full((T), np.nan)
    Q_new = np.full((T), np.nan)
    delta_Q_traj = np.full((T), np.nan)
    
    #Initialization:
    llik = np.zeros((S))
    old_session = -0.5 #sessions without bins equal minus 1, so this step assures only sessions=>0 will get paramters.
    session     = -0.5
    R_r         = np.nan
    R_pun       = np.nan
    
    if simulate_data:
        T = s_dat_lea.T[n];
        s_dat_lea.C[:,n]      = s_dat_lea.C[:,n]   #C-number of the chosen stimulus (e.g., 7). Trials that were not played get -1.  
        s_dat_lea.C_st[:,n]   = s_dat_lea.C_st[:,n] #C_st-0/1-choice of the left/right stimulus. Trials that were not played get -1. 
        s_dat_lea.o[:,n]      = s_dat_lea.o[:,n] #Trials that were not played get -2. 
        
    counter_num = np.zeros((3, Nc, S))
    counter_den = np.zeros((3, Nc, S))
    for t in range(T):
        # Extract the relevant state information:
        st = s_dat_lea.stims[t,:,n]
        if st[1]==-1000: #if fmri accept-reject trial: then change st[1] from -1000 to Nc-1 (relevant both for real and simulated data)
            st[1]=Nc-1 #minus 1 is because stim numbers start from 0
        if not simulate_data:
            C  = s_dat_lea.C[t,n]; C_st = s_dat_lea.C_st[t,n]
            o  = s_dat_lea.o[t,n]; C_not  = s_dat_lea.C_not[t,n]           
        else:
            C_st = 1 #arbitrary choice of 1. Afterwards the probability of this choice gets calculated, and then choice is redefined
            C = st[C_st]
            C_not = st[1-C_st]
        if C==-1000: #if the subject (or the simulation) chose stimulus -1000 (accept in the fmri), then change C to the last index of Q
           C=Nc-1
        if C_not==-1000: #if the subject (or the simulation) did not choose stimulus -1000 (reject in the fmri), then change C_not to the last index of Q
           C_not=Nc-1 
        feedback = s_dat_lea.feedback[t,n]
        
        # Update the parameters if necessary:
        session = s_dat_lea.session[t,n]
        if (session > old_session): 
            R_r         = P_aux[f'R_r_{session}'].values
            R_pun       = P_aux[f'R_pun_{session}'].values           
            old_session = session      
        
        if not simulate_data:
            # Compute the logprobabilityies of the choice                
            if (st[0] >= min_stim) and (st[1] >= min_stim) and (st[1] != Nc-1): # no practice trials outside the scanner
                logp = Q[C,:] - logsumexp_dim0(Q[st,:]) 
                llik = llik + logp * (1. - feedback)              # Only add on NF trials       (!!)
            elif (st[0] >= min_stim) and (st[1] == Nc-1):    #fmri no feedback trials
                Q[Nc-1,:] = (0.4 * R_r) + (0.4 * R_pun)  #Nc us the index of stimulus -1000 
                logp = Q[C,:] - logsumexp_dim0(Q[st,:]) 
                llik = llik + logp * (1. - feedback) 
            else:
                logp = np.nan
        else:
            if (st[0] >= min_stim) and (st[1] >= min_stim) and (st[1] != Nc-1) and (feedback < .5): # no practice +no feedback trials outside the scanner
                logp = Q[C,:] - logsumexp_dim0(Q[st,:]) 
        
                C_st = np.random.binomial(1, np.exp(logp[0]))
                C    = st[C_st]           
                o = gen_outcome(s_dat_lea.p_Pun[t,C_st,n], s_dat_lea.p_R[t,C_st,n])
            
                s_dat_lea.C_st[t,n] = C_st
                s_dat_lea.C[t,n]    = C
                s_dat_lea.o[t,n]    = o            
            elif (st[0] >= min_stim) and (st[1] == Nc-1): #accept reject trials
                Q[Nc-1,:] = (0.4 * R_r) + (0.4 * R_pun)
                logp = Q[C,:] - logsumexp_dim0(Q[st,:]) 
                C_st = np.random.binomial(1, np.exp(logp[0]))
                C    = st[C_st]
                o = gen_outcome(s_dat_lea.p_Pun[t,C_st,n], s_dat_lea.p_R[t,C_st,n])            
                s_dat_lea.C_st[t,n] = C_st
                s_dat_lea.C[t,n]    = C
                s_dat_lea.o[t,n]    = o
            else:
                C = s_dat_lea.C[t,n]
                o = s_dat_lea.o[t,n]
                
        if simulate_data and C==-1000: #if the stimulus equals -1000 (during simulation), then change C to the last index of Q
            C=Nc-1 
            
       
        #outcome counters (numerator=num; denominator=den):
        if o > .5: #if reward
            counter_num[0, C, :] = counter_num[0, C, :] + feedback * R_r #counts rewards +weight
            counter_den[0, C, :] = counter_den[0, C, :] + feedback #counts rewards 
        elif o < -.5: #if punishment
            counter_num[1, C, :] = counter_num[1, C, :] + feedback * R_pun #counts punishment +weight 
            counter_den[1, C, :] = counter_den[1, C, :] + feedback #counts punishment
        else:
            counter_den[2, C, :] = counter_den[2, C, :] + feedback #counts neutral
            
        # If return_Q, store the RPE:
        if return_Q:
            current_value = Q[C,0]
            current_value_unchosen = Q[C_not,0]
            Q_old[t] = current_value
            Q_unchosen[t] = current_value_unchosen
            if feedback > .5:
                if o> .5:
                    RPE[t] = R_r * o - current_value #Q[C,0]
                elif o < -.5:
                    RPE[t] = -R_pun * o - current_value #Q[C,0]
                else:
                    RPE[t] = o - current_value #Q[C,0]
            
            
        # Update Q only on F trials:
        reward_minus_pun = counter_num[0, C, :] + counter_num[1, C, :]
        sum_all = counter_den[0, C, :] + counter_den[1, C, :] + counter_den[2, C, :]
        if feedback >.5:
            Q[C,:] = np.divide(reward_minus_pun, sum_all)
        
        if return_Q and feedback > .5:        
            Q_new[t] = Q[C,0]
            delta_Q_traj[t] = Q[C,0] - current_value
        
        ## change Qs/RPEs/delta_Q to nan if involved pratice stimuli:
        
        if return_Q and (feedback < .5) and ((st[0] < min_stim) or (st[1] < min_stim)): #assign nan for Qs in NF trials with pratice stimuli
            Q_old[t] = np.nan
            Q_unchosen[t] = np.nan
            
        if return_Q and (feedback > .5) and (C < min_stim): #assign nan for Q, RPEs and delta_Q in F trials where the chosen stimulus was learned during the pratice session:
            Q_old[t] = np.nan
            Q_new[t] = np.nan
            RPE[t] = np.nan
            delta_Q_traj[t] = np.nan
        if return_Q and (feedback > .5) and (C_not < min_stim): #assign nan for Q of the unchosen stimulus if it was learned during the pratice session
            Q_unchosen[t] = np.nan
        
        
        # Store relevant quantities for return_Q = True, for the first sample/particle:
        if return_Q:
            Q_traj[:,t] = Q[C,0] #Q[:,0] 
            p_t[t]      = np.exp(logp)
            
    if simulate_data:
        s_dat_lea.C = s_dat_lea.C.astype('int32')
        s_dat_lea.C_st = s_dat_lea.C_st.astype('int32')
        s_dat_lea.o    = s_dat_lea.o.astype('int32')
        s_dat_lea.compute_acc()
    
    if return_Q:
        return llik, (Q_traj, None, None, p_t, RPE, None, delta_Q_traj, Q_old, Q_new, Q_unchosen)
    else:
        return llik, None
            
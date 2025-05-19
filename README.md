# Experience-based-risk-taking-
Part 1: The follwing code files used for the logestic regression analyses:

a. one_category.R: run the model that is based on learning time of risy image/safe image/decision time
b. three_category.R: run the joint model that is based on both learning time of risy image/safe image and decision time
c. one_caegory.stan-the one category model
d. three_category.stan-three category model
e. code.R: loading data

Part 2: Reinforcement-learning modelling:
Installation guide for RL modelling (Python-based):
On Windows, it’s recommended to access Python and all its libraries through Anaconda. You can get the latest relevant version here:

https://www.anaconda.com/products/individual
The EMA environment
Within Anaconda, we’ll create virtual environments in which we can install all the relevant libraries for different projects. Virtual environments just make sure that the versions of all the libraries are attuned to one another. Unlike R and MATLAB, there’s no assurance all versions of all libraries work with each other. We’ll create one environment for the main code, and then another for the MCMC package for hierarchical bayesian models later (“hlb environment” below).

After installing Anaconda, open “Anaconda Prompt” from the task bar:


Then a black terminal will open up. (‘dir’ and ‘cd’ can be used to locate and switch to folders).

Then:

conda create -n EMA

conda activate EMA
conda install python pandas numpy matplotlib plotly h5py scipy scikit-learn statsmodels jupyterlab pytables

Then, for either, run:

pip install jupyterlab "ipywidgets>=7.5"

jupyter labextension install jupyterlab-plotly@4.14.3

conda install -c plotly plotly-orca==1.3.1 psutil


To test the environment, open `Anaconda navigator`, and select the virtual environment ‘EMA’ in the drop down menu:

Then install jupyterlab if it’s not installed yet, and open it.


How to run RL modelling?
Open risk_fitting_time_based_bins-modeling_example.ipynb
this notebook file will guide you how to run the compuatationl modelling. 
You also need to download 'RLfcts_article', which include all functions used for the analysis, and 'decision_num_no_inver_bins'
and 'learning_num_no_inver_bins_zero_counters', which are the likelihood function of the two main models. 
The zip file include datasets of 5 participants to test the code on. 

Part 3:
mixed models: the file 'mixed_model_code.R' includes the stan-glmer mixed models.This code reads a few csv file that are attached in the zip file.




predictor1 = 'stimrisky_learning_bin'
predictor2 = 'stimsafe_learning_bin'
predictor3 = 'decision_bin'
model_name = 'three_category_riskyplussafe'
group = 'group1'   # e.g., 'group1' 
simulated_data = '' #data file name
iteration_suffix = '' # e.g. '_b' or '_c'

library(rstan)
library(bridgesampling)
library(HDInterval)
options(mc.cores = 5)
source("G:/My Drive/Projects/Variations in risk taking/analysis_code/code.R")

if (simulated_data=='') {
  all_data = analysis(3, subpath = group)
  all_trials = all_data$all_trials
} else {
  load(paste0("G:/My Drive/Projects/Variations in risk taking/simulated_data/",simulated_data,".RData"))
  all_trials = eval(parse(text = group))
  subjects = unique(all_trials$subject)
  all_data = analysis(3, subjects = subjects, subpath = group)
}

all_bins_learning = all_data$all_bins
all_bins_decision = all_data$all_bins

# preparing data
bin11 = all_trials[,"stimrisky_bin1"]
bin12 = all_trials[,"stimrisky_bin2"]
bin11_p = all_trials[,"stimrisky_bin1_p"]
bin12_p = all_trials[,"stimrisky_bin2_p"]
bin21 = all_trials[,"stimsafe_bin1"]
bin22 = all_trials[,"stimsafe_bin2"]
bin21_p = all_trials[,"stimsafe_bin1_p"]
bin22_p = all_trials[,"stimsafe_bin2_p"]
bin3 = all_trials[,predictor3]
subject = as.numeric(factor(all_trials$subject))
bin_subject_learning = as.numeric(factor(all_bins_learning$subject))
bin_subject_decision = as.numeric(factor(all_bins_decision$subject))
n_bins_leanring = max(all_bins_learning$bin)
n_bins_decision = max(all_bins_decision$bin)
n_subject = length(unique(subject))
n = nrow(all_trials)

# remove unnecessary bin-subject combinations
removed = rep(0, n_subject)
for (s in 1:n_subject){
  for (b in 1:n_bins_leanring) {
    if (removed[s]+b>n_bins_leanring) break
    while (sum(subject==s & (bin11==b & bin11_p>0 | bin12==b & bin12_p>0 | bin21==b & bin21_p>0 | bin22==b & bin22_p>0))==0){
      bin11[subject==s & bin11>b] = bin11[subject==s & bin11>b] - 1
      bin12[subject==s & bin12>b] = bin12[subject==s & bin12>b] - 1
      bin21[subject==s & bin21>b] = bin21[subject==s & bin21>b] - 1
      bin22[subject==s & bin22>b] = bin22[subject==s & bin22>b] - 1
      if (sum(bin_subject_learning==s & all_bins_learning$bin==b)>0){
        to_remove = which(bin_subject_learning==s & all_bins_learning$bin==b) 
        bin_subject_learning = bin_subject_learning[-to_remove]
        all_bins_learning = all_bins_learning[-to_remove,]
      }
      all_bins_learning$bin[bin_subject_learning==s & all_bins_learning$bin>b] = all_bins_learning$bin[bin_subject_learning==s & all_bins_learning$bin>b] - 1
      removed[s] = removed[s] + 1
      if (removed[s]+b>n_bins_leanring) break
    }
  }
}

# which subject each bin belongs to
bin_b_subject_learning = c() 
for (s in 1:n_subject){
  bin_b_subject_learning = c(bin_b_subject_learning, rep(s,n_bins_leanring-removed[s]));
}

# which bin each trial belongs to
trial_bin11_learning_b=c()
trial_bin12_learning_b=c()
trial_bin21_learning_b=c()
trial_bin22_learning_b=c()
trial_bin11_learning_b_p=c()
trial_bin12_learning_b_p=c()
trial_bin21_learning_b_p=c()
trial_bin22_learning_b_p=c()
for (i in 1:n){
  trial_bin11_learning_b[i] = (subject[i]-1)*n_bins_leanring + bin11[i] - sum(removed[1:(subject[i]-1)])*(subject[i]>1);
  trial_bin12_learning_b[i] = (subject[i]-1)*n_bins_leanring + bin12[i] - sum(removed[1:(subject[i]-1)])*(subject[i]>1);
  trial_bin21_learning_b[i] = (subject[i]-1)*n_bins_leanring + bin21[i] - sum(removed[1:(subject[i]-1)])*(subject[i]>1);
  trial_bin22_learning_b[i] = (subject[i]-1)*n_bins_leanring + bin22[i] - sum(removed[1:(subject[i]-1)])*(subject[i]>1);
  trial_bin11_learning_b_p[i] = bin11_p[i];
  trial_bin12_learning_b_p[i] = bin12_p[i];
  trial_bin21_learning_b_p[i] = bin21_p[i];
  trial_bin22_learning_b_p[i] = bin22_p[i];
}

# remove unnecessary bin-subject combinations
removed = rep(0, n_subject)
for (s in 1:n_subject){
  for (b in 1:n_bins_decision) {
    if (removed[s]+b>n_bins_decision) break
    while (sum(subject==s & bin3==b)==0){
      bin3[subject==s & bin3>b] = bin3[subject==s & bin3>b] - 1
      if (sum(bin_subject_decision==s & all_bins_decision$bin==b)>0){
        to_remove = which(bin_subject_decision==s & all_bins_decision$bin==b) 
        bin_subject_decision = bin_subject_decision[-to_remove]
        all_bins_decision = all_bins_decision[-to_remove,]
      }
      all_bins_decision$bin[bin_subject_decision==s & all_bins_decision$bin>b] = all_bins_decision$bin[bin_subject_decision==s & all_bins_decision$bin>b] - 1
      removed[s] = removed[s] + 1
      if (removed[s]+b>n_bins_decision) break
    }
  }
}

# which subject each bin belongs to
bin_b_subject_decision = c() 
for (s in 1:n_subject){
  bin_b_subject_decision = c(bin_b_subject_decision, rep(s,n_bins_decision-removed[s]));
}

# which bin each trial belongs to
trial_bin_decision_b=c()
for (i in 1:n){
  trial_bin_decision_b[i] = (subject[i]-1)*n_bins_decision + bin3[i] - sum(removed[1:(subject[i]-1)])*(subject[i]>1);
}

data = list(
  n = n,
  n_subject = n_subject,
  n_bins_learning = length(bin_b_subject_learning),
  n_bins_decision = length(bin_b_subject_decision),
  choice = all_trials$risky_choice,
  subject = subject,
  avg_outcome = all_trials$avg_outcome,
  avg_accuracy = all_trials$avg_accuracy,
  bin_b_subject_learning = bin_b_subject_learning,
  bin_b_subject_decision = bin_b_subject_decision,
  trial_bin11_learning_b = trial_bin11_learning_b,
  trial_bin12_learning_b = trial_bin12_learning_b,
  trial_bin21_learning_b = trial_bin21_learning_b,
  trial_bin22_learning_b = trial_bin22_learning_b,
  trial_bin11_learning_b_p = trial_bin11_learning_b_p,
  trial_bin12_learning_b_p = trial_bin12_learning_b_p,
  trial_bin21_learning_b_p = trial_bin21_learning_b_p,
  trial_bin22_learning_b_p = trial_bin22_learning_b_p,
  trial_bin_decision_b = trial_bin_decision_b
)

# Running stan code
cat("Parsing model... ")
model = stan_model(paste0(model_name,".stan"))
cat("Done\n")
cat("Sampling from model... ")
fit = sampling(model,data,iter=45000,warmup=5000, thin=4, chains=5, control = list(adapt_delta = 0.99, max_treedepth = 12), pars = c("bin_sd_learning", "bin_sd_learning_subject_risky", "bin_sd_learning_subject_safe", "bin_sd_decision", "bin_learning_b_risky", "bin_learning_b_safe", "bin_decision_b", "avg_outcome_b_subject", "avg_accuracy_b_subject"), include = FALSE)
cat("Done\n")
#print(fit)
params = extract(fit)
#hdi((params$bin_var_learning_vs_decision_alpha_minus1+1)/(params$bin_var_learning_vs_decision_alpha_minus1+params$bin_var_learning_vs_decision_beta_minus1+2))
#hdi((params$bin_var_risky_vs_safe_alpha_minus1+1)/(params$bin_var_risky_vs_safe_alpha_minus1+params$bin_var_risky_vs_safe_beta_minus1+2))
#hdi(params$bin_sd_sum_shape/params$bin_sd_sum_rate)

# saving
if (simulated_data!='') {
  simulation_prefix = paste0(simulated_data,'_')
} else {simulation_prefix = ''}
save(fit, model, predictor1, predictor2, predictor3, data, all_trials, file = paste0("G:/My Drive/Projects/Variations in risk taking/analysis_results/",simulation_prefix, group,'_', model_name, iteration_suffix, ".RData"))

# Computing model evidence
bridge <- bridge_sampler(fit, silent = TRUE)
print(bridge)

# add fitted parameter to all_bins
params$bin_sd_learning = sqrt(params$bin_var_learning_vs_decision_subject) * params$bin_sd_sum_subject
params$bin_sd_decision = sqrt(1-params$bin_var_learning_vs_decision_subject) * params$bin_sd_sum_subject;
params$bin_learning_b = params$bin_learning_b_unscaled * params$bin_sd_learning[data$bin_b_subject_learning]
params$bin_decision_b = params$bin_decision_b_unscaled * params$bin_sd_decision[data$bin_b_subject_decision]

subject_names = levels(factor(all_trials$subject))
all_bins_learning$risk_seeking = NA
for (i in 1:nrow(all_bins_learning)){
  subject_name = subject_names[data$bin_b_subject_learning[i]]
  bin_num = sum(which(data$bin_b_subject_learning[i]==data$bin_b_subject_learning)<=i)
  index = which(all_bins_learning$subject==subject_name)[bin_num]
  all_bins_learning$risk_seeking[index] = mean(params$overall_mean + params$subject_b[,data$bin_b_subject_learning[i]] + params$bin_learning_b[,i] + mean(params$bin_decision_b[,data$bin_b_subject_decision==data$bin_b_subject_learning[i]]))
}
all_bins_decision$risk_seeking = NA
for (i in 1:nrow(all_bins_decision)){
  subject_name = subject_names[data$bin_b_subject_decision[i]]
  bin_num = sum(which(data$bin_b_subject_decision[i]==data$bin_b_subject_decision)<=i)
  index = which(all_bins_decision$subject==subject_name)[bin_num]
  all_bins_decision$risk_seeking[index] = mean(params$overall_mean + params$subject_b[,data$bin_b_subject_decision[i]] + params$bin_decision_b[,i] + mean(params$bin_learning_b[,data$bin_b_subject_learning==data$bin_b_subject_decision[i]]))
}

# saving
save(fit, model, bridge, predictor1, predictor2, predictor3, data, all_trials, all_bins_decision, all_bins_learning, file = paste0("G:/My Drive/Projects/Variations in risk taking/analysis_results/",simulation_prefix, group,'_', model_name, iteration_suffix, ".RData"))


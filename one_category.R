predictor = 'stimrisky_learning_bin' # 'stimrisky_learning_bin' or 'stimsafe_learning_bin' or 'decision_bin'
model_name = 'one_category'
group = 'group1' # 'group1' or 'group2' or name of simulation  
simulated_data = '' #data file name

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

all_bins = all_data$all_bins

# preparing data
if (predictor == 'stimrisky_learning_bin'){
  bin1 = all_trials[,"stimrisky_bin1"]
  bin2 = all_trials[,"stimrisky_bin2"]
  bin1_p = all_trials[,"stimrisky_bin1_p"]
  bin2_p = all_trials[,"stimrisky_bin2_p"]
} else if (predictor == 'stimsafe_learning_bin'){
  bin1 = all_trials[,"stimsafe_bin1"]
  bin2 = all_trials[,"stimsafe_bin2"]
  bin1_p = all_trials[,"stimsafe_bin1_p"]
  bin2_p = all_trials[,"stimsafe_bin2_p"]
} else {
  bin1 = all_trials[,predictor]
  bin1_p = rep(1, length(bin1))
  bin2 = rep(1, length(bin1))
  bin2_p = rep(0, length(bin1))
}


subject = as.numeric(factor(all_trials$subject))
bin_subject = as.numeric(factor(all_bins$subject))
n_bins = max(all_bins$bin)
n_subject = length(unique(subject))
n = nrow(all_trials)

# remove unnecessary bin-subject combinations
removed = rep(0, n_subject)
for (s in 1:n_subject){
  for (b in 1:n_bins) {
    if (removed[s]+b>n_bins) break
    while (sum(subject==s & bin1==b & bin1_p>0)==0 && sum(subject==s & bin2==b & bin2_p>0)==0){
      bin1[subject==s & bin1>b] = bin1[subject==s & bin1>b] - 1
      bin2[subject==s & bin2>b] = bin2[subject==s & bin2>b] - 1
      if (sum(bin_subject==s & all_bins$bin==b)>0){
        to_remove = which(bin_subject==s & all_bins$bin==b) 
        bin_subject = bin_subject[-to_remove]
        all_bins = all_bins[-to_remove,]
      }
      all_bins$bin[bin_subject==s & all_bins$bin>b] = all_bins$bin[bin_subject==s & all_bins$bin>b] - 1
      removed[s] = removed[s] + 1
      if (removed[s]+b>n_bins) break
    }
  }
}

# which subject each bin belongs to
bin_b_subject = c() 
for (s in 1:n_subject){
  bin_b_subject = c(bin_b_subject, rep(s,n_bins-removed[s]));
}

# which bin each trial belongs to
trial_bin1_b=c()
trial_bin2_b=c()
trial_bin1_b_p=c()
trial_bin2_b_p=c()
for (i in 1:n){
  trial_bin1_b[i] = (subject[i]-1)*n_bins + bin1[i] - sum(removed[1:(subject[i]-1)])*(subject[i]>1);
  trial_bin2_b[i] = (subject[i]-1)*n_bins + bin2[i] - sum(removed[1:(subject[i]-1)])*(subject[i]>1);
  trial_bin1_b_p[i] = bin1_p[i];
  trial_bin2_b_p[i] = bin2_p[i];
}


data = list(
  n = n,
  n_subject = n_subject,
  n_bins = length(bin_b_subject),
  choice = all_trials$risky_choice,
  subject = subject,
  avg_outcome = all_trials$avg_outcome,
  bin_b_subject = bin_b_subject,
  trial_bin1_b = trial_bin1_b,
  trial_bin2_b = trial_bin2_b,
  trial_bin1_b_p = trial_bin1_b_p,
  trial_bin2_b_p = trial_bin2_b_p,
  avg_accuracy = all_trials$avg_accuracy
)

# Running stan code
cat("Parsing model... ")
model = stan_model(paste0(model_name,".stan"))
cat("Done\n")
cat("Sampling from model... ")
fit = sampling(model,data,iter=42500,warmup=2500, thin=4, chains=5, control = list(adapt_delta = 0.99, max_treedepth = 12), pars = c("bin_b_scaled", "avg_outcome_b_subject", "avg_accuracy_b_subject"), include = FALSE)
#print(fit)
cat("Done\n")
params = extract(fit)
#hdi(params$subject_b_sd)
#hdi(params$bin_sd_sum_shape*params$bin_sd_sum_rate)

# saving
if (simulated_data!='') {
  simulation_prefix = paste0(simulated_data,'_')
} else {simulation_prefix = ''}
save(fit, model, data, predictor, all_trials, all_bins, file = paste0("G:/My Drive/Projects/Variations in risk taking/analysis_results/",simulation_prefix,group,'_', predictor,".RData"))

# Computing model evidence
bridge <- bridge_sampler(fit, silent = TRUE)
print(bridge)

# add fitted parameter to all_bins
params$bin_b_scaled = params$bin_b * params$bin_sd_sum_subject[data$bin_b_subject];
subject_names = levels(factor(all_trials$subject))
all_bins$risk_seeking = NA
for (i in 1:nrow(all_bins)){
  subject_name = subject_names[data$bin_b_subject[i]]
  bin_num = sum(which(data$bin_b_subject[i]==data$bin_b_subject)<=i)
  index = which(all_bins$subject==subject_name)[bin_num]
  all_bins$risk_seeking[index] = mean(params$overall_mean + params$subject_b[,data$bin_b_subject[i]] + params$bin_b_scaled[,i])
}

# saving
save(fit, model, bridge, data, predictor, all_trials, all_bins, file = paste0("G:/My Drive/Projects/Variations in risk taking/analysis_results/",simulation_prefix,group,'_', predictor,".RData"))


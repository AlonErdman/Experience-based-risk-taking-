data {
  int n;
  int n_subject;
  int n_bins_learning;
  int n_bins_decision;
  int<lower=0, upper=1> choice[n];
  int subject[n];
  vector[n] avg_outcome;
  vector[n] avg_accuracy;
  int bin_b_subject_learning[n_bins_learning];
  int bin_b_subject_decision[n_bins_decision];
  int trial_bin11_learning_b[n];
  int trial_bin12_learning_b[n];
  int trial_bin21_learning_b[n];
  int trial_bin22_learning_b[n];
  vector[n] trial_bin11_learning_b_p;
  vector[n] trial_bin12_learning_b_p;
  vector[n] trial_bin21_learning_b_p;
  vector[n] trial_bin22_learning_b_p;
  int trial_bin_decision_b[n];
}

parameters {
  real<lower=0> bin_var_learning_vs_decision_alpha_minus1;
  real<lower=0> bin_var_risky_vs_safe_alpha_minus1;
  real<lower=0> bin_var_learning_vs_decision_beta_minus1;
  real<lower=0> bin_var_risky_vs_safe_beta_minus1;
  vector<lower=0,upper=1>[n_subject] bin_var_learning_vs_decision_subject;
  vector<lower=0,upper=1>[n_subject] bin_var_risky_vs_safe_subject;
  
  real<lower=0> bin_sd_sum_shape;
  real<lower=0> bin_sd_sum_rate;
  vector<lower=0>[n_subject] bin_sd_sum_subject;

  real overall_mean;
  real avg_outcome_b;
  real avg_accuracy_b;
  real<lower=0> subject_b_sd;
  real<lower=0> avg_outcome_b_subject_sd;
  real<lower=0> avg_accuracy_b_subject_sd;
  vector[n_subject] subject_b;
  vector[n_subject] avg_outcome_b_subject_deviation;
  vector[n_subject] avg_accuracy_b_subject_deviation;
  
  vector[n_bins_learning] bin_learning_b_unscaled;
  vector[n_bins_decision] bin_decision_b_unscaled;
}

transformed parameters{
  vector<lower=0>[n_subject] bin_sd_learning = sqrt(bin_var_learning_vs_decision_subject) .* bin_sd_sum_subject;
  vector<lower=0>[n_subject] bin_sd_learning_subject_risky = bin_sd_learning .* sqrt(bin_var_risky_vs_safe_subject);
  vector<lower=0>[n_subject] bin_sd_learning_subject_safe = bin_sd_learning .* sqrt(1 - bin_var_risky_vs_safe_subject);
  vector<lower=0>[n_subject] bin_sd_decision = sqrt(1-bin_var_learning_vs_decision_subject) .* bin_sd_sum_subject;
  vector[n_bins_learning] bin_learning_b_risky = bin_learning_b_unscaled .* bin_sd_learning_subject_risky[bin_b_subject_learning];
  vector[n_bins_learning] bin_learning_b_safe = bin_learning_b_unscaled .* bin_sd_learning_subject_safe[bin_b_subject_learning];
  vector[n_bins_decision] bin_decision_b = bin_decision_b_unscaled .* bin_sd_decision[bin_b_subject_decision];
  vector[n_subject] avg_outcome_b_subject = avg_outcome_b + avg_outcome_b_subject_deviation * avg_outcome_b_subject_sd;
  vector[n_subject] avg_accuracy_b_subject = avg_accuracy_b + avg_accuracy_b_subject_deviation * avg_accuracy_b_subject_sd;
}

model {
  target += bernoulli_logit_lpmf(choice | overall_mean + subject_b[subject]*subject_b_sd + bin_decision_b[trial_bin_decision_b] + bin_learning_b_risky[trial_bin11_learning_b].*trial_bin11_learning_b_p + bin_learning_b_risky[trial_bin12_learning_b].*trial_bin12_learning_b_p - bin_learning_b_safe[trial_bin21_learning_b].*trial_bin21_learning_b_p - bin_learning_b_safe[trial_bin22_learning_b].*trial_bin22_learning_b_p + avg_outcome_b_subject[subject] .* avg_outcome  + avg_accuracy_b_subject[subject] .* avg_accuracy);
  
  target += normal_lpdf(overall_mean | 0, 5);
  
  target += normal_lpdf(subject_b | 0, 1);
  target += gamma_lpdf (subject_b_sd | 1, 1);
  
  target += normal_lpdf(bin_learning_b_unscaled |0, 1);
  target += normal_lpdf(bin_decision_b_unscaled |0, 1);

  target += gamma_lpdf(bin_var_learning_vs_decision_alpha_minus1 |1,.01);
  target += gamma_lpdf(bin_var_learning_vs_decision_beta_minus1 |1,.01);
  target += beta_lpdf(bin_var_learning_vs_decision_subject |bin_var_learning_vs_decision_alpha_minus1+1, bin_var_learning_vs_decision_beta_minus1+1);
  
  target += gamma_lpdf(bin_var_risky_vs_safe_alpha_minus1 |1,.01);
  target += gamma_lpdf(bin_var_risky_vs_safe_beta_minus1 |1,.01);
  target += beta_lpdf(bin_var_risky_vs_safe_subject |bin_var_risky_vs_safe_alpha_minus1+1, bin_var_risky_vs_safe_beta_minus1+1);
  
  target += gamma_lpdf(bin_sd_sum_shape |1, 1);
  target += gamma_lpdf(bin_sd_sum_rate |1, 1);
  target += gamma_lpdf(bin_sd_sum_subject |bin_sd_sum_shape,bin_sd_sum_rate);
  
  target += normal_lpdf(avg_outcome_b |0, 5);
  target += normal_lpdf(avg_outcome_b_subject_deviation |0,1);
  target += gamma_lpdf(avg_outcome_b_subject_sd |1, 1);
  
  target += normal_lpdf(avg_accuracy_b |0, 5);
  target += normal_lpdf(avg_accuracy_b_subject_deviation |0,1);
  target += gamma_lpdf(avg_accuracy_b_subject_sd |1, 1);

}

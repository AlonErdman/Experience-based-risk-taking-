data {
  int n;
  int n_subject;
  int n_bins;
  int<lower=0, upper=1> choice[n];
  int subject[n];
  vector[n] avg_outcome;
  vector[n] avg_accuracy;
  int bin_b_subject[n_bins];
  int trial_bin1_b[n];
  int trial_bin2_b[n];
  vector[n] trial_bin1_b_p;
  vector[n] trial_bin2_b_p;
}

parameters {

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
  
  vector[n_bins] bin_b;
}

transformed parameters{
  vector[n_bins] bin_b_scaled = bin_b .* bin_sd_sum_subject[bin_b_subject];
  vector[n_subject] avg_outcome_b_subject = avg_outcome_b + avg_outcome_b_subject_deviation * avg_outcome_b_subject_sd;
  vector[n_subject] avg_accuracy_b_subject = avg_accuracy_b + avg_accuracy_b_subject_deviation * avg_accuracy_b_subject_sd;
}

model {
  target += bernoulli_logit_lpmf(choice | overall_mean + subject_b[subject]*subject_b_sd + bin_b_scaled[trial_bin1_b].*trial_bin1_b_p + bin_b_scaled[trial_bin2_b].*trial_bin2_b_p  + avg_outcome_b_subject[subject] .* avg_outcome  + avg_accuracy_b_subject[subject] .* avg_accuracy);
  
  target += normal_lpdf(overall_mean | 0, 5);
  
  target += normal_lpdf(subject_b | 0, 1);
  target += gamma_lpdf (subject_b_sd | 1, 1);
  
  target += normal_lpdf(bin_b | 0, 1);

  target += gamma_lpdf(bin_sd_sum_shape |1,1);
  target += gamma_lpdf(bin_sd_sum_rate |1,1);
  target += gamma_lpdf(bin_sd_sum_subject |bin_sd_sum_shape,bin_sd_sum_rate);
  
  target += normal_lpdf(avg_outcome_b |0, 5);
  target += normal_lpdf(avg_outcome_b_subject_deviation |0,1);
  target += gamma_lpdf(avg_outcome_b_subject_sd |1, 1);
  
  target += normal_lpdf(avg_accuracy_b |0, 5);
  target += normal_lpdf(avg_accuracy_b_subject_deviation |0,1);
  target += gamma_lpdf(avg_accuracy_b_subject_sd |1, 1);

}

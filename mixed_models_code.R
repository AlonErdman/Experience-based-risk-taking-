rm(list = ls())
library("bayesplot")
library("rstanarm")
library("lme4")
library("dplyr")
library("ggplot2")
library("lmerTest")
library('gridExtra')
library('tidyr')
library('ggdist')
library('see')
library('bayesplot')
library('gridExtra')
library('grid')
library('gtable')
library('HDInterval')
library('coin')

#risk seeking lea decision/lea weights corr:
first_sample_risk_dec_lea= read.csv('G:/My Drive/Lab/risk_seeking_article/risk_seeking_lea_dec_lea_weights/test_risk_lea_weights_dec_lea_first_sample.csv')
second_sample_risk_dec_lea= read.csv('G:/My Drive/Lab/risk_seeking_article/risk_seeking_lea_dec_lea_weights/test_risk_lea_weights_dec_lea_second_sample.csv')
first_sample_risk_dec_lea$log_abs_pun=log(abs(first_sample_risk_dec_lea$mean_P_lea))
first_sample_risk_dec_lea$log_R=log(abs(first_sample_risk_dec_lea$mean_R_lea))
second_sample_risk_dec_lea$log_abs_pun=log(abs(second_sample_risk_dec_lea$mean_P_lea))
second_sample_risk_dec_lea$log_R=log(abs(second_sample_risk_dec_lea$mean_R_lea))
#switch rates:
first_sample_switch= read.csv('G:/My Drive/Lab/risk_seeking_article/perm_test_for_switch/switch_first.csv')
#second_sample_switch= read.csv('G:/My Drive/Lab/risk_seeking_article/perm_test_for_switch/switch_second.csv')

first_sample_EEG= read.csv('E:/EEG_HR_new_scripts/mixed/ERP_downs_smoothed_withbaseline/sample1_ERP_smoothed_withbaseline_200pre200post_down128_filtered.csv')
second_sample_EEG= read.csv('E:/EEG_HR_new_scripts/mixed/ERP_downs_smoothed_withbaseline/sample2_ERP_smoothed_withbaseline_200pre200post_down128_filtered.csv')
first_sample_HR= read.csv('E:/EEG_HR_new_scripts/mixed/HR/smoothed_withbaseline_for_article/sample1_HR_smoothed_with_baseline_20hz.csv')
second_sample_HR= read.csv('E:/EEG_HR_new_scripts/mixed/HR/smoothed_Withbaseline_for_article/sample2_HR_smoothed_with_baseline_20hz.csv')
#screen subject with bad EEG signal:
first_sample_EEG <- first_sample_EEG[!(first_sample_EEG$subject %in% c(308)), ] #only 63%
second_sample_EEG <- second_sample_EEG[!(second_sample_EEG$subject %in% c(415 , 432)), ] #both <30%
#screen subject with bad HR signal:
second_sample_HR <- second_sample_HR[!(second_sample_HR$subject %in% c(415,432)), ] #all <60%
#second_sample_HR <- second_sample_HR[!(second_sample_HR$subject %in% c(415,432,433)), ] #all <60%

first_sample_HR$log_abs_pun=log(abs(first_sample_HR$mean_P_lea))
second_sample_HR$log_abs_pun=log(abs(second_sample_HR$mean_P_lea))
first_sample_HR$log_R=log(first_sample_HR$mean_R_lea)
second_sample_HR$log_R=log(second_sample_HR$mean_R_lea)
first_sample_HR$decodability_negative_PE_scaled = scale(first_sample_HR$decodability_negative_PE)  
second_sample_HR$decodability_negative_PE_scaled = scale(second_sample_HR$decodability_negative_PE) 
first_sample_HR$decodability_positive_PE_scaled = scale(first_sample_HR$decodability_positive_PE)  
second_sample_HR$decodability_positive_PE_scaled = scale(second_sample_HR$decodability_positive_PE)
first_sample_EEG$log_abs_pun=log(abs(first_sample_EEG$mean_P_lea))
second_sample_EEG$log_abs_pun=log(abs(second_sample_EEG$mean_P_lea))
first_sample_EEG$log_R=log(first_sample_EEG$mean_R_lea)
second_sample_EEG$log_R=log(second_sample_EEG$mean_R_lea)
first_sample_EEG$decodability_negative_PE_scaled = scale(first_sample_EEG$decodability_negative_PE)  
second_sample_EEG$decodability_negative_PE_scaled = scale(second_sample_EEG$decodability_negative_PE)  
first_sample_EEG$decodability_positive_PE_scaled = scale(first_sample_EEG$decodability_positive_PE)  
second_sample_EEG$decodability_positive_PE_scaled = scale(second_sample_EEG$decodability_positive_PE) 

both_samples_EEG=rbind(first_sample_EEG,second_sample_EEG)
both_samples_HR=rbind(first_sample_HR,second_sample_HR)


#first panel-risk seeking-log asym lea corr:
model_risk_asym_dec_lea_first <- stan_glmer(risk_seeking_lea ~ mean_asym_lea + (mean_asym_lea | subject), data = first_sample_risk_dec_lea, iter = 5000)
model_risk_asym__dec_lea_second <- stan_glmer(risk_seeking_lea ~ mean_asym_lea + (mean_asym_lea | subject), data = second_sample_risk_dec_lea, iter = 5000)
#model_risk_lea_asym_lea_first <- readRDS("G:/My Drive/Lab/risk_seeking_article/stan_results/model_risk_lea_asym_lea_first.rds")
#model_risk_lea_asym_lea_second<- readRDS("G:/My Drive/Lab/risk_seeking_article/stan_results/model_risk_lea_asym_lea_second.rds")
posterior_interval(model_risk_lea_asym_lea_first, level = 0.95)
posterior_interval(model_risk_lea_asym_lea_second, level = 0.95)
mean(bayes_R2(model_risk_lea_asym_lea_first))
mean(bayes_R2(model_risk_lea_asym_lea_second))


##second panel-histogram corr:
# Calculate within-subject correlation
#within_subject_cor <- list()
cor_risk_seeking_lea_lea_first <- sapply(unique(first_sample_risk_dec_lea$subject), function(sub) {
  subset_data <- first_sample_risk_dec_lea[first_sample_risk_dec_lea$subject == sub, ]
  correlation_value <- cor(subset_data$mean_asym_lea,
                           subset_data$risk_seeking_lea, use="complete.obs")
  return(correlation_value)
})

cor_risk_seeking_lea_lea_second <- sapply(unique(second_sample_risk_dec_lea$subject), function(sub) {
  subset_data <- second_sample_risk_dec_lea[second_sample_risk_dec_lea$subject == sub, ]
  correlation_value <- cor(subset_data$mean_asym_lea,
                           subset_data$risk_seeking_lea, use="complete.obs")
  return(correlation_value)
})

# Calculate within-subject correlation for 'risk_seeking_lea' and 'mean_asym_dec'
cor_risk_seeking_lea_dec_first <- sapply(unique(first_sample_risk_dec_lea$subject), function(sub) {
  subset_data <- first_sample_risk_dec_lea[first_sample_risk_dec_lea$subject == sub, ]
  correlation_value <- cor(subset_data$risk_seeking_lea,
                           subset_data$mean_asym_dec, use="complete.obs")
  return(correlation_value)
})

# Calculate within-subject correlation for 'risk_seeking_lea' and 'mean_asym_dec'
cor_risk_seeking_lea_dec_second <- sapply(unique(second_sample_risk_dec_lea$subject), function(sub) {
  subset_data <- second_sample_risk_dec_lea[second_sample_risk_dec_lea$subject == sub, ]
  correlation_value <- cor(subset_data$risk_seeking_lea,
                           subset_data$mean_asym_dec, use="complete.obs")
  return(correlation_value)
})


# Convert the results to a data frame
within_subject_corr_first <- data.frame(subject = unique(first_sample_risk_dec_lea$subject),
                        cor_mean_asym_lea = cor_risk_seeking_lea_lea_first,
                        cor_mean_asym_dec = cor_risk_seeking_lea_dec_first)
# Convert the results to a data frame
within_subject_corr_second <- data.frame(subject = unique(second_sample_risk_dec_lea$subject),
                                        cor_mean_asym_lea = cor_risk_seeking_lea_lea_second,
                                        cor_mean_asym_dec = cor_risk_seeking_lea_dec_second)

p1_mean <- mean(within_subject_corr_first$cor_mean_asym_lea)
p1_SE <- sd(within_subject_corr_first$cor_mean_asym_lea)/sqrt(length(within_subject_corr_first$cor_mean_asym_lea))
p2_mean <- mean(within_subject_corr_first$cor_mean_asym_dec)
p2_SE <- sd(within_subject_corr_first$cor_mean_asym_dec)/sqrt(length(within_subject_corr_first$cor_mean_asym_dec))
p3_mean <- mean(within_subject_corr_second$cor_mean_asym_lea)
p3_SE <- sd(within_subject_corr_second$cor_mean_asym_lea)/sqrt(length(within_subject_corr_second$cor_mean_asym_lea))
p4_mean <- mean(within_subject_corr_second$cor_mean_asym_dec)
p4_SE <- sd(within_subject_corr_second$cor_mean_asym_dec)/sqrt(length(within_subject_corr_second$cor_mean_asym_dec))



# Function to perform bootstrap and calculate confidence interval for the mean
bootstrap_mean_ci <- function(data, alpha = 0.05, n_bootstrap = 10000) {
  # Function to generate a bootstrap sample
  generate_bootstrap_sample <- function(x) {
    return(sample(x, replace = TRUE))
  }
  
  # Perform bootstrap
  bootstrap_means <- replicate(n_bootstrap, mean(generate_bootstrap_sample(data)))
  
  # Calculate confidence interval
  ci <- quantile(bootstrap_means, c(alpha / 2, 1 - alpha / 2))
  
  return(ci)
}

# Calculate 95% bootstrap confidence interval for the mean
ci <- bootstrap_mean_ci(within_subject_corr_second$cor_mean_asym_lea)




##third panel-variance:
##SD plot

subject_var_first <- first_sample_risk_dec_lea %>%
  group_by(subject) %>%
  summarize(var_log_abs_pun = var(log_abs_pun),
            var_log_R = var(log_R))

subject_slope_first <- first_sample_risk_dec_lea %>%
  group_by(subject) %>%
  summarize(slope = var(log_abs_pun)-var(log_R))

subject_var_second <- second_sample_risk_dec_lea %>%
  group_by(subject) %>%
  summarize(var_log_abs_pun = var(log_abs_pun),
            var_log_R = var(log_R))

subject_slope_second <- second_sample_risk_dec_lea %>%
  group_by(subject) %>%
  summarize(slope = var(log_abs_pun)-var(log_R))



# Reshape the data to a longer format
subject_var_first_long <- pivot_longer(subject_var_first, cols = c(var_log_abs_pun, var_log_R), names_to = "Variable")
subject_var_second_long <- pivot_longer(subject_var_second, cols = c(var_log_abs_pun, var_log_R), names_to = "Variable")


subject_var_first_long <- subject_var_first_long %>%
  mutate(
    adjusted_x = ifelse(Variable == "var_log_abs_pun", 1.05, 1.95)
  )

subject_var_second_long <- subject_var_second_long %>%
  mutate(
    adjusted_x = ifelse(Variable == "var_log_abs_pun", 1.05, 1.95)
  )
# Calculate means and standard errors for Punishment and Reward
means_first <- subject_var_first_long %>%
  group_by(Variable) %>%
  summarize(
    mean_value = mean(value),
    se = sd(value) / sqrt(n())
  )

means_first_slope <- subject_slope_first %>%
  summarize(
    mean_value = mean(slope),
    se = sd(slope) / sqrt(n())
  )

means_second <- subject_var_second_long %>%
  group_by(Variable) %>%
  summarize(
    mean_value = mean(value),
    se = sd(value) / sqrt(n())
  )

means_second_slope <- subject_slope_second %>%
  summarize(
    mean_value = mean(slope),
    se = sd(slope) / sqrt(n())
  )


model_EEG_deco <- stan_glmer(decodability_negative_PE_scaled ~ log_abs_pun + (log_abs_pun | subject), data = both_samples_EEG, iter = 5000)
model_HR_deco <- stan_glmer(decodability_negative_PE_scaled ~ log_abs_pun + (log_abs_pun | subject), data = second_sample_HR, iter = 5000)


# Extract the fixed effects coefficients and their standard errors
posterior_samples_slope_EEG <- as.matrix(model_EEG_deco)[, "log_abs_pun"]
posterior_samples_intercept_EEG <- as.matrix(model_EEG_deco)[, "(Intercept)"]
posterior_samples_slope_HR <- as.matrix(model_HR_deco_second)[, "log_abs_pun"]
posterior_samples_intercept_HR <- as.matrix(model_HR_deco)[, "(Intercept)"]
posterior_EEG=data.frame(slope=posterior_samples_slope_EEG,intercept=posterior_samples_intercept_EEG)
posterior_HR=data.frame(slope=posterior_samples_slope_HR,intercept=posterior_samples_intercept_HR)


conf_int_EEG <- posterior_interval(model_EEG_deco, level = 0.95)
conf_int_HR <- posterior_interval(model_HR_deco_second, level = 0.95)












































model_decod <- stan_glmer(decodability_negative_PE_scaled ~ log_abs_pun + (log_abs_pun | subject), data = both_samples_EEG)

# Extract the fixed effects coefficients and their standard errors
posterior_samples_slope <- as.matrix(model_decod)[, "log_abs_pun"]
posterior_samples_intercept <- as.matrix(model_decod)[, "(Intercept)"]
posterior=data.frame(slope=posterior_samples_slope,intercept=posterior_samples_intercept)


# Create the plot with individual subject lines, group trend line, and confidence intervals
my_plot <- ggplot() +
  geom_point(data = both_samples_EEG, aes(x = log_abs_pun, y = decodability_negative_PE_scaled), color = "gray45", alpha = 0.5) +
  geom_smooth(data = both_samples_EEG, aes(x = log_abs_pun, y = decodability_negative_PE_scaled, color = subject, group = subject), method = "lm", se = FALSE, color = "gray45", alpha = 0.6, size=0.5) +
  #geom_line(data = group_data, aes(x = log_abs_pun, y = decodability_negative_PE_scaled), color = "royalblue3", size = 1) +
  geom_abline(data = posterior, aes(slope = slope, intercept = intercept), color = "royalblue3", alpha = 0.01)+
  theme(legend.position = "none", panel.background = element_rect(fill = "white"), axis.title.x = element_text(size = 15), axis.title.y = element_text(size = 15), axis.text.x=element_text(size = 14), axis.text.y=element_text(size = 14))+
  xlab(expression(log(W^pun)))+
  ylab("Negative PE EEG decodability")+
  ylim(-2,4)+
  xlim(-1.5,2)

log_abs_pun = both_samples_EEG$log_abs_pun
my_plot <- ggplot() +
  geom_point(data = both_samples_EEG, aes(x = log_abs_pun, y = decodability_negative_PE_scaled), color = "gray45", alpha = 0.5) +
  geom_smooth(data = both_samples_EEG, aes(x = log_abs_pun, y = decodability_negative_PE_scaled, color = subject, group = subject), method = "lm", se = FALSE, color = "gray45", alpha = 0.6, size=0.5) +
  #geom_line(data = group_data, aes(x = log_abs_pun, y = decodability_negative_PE_scaled), color = "royalblue3", size = 1) +
  geom_ribbon(data = both_samples_EEG, aes(ymin = intercept_HDI[1] + slope_HDI[1] * log_abs_pun,
                  ymax = intercept_HDI[2] + slope_HDI[2] * log_abs_pun, x = log_abs_pun), fill = "royalblue3", alpha=0.5)+
  #geom_ribbon(data = both_samples_EEG, aes(x = log_abs_pun, ymin = ci_lower, ymax = ci_upper), fill = "gray", alpha = 0.2) +
  theme(legend.position = "none", panel.background = element_rect(fill = "white"), axis.title.x = element_text(size = 15), axis.title.y = element_text(size = 15), axis.text.x=element_text(size = 14), axis.text.y=element_text(size = 14))+
  xlab(expression(log(abs(W^pun))))+
  ylab("Negative PE EEG decodability")+
  ylim(-2,4)+
  xlim(-1.5,2)


my_plot <- ggplot() +
  geom_point(data = both_samples_EEG, aes(x = log_abs_pun, y = decodability_negative_PE_scaled), color = "gray45", alpha = 0.5) +
  geom_smooth(data = both_samples_EEG, aes(x = log_abs_pun, y = decodability_negative_PE_scaled, color = subject, group = subject), method = "lm", se = FALSE, color = "gray45", alpha = 0.6, size=0.5) +
  #geom_line(data = group_data, aes(x = log_abs_pun, y = decodability_negative_PE_scaled), color = "royalblue3", size = 1) +
  stat_lineribbon(aes(y = .epred))+
  #geom_ribbon(data = both_samples_EEG, aes(x = log_abs_pun, ymin = ci_lower, ymax = ci_upper), fill = "gray", alpha = 0.2) +
  theme(legend.position = "none", panel.background = element_rect(fill = "white"), axis.title.x = element_text(size = 15), axis.title.y = element_text(size = 15), axis.text.x=element_text(size = 14), axis.text.y=element_text(size = 14))+
  xlab(expression(log(abs(W^pun))))+
  ylab("Negative PE EEG decodability")+
  ylim(-2,4)+
  xlim(-1.5,2)



ggsave("plotEEG.png", plot = my_plot, width = 6, height = 6, dpi = 300)

# Set the PDF file dimensions
pdf("histogram_plot.pdf", width = 10, height = 8)
plot(posterior_samples_slope,posterior_samples_intercept)
# Create the histogram
plot(density(posterior_samples_slope))

# Close the PDF device
dev.off()





















#####
# Fit the mixed-effects model
model_decod <- lmer(decodability_negative_PE_scaled ~ log_abs_pun + (log_abs_pun | subject), data = both_samples_EEG)

# Extract the fixed effects coefficients and their standard errors
fixed_effects <- fixef(model_decod)
slope <- fixed_effects["log_abs_pun"]
intercept <- fixed_effects["(Intercept)"]

# Get the standard errors from the model summary
summary_model <- summary(model_decod)
fixed_effects_se <- summary_model$coefficients[, "Std. Error"]

# Create a data frame for the group trend line
group_data <- data.frame(log_abs_pun = range(both_samples_EEG$log_abs_pun))
group_data$subject <- "Group"
group_data$decodability_negative_PE_scaled <- predict(model_decod, newdata = group_data, re.form = NA)

# Calculate confidence intervals for the group trend line based on standard errors
alpha <- 0.05  # Desired significance level
z_value <- qnorm(1 - alpha / 2)  # z-value based on alpha
group_data$ci_lower <- group_data$decodability_negative_PE_scaled - z_value * fixed_effects_se["log_abs_pun"]
group_data$ci_upper <- group_data$decodability_negative_PE_scaled + z_value * fixed_effects_se["log_abs_pun"]

# Create the plot with individual subject lines, group trend line, and confidence intervals
ggplot() +
  geom_point(data = both_samples_EEG, aes(x = log_abs_pun, y = decodability_negative_PE_scaled), color = "gray50", alpha = 0.5) +
  geom_smooth(data = both_samples_EEG, aes(x = log_abs_pun, y = decodability_negative_PE_scaled, color = subject, group = subject), method = "lm", se = FALSE, color = "gray50") +
  geom_line(data = group_data, aes(x = log_abs_pun, y = decodability_negative_PE_scaled), color = "red", linetype = "dashed", size = 1) +
  geom_ribbon(data = group_data, aes(x = log_abs_pun, ymin = ci_lower, ymax = ci_upper), fill = "gray", alpha = 0.2) +
  theme(legend.position = "none")








scatter_plot <- ggplot(both_samples_EEG, aes(x = log_abs_pun, y = decodability_negative_PE_scaled)) +
  geom_point()

# Add lines for each posterior sample
scatter_plot_with_posterior <- scatter_plot +
  geom_abline(data = posterior, aes(slope = slope, intercept = intercept), color = "red", alpha = 0.1)

# Display the plot
scatter_plot_with_posterior







##SD plot
first_sample_EEG$log_abs_pun=log(abs(first_sample_EEG$mean_P_lea))
first_sample_EEG$log_R=log(abs(first_sample_EEG$mean_R_lea))


subject_var <- first_sample_EEG %>%
  group_by(subject) %>%
  summarize(var_log_abs_pun = var(log_abs_pun),
            var_log_R = var(log_R))

overall_var <- subject_var %>%
  summarize(overall_var_log_abs_pun = mean(var_log_abs_pun),
            overall_var_log_R = mean(var_log_R))

subject_stderr <- subject_var %>%
  summarize(stderr_log_abs_pun = sd(var_log_abs_pun) / sqrt(n()),
            stderr_log_R = sd(var_log_R) / sqrt(n()))

df <- data.frame(variable = c("f","g"), var = c(overall_var$overall_var_log_abs_pun, overall_var$overall_var_log_R),
                 stderr = c(subject_stderr$stderr_log_abs_pun, subject_stderr$stderr_log_R))

var_plot <- ggplot(df, aes(x = variable, y = var, fill = variable)) +
  geom_bar(stat = "identity") +
  geom_errorbar(aes(ymin = var - stderr, ymax = var + stderr), width = 0.2, size = 1, color = "black") +
  theme_bw() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text.y = element_text(size = 20)) + 
  labs(fill = "Variable") +
  guides(fill = "none")+
  xlab(NULL)+
  ylab(NULL)+
  scale_fill_manual(values = c("#FF0000", "#006400"))+
  scale_y_continuous(expand = c(0, 0, 0.01, 0.01))




ggsave(paste0("G:/My Drive/Lab/risk_seeking_article/var_figures_R/" ,'risk-asym-corr_first',".svg"), plot = risk_asym_plot_first, width = 10, height = 10)




# Assuming your dataframe is named EEG_PE_decodability

# Extract mean decodability values for each condition
positive_PE_data <- EEG_PE_decodability$mean_decodability_positive_PE_EEG
negative_PE_data <- EEG_PE_decodability$mean_decodability_negative_PE_EEG

# Define the observed mean decodability values
observed_mean_positive <- mean(positive_PE_data)
observed_mean_negative <- mean(negative_PE_data)

# Define the number of permutations
n_permutations <- 1000

# Initialize vectors to store permuted means
permuted_means_positive <- numeric(n_permutations)
permuted_means_negative <- numeric(n_permutations)

# Permutation procedure for positive PE condition
for (i in 1:n_permutations) {
  # Generate permuted data by randomly changing signs
  permuted_data_positive <- sample(c(positive_PE_data, -positive_PE_data), replace = FALSE)
  
  # Calculate the mean decodability for permuted sample
  permuted_means_positive[i] <- mean(permuted_data_positive)
}

# Permutation procedure for negative PE condition
for (i in 1:n_permutations) {
  # Generate permuted data by randomly changing signs
  permuted_data_negative <- sample(c(negative_PE_data, -negative_PE_data), replace = FALSE)
  
  # Calculate the mean decodability for permuted sample
  permuted_means_negative[i] <- mean(permuted_data_negative)
}

# Calculate p-values for positive and negative PE conditions
p_value_positive <- mean(permuted_means_positive >= observed_mean_positive)
p_value_negative <- mean(permuted_means_negative >= observed_mean_negative)

# Print p-values
cat("Positive PE p-value:", p_value_positive, "\n")
cat("Negative PE p-value:", p_value_negative, "\n")


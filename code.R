setwd("G:/My Drive/Projects/Variations in risk taking/analysis_code")
library(RSQLite)
library(anytime)

get_path <- function(subpath = c()){
  path = file.path(dirname(dirname(rstudioapi::getSourceEditorContext()$path)),"schedule_files")
  if (length(subpath)>0) {
    path = file.path(path, subpath)
  }
  return(path)
}
  
get_subject_list <- function(subpath = c()) {
  path = get_path(subpath)
  files = dir(path = path)
  subjects = sapply(strsplit(as.character(files), "_"), "[[", 1)
}

get_db <- function(subpath, subject) {
  filename = paste0(as.character(subject),"_schedule.db")
  path = get_path(subpath)
  dbname = file.path(path, filename)
  dbConnect(RSQLite::SQLite(), dbname)
}

get_trials <- function(subpath, subject) {
  db = get_db(subpath, subject)
  trials = dbReadTable(db, "trials")
  dbDisconnect(db)
  return(trials)
}

get_risky_trials <-function(subpath, subject){
  db = get_db(subpath, subject)
  trials23 = dbGetQuery(db, 'SELECT block, trial, stim1, stim2, choice, choice_time FROM trials WHERE feedback=0 AND stim1>=18 AND stim2>=18 AND stim1 IN (SELECT number FROM stimuli WHERE condition=2) AND stim2 IN (SELECT number FROM stimuli WHERE condition=3)')  
  trials23$stim1condition=2
  trials23$stim2condition=3
  trials32 = dbGetQuery(db, 'SELECT block, trial, stim1, stim2, choice, choice_time FROM trials WHERE feedback=0 AND stim1>=18 AND stim2>=18 AND stim1 IN (SELECT number FROM stimuli WHERE condition=3) AND stim2 IN (SELECT number FROM stimuli WHERE condition=2)')  
  trials32$stim1condition=3
  trials32$stim2condition=2
  dbDisconnect(db)
  trials = rbind(trials23,trials32)
  trials = trials[order(trials$choice_time),]
  trials$choice_time = as.integer(trials$choice_time/1000)
  return(trials)
}

get_avg_feedback_time <- function(subpath, subject, stimulus){
  db = get_db(subpath, subject)
  query = sprintf('SELECT AVG(feedback_time) FROM trials WHERE feedback=1 AND ((stim1=%d AND choice=0) OR (stim2=%d AND choice=1))',stimulus,stimulus)
  time = dbGetQuery(db, query)
  if (is.na(time)){
    query = sprintf('SELECT AVG(feedback_time) FROM trials WHERE feedback=1 AND ((stim1=%d AND choice=1) OR (stim2=%d AND choice=0))',stimulus,stimulus)
    time = dbGetQuery(db, query)
  }
  dbDisconnect(db)
  return(as.integer(time/1000))
}

add_feedback_times <-function(subpath,subject,trials){
  stimuli = data.frame(number = unique(c(trials$stim1,trials$stim2)))
  stimuli$avg_feedback_time = NA
  stimuli$stimsafe_feedback_time = NA
  stimuli$stimrisky_feedback_time = NA
  for (i in 1:nrow(stimuli)) 
    stimuli$avg_feedback_time[i] = get_avg_feedback_time(subpath,subject,stimuli$number[i])
  for (i in 1:nrow(trials)) {
    if (trials$stim1condition[i]==2) {
      trials$stimsafe_feedback_time[i] = stimuli$avg_feedback_time[stimuli$number==trials$stim1[i]]
      trials$stimrisky_feedback_time[i] = stimuli$avg_feedback_time[stimuli$number==trials$stim2[i]]
    } else {
      trials$stimsafe_feedback_time[i] = stimuli$avg_feedback_time[stimuli$number==trials$stim2[i]]
      trials$stimrisky_feedback_time[i] = stimuli$avg_feedback_time[stimuli$number==trials$stim1[i]]
    }
  }
  trials$min_time = get_min_time(subpath,subject)
  return(trials)
}

get_stimulus_bins <- function(subpath, subject, stimulus, start_time, time_per_bin){
  db = get_db(subpath, subject)
  query = sprintf('SELECT feedback_time FROM trials WHERE feedback=1 AND ((stim1=%d AND choice=0) OR (stim2=%d AND choice=1))',stimulus,stimulus)
  times = dbGetQuery(db, query)
  dbDisconnect(db)
  bins = floor((times - start_time * 1000) / time_per_bin )+1
  return(bins)
}

get_stimulus_bins_notchosen <- function(subpath, subject, stimulus, start_time, time_per_bin){
  db = get_db(subpath, subject)
  query = sprintf('SELECT feedback_time FROM trials WHERE feedback=1 AND ((stim1=%d AND choice=1) OR (stim2=%d AND choice=0))',stimulus,stimulus)
  times = dbGetQuery(db, query)
  dbDisconnect(db)
  bins = floor((times - start_time * 1000) / time_per_bin )+1
  return(bins)
}

add_stimuli_bins <-function(subpath, subject,trials, start_time, time_per_bin){
  stimuli = data.frame(number = unique(c(trials$stim1,trials$stim2)))
  stimuli$bin1 = NA
  stimuli$bin2 = 1
  stimuli$bin1_p = NA
  stimuli$bin2_p = 0
  for (i in 1:nrow(stimuli)) {
    bins = get_stimulus_bins(subpath, subject, stimuli$number[i], start_time, time_per_bin)
    if (length(bins$feedback_time)==0){
      bins = get_stimulus_bins_notchosen(subpath, subject, stimuli$number[i], start_time, time_per_bin)
    } 
    bins_hist = hist(bins$feedback_time, breaks = (1:(max(bins$feedback_time)+1))-0.5, plot = FALSE)
    bins_sorted = sort(bins_hist$counts, decreasing = TRUE, index.return = TRUE)
    stimuli$bin1[i] = bins_sorted$ix[1]
    stimuli$bin1_p[i] = bins_sorted$x[1] / sum(bins_sorted$x)
    if (length(bins_sorted$x)>=2) {
      stimuli$bin2[i] = bins_sorted$ix[2]
      stimuli$bin2_p[i] = bins_sorted$x[2] / sum(bins_sorted$x)
    } 
  }
  trials$stimsafe_bin1 = NA
  trials$stimsafe_bin2 = NA
  trials$stimsafe_bin1_p = NA
  trials$stimsafe_bin2_p = NA
  trials$stimrisky_bin1 = NA
  trials$stimrisky_bin2 = NA
  trials$stimrisky_bin1_p = NA
  trials$stimrisky_bin2_p = NA
  for (i in 1:nrow(trials)) {
    if (trials$stim1condition[i]==2) {
      stimsafe = trials$stim1[i]
      stimrisky = trials$stim2[i]
    } else {
      stimsafe = trials$stim2[i]
      stimrisky = trials$stim1[i]
    }
    trials$stimsafe_bin1[i] = stimuli$bin1[stimuli$number==stimsafe]
    trials$stimsafe_bin2[i] = stimuli$bin2[stimuli$number==stimsafe]
    trials$stimsafe_bin1_p[i] = stimuli$bin1_p[stimuli$number==stimsafe]
    trials$stimsafe_bin2_p[i] = stimuli$bin2_p[stimuli$number==stimsafe]
    trials$stimrisky_bin1[i] = stimuli$bin1[stimuli$number==stimrisky]
    trials$stimrisky_bin2[i] = stimuli$bin2[stimuli$number==stimrisky]
    trials$stimrisky_bin1_p[i] = stimuli$bin1_p[stimuli$number==stimrisky]
    trials$stimrisky_bin2_p[i] = stimuli$bin2_p[stimuli$number==stimrisky]
  }
  return(trials)
}

get_min_time <- function(subpath, subject){
  db = get_db(subpath, subject)
  query = 'SELECT MIN(feedback_time) FROM trials WHERE stim1>=18 OR stim2>=18'
  time = dbGetQuery(db, query)
  dbDisconnect(db)
  return(as.integer(time/1000))
}

get_avg_outcome <- function(subpath, subject, stimulus){
  db = get_db(subpath, subject)
  query = sprintf('SELECT AVG(outcome) FROM trials WHERE feedback=1 AND ((stim1=%d AND choice=0) OR (stim2=%d AND choice=1))',stimulus,stimulus)
  outcome = dbGetQuery(db, query)
  if (is.na(outcome)){
    outcome = 0
  }
  dbDisconnect(db)
  return(outcome)
}

get_avg_accuracy <- function(subpath, subject, stimulus){
  db = get_db(subpath, subject)
  query = sprintf('SELECT COUNT(outcome) FROM trials WHERE feedback=1 AND ((stim1=%d AND choice=0 AND stim2 IN (SELECT number FROM stimuli WHERE condition = 0)) OR (stim2=%d AND choice=1 AND stim1 IN (SELECT number FROM stimuli WHERE condition = 0)))',stimulus,stimulus)
  good1 = dbGetQuery(db, query)[[1]]
  query = sprintf('SELECT COUNT(outcome) FROM trials WHERE feedback=1 AND ((stim1=%d AND choice=1 AND stim2 IN (SELECT number FROM stimuli WHERE condition = 1)) OR (stim2=%d AND choice=0 AND stim1 IN (SELECT number FROM stimuli WHERE condition = 1)))',stimulus,stimulus)
  good2 = dbGetQuery(db, query)[[1]]
  query = sprintf('SELECT COUNT(outcome) FROM trials WHERE feedback=1 AND ((stim1=%d AND choice=1 AND stim2 IN (SELECT number FROM stimuli WHERE condition = 0)) OR (stim2=%d AND choice=0 AND stim1 IN (SELECT number FROM stimuli WHERE condition = 0)))',stimulus,stimulus)
  bad1 = dbGetQuery(db, query)[[1]]
  query = sprintf('SELECT COUNT(outcome) FROM trials WHERE feedback=1 AND ((stim1=%d AND choice=0 AND stim2 IN (SELECT number FROM stimuli WHERE condition = 1)) OR (stim2=%d AND choice=1 AND stim1 IN (SELECT number FROM stimuli WHERE condition = 1)))',stimulus,stimulus)
  bad2 = dbGetQuery(db, query)[[1]]
  dbDisconnect(db)
  avg_accuracy = (good1 + good2)/(good1 + good2 + bad1 + bad2)
}

add_avg_outcome <-function(subpath, subject,trials){
  stimuli = data.frame(number = unique(c(trials$stim1,trials$stim2)))
  stimuli$avg_outcome = NA
  for (i in 1:nrow(stimuli)) 
    stimuli$avg_outcome[i] = get_avg_outcome(subpath, subject,stimuli$number[i])[[1]]
  trials$avg_outcome = NA
  for (i in 1:nrow(trials)) {
    if (trials$stim1condition[i]==2) {
      trials$avg_outcome[i] = stimuli$avg_outcome[stimuli$number==trials$stim2[i]] - stimuli$avg_outcome[stimuli$number==trials$stim1[i]]
    } else {
      trials$avg_outcome[i] = stimuli$avg_outcome[stimuli$number==trials$stim1[i]] - stimuli$avg_outcome[stimuli$number==trials$stim2[i]]
    }
  }
  trials$avg_outcome = (trials$avg_outcome - mean(trials$avg_outcome)) / sd(trials$avg_outcome)
  return(trials)
}

add_avg_accuracy <-function(subpath, subject,trials){
  stimuli = data.frame(number = unique(c(trials$stim1,trials$stim2)))
  stimuli$avg_accuracy = NA
  for (i in 1:nrow(stimuli)) 
    stimuli$avg_accuracy[i] = get_avg_accuracy(subpath, subject,stimuli$number[i])
  trials$avg_accuracy = NA
  for (i in 1:nrow(trials))
    trials$avg_accuracy[i] = (stimuli$avg_accuracy[stimuli$number==trials$stim2[i]] + stimuli$avg_accuracy[stimuli$number==trials$stim1[i]])/2
  trials$avg_accuracy = (trials$avg_accuracy - mean(trials$avg_accuracy)) / sd(trials$avg_accuracy)
  return(trials)
}
  
get_stimuli <- function(subpath, subject) {
  db = get_db(subpath, subject)
  trials = dbReadTable(db, "stimuli")
  dbDisconnect(db)
  return(trials)
}

bin_times <- function(times, time_per_bin, start_time){
  t = start_time
  bin = 1
  bins = rep(NA,length(times))
  while (t <= max(times)){
    bin_t = which(times>=t & times<t+time_per_bin)
    if (length(bin_t)>0) bins[bin_t] = bin
    bin = bin + 1
    t = t + time_per_bin
  }
  return(bins)
}

days <- function(start_day, end_day, bin_width, start_time){
  from = seq(start_day,end_day,bin_width)
  from_ms = seq(start_time*1000, length.out = length(from), by = bin_width*24*60*60*1000)
  to = from + bin_width - 1
  to_ms = from_ms + bin_width*24*60*60*1000 - 1
  return(data.frame(bin = 1:length(from),from=from, to=to, from_ms = from_ms, to_ms = to_ms))
}

analysis <- function(days_per_bin=3, subjects = c(), subpath = c()) {
  if (length(subjects)==0) subjects = get_subject_list(subpath)
  num_subjects = length(subjects)
  all_trials = data.frame()
  all_bins = data.frame()
  cat("Reading data from subject")
  for (i in 1:num_subjects){
    subject = subjects[i]
    cat(sprintf(" %s", subject))
    trials <- get_risky_trials(subpath, subject)
    trials <- add_feedback_times(subpath, subject, trials)
    trials <- add_avg_outcome(subpath, subject, trials)
    trials <- add_avg_accuracy(subpath, subject, trials)
    exclude = which(is.na(trials$learning_time)|is.na(trials$decision_time))
    if (length(exclude)>0) trials <- trials[-exclude]
    start_day = anydate(trials$min_time[1])
    end_day = anydate(max(trials$choice_time))
    start_time = anytime(trials$min_time[1])
    hours = as.numeric(format(start_time, format = "%H"))
    minutes = as.numeric(format(start_time, format = "%M"))
    seconds = as.numeric(format(start_time, format = "%S"))
    start_time = as.numeric(start_time) - seconds - 60*minutes - 3600*hours
    
    
    trials <- add_stimuli_bins(subpath, subject, trials, start_time, days_per_bin*24*60*60*1000)
    trials$stimsafe_learning_bin = bin_times(trials$stimsafe_feedback_time,days_per_bin*24*60*60,start_time)
    trials$stimrisky_learning_bin = bin_times(trials$stimrisky_feedback_time,days_per_bin*24*60*60,start_time)
    trials$decision_bin = bin_times(trials$choice_time,days_per_bin*24*60*60,start_time)
    trials$subject = subject
    bins = days(start_day, end_day, days_per_bin, start_time)
    bins$subject = subject
    if (i==1) {
      all_trials = trials
      all_bins = bins
    } else {
      all_trials = rbind(all_trials,trials)
      all_bins = rbind(all_bins,bins)
    }
  }
  cat(".\n")
  all_trials$risky_choice = as.numeric((all_trials$choice==0 & all_trials$stim1condition==3) | (all_trials$choice==1 & all_trials$stim2condition==3))
  return(list(all_trials=all_trials, all_bins=all_bins))
}


  

# ======================================================================================================== -
# Author: Jonas Anderegg, jonas.anderegg@usys.ethz.ch
# Project: LesionZoo
# Date: 08.06.2021
# Analyze sampled training data and train classifiers
# ======================================================================================================== -

rm(list = ls())

.libPaths("T:/R3UserLibs")
library(data.table)
library(tidyverse)
library(ggsci)
library(prospectr)
library(caret)
library(pryr)
library(Rvenn)
library(nls.multstart)
library(segmented)

wd = "Z:/Public/Jonas/001_LesionZoo"
setwd(wd)

# load auxiliary functions
source("RCode/model_utils.R")

# ======================================================================================================== -
# Load and pre-process data ----
# ======================================================================================================== -

# get file names (new)
fnames_pos = list.files("TrainingData_Lesions/Positives/Segments/Scans/Profiles/spl_n", full.names = T, pattern = ".csv")
fnames_neg = list.files("TrainingData_Lesions/Negatives/Segments/Scans/Profiles/spl_n", full.names = T, pattern = ".csv")
fnames_new <- c(fnames_pos, fnames_neg)

# get file names (original)
fnames_pos = list.files("TrainingData_Lesions/Positives/Segments/original/Scans/Profiles/spl_n", full.names = T, pattern = ".csv")
fnames_neg = list.files("TrainingData_Lesions/Negatives/Segments/original/Scans/Profiles/spl_n", full.names = T, pattern = ".csv")
fnames_orig <- c(fnames_pos, fnames_neg)

# gt file names (reinforce 1st iteration)
fnames_pos = list.files("TrainingData_Lesions/Positives/Segments/reinforce_iter1/Scans/Profiles/spl_n", full.names = T, pattern = ".csv")
fnames_neg = list.files("TrainingData_Lesions/Negatives/Segments/reinforce_iter1/Scans/Profiles/spl_n", full.names = T, pattern = ".csv")
fnames_iter1 <- c(fnames_pos, fnames_neg)

# ALL
fnames <- c(fnames_new, fnames_orig, fnames_iter1)

# template
template <-  data.table::fread(fnames[1]) %>% as_tibble() %>% slice(0)

# calculate means of smoothed profiles
means = list()
for (j in 1:length(fnames)){
  
  print(paste("Processing", j, "/", length(fnames)))
  
  # load data
  file = fnames[j]
  data <- data.table::fread(file, header = TRUE)
  
  # get file name
  bname <- basename(file) %>% gsub(".csv", "", .)
  
  # get sample label
  label <- ifelse(grepl("Positives", file), "pos", "neg")
  
  # add to data
  data$.id <- bname
  data$label <- label
  
  # to tibble
  data <- data %>% as_tibble()
  
  # use template to complete df if necessary (missing variables for smaller lesions)
  data <- bind_rows(template, data)
  
  # re-arrange
  data <- data %>% dplyr::select(.id, label, everything())
  
  # separate data from sample info
  info <- data[1:2]
  preds <- data[3:length(data)]
  
  # reshape data 
  # each channel profile must have its own row for smoothing
  out <- list()
  for (i in 1:42){
    # select data for channel
    df <- preds[((i-1)*76+1):(i*76)]
    # get channel name
    channel <- strsplit(names(df),"_") %>% lapply("[[", 1) %>% unlist() %>% unique()
    # get color_space name
    color_space <- strsplit(names(df),"_") %>% lapply("[[", 2) %>% unlist() %>% unique()
    # get channel and color_space
    channel_color_space <- paste(channel, color_space, sep = "_")
    # get type 
    type <- strsplit(names(df),"_") %>% lapply("[[", 4) %>% unlist() %>% unique()
    # add this to the data
    names(df) <- strsplit(names(df), "_") %>% lapply("[[", 3) %>% unlist()
    df$channel <- channel
    df$color_space <- color_space
    df$channel_color_space <- channel_color_space
    df$type <- type
    # rearrange columns
    df <- df %>% dplyr::select(channel, color_space, channel_color_space, type, everything())
    # "extrapolate" by simple "extension"
    na_cols <- which(colSums(is.na(df)) > 0)
    non_na_col <- which(colSums(!is.na(df[4:length(df)])) > 0)[1] + 3
    df[,na_cols] <- df[,non_na_col]
    
    # add info
    out[[i]] <- cbind(info, df)
  }
  lw0 <- data.table::rbindlist(out) %>% as_tibble()
  
  # apply the moving average
  lw_smth7 <- prospectr::movav(lw0[7:length(lw0)], w = 7)
  
  # reshape for summarising and plotting
  lw_smth <- lw_smth7 %>% as_tibble() %>% bind_cols(lw0[1:6], .) %>% 
    pivot_longer(., 7:length(.), names_to = "posY", values_to = "value") %>% 
    mutate(posY = as.numeric(posY))
  
  # calculate means over all profiles
  means[[j]] <- lw_smth %>% group_by(.id, label, channel, color_space, posY, type) %>% 
    summarise(mean = mean(value),
              sd = sd(value))
  
}

lapply(means, nrow)

all_means <- means %>% bind_rows() %>% group_by(.id, label) %>% group_nest() %>% 
  mutate(checker = purrr::map_lgl(data, ~any(is.na(.[.$posY == -45, "mean"])))) %>% 
  filter(checker != TRUE) 

n_pos <- all_means %>% filter(label == "pos") %>% nrow()
n_neg <- all_means %>% filter(label == "neg") %>% nrow()

all_means <- all_means%>% 
  unnest(data) %>% 
  dplyr::select(-checker)

saveRDS(all_means, "Data_products/average_profiles_spl_segments_iter1.rds")

# ======================================================================================================== -
# Plot data ----
# ======================================================================================================== -

all_means <- readRDS("Data_products/average_profiles_spl_segments_iter1.rds")

# plot profiles
plot <- ggplot(all_means) +
  geom_line(aes(x = posY, y = mean, color = label, group = interaction(.id, channel, color_space))) +
  facet_wrap(~interaction(channel, color_space)) +
  xlab("Distance from Lesion Boundary [Pixel]") +  ylab("Scaled Pixel Intensity") +
  ggsci::scale_color_npg() +
  ggsci::scale_fill_npg() +
  theme_bw() +
  theme(panel.grid = element_blank())

png("Output/Figures/profiles_samples.png", width = 10, height = 7, units = 'in', res = 300)
plot(plot)
dev.off()

# Calculate grand means over lesion classes
# calculate means over all profiles
means <- all_means %>% group_by(label, channel, color_space, posY) %>% 
  summarise(mean_ = mean(mean, na.rm = T),
            sd = sd(mean, na.rm = T))

# plot profiles
plot <- ggplot(means) +
  geom_line(aes(x = posY, y = mean_, color = label, group = interaction(label, channel, color_space)), size = 1) +
  geom_ribbon(aes(x = posY, ymin = mean_ - sd,
                  ymax = mean_ + sd,
                  fill = label,
                  group = interaction(label, channel, color_space)), alpha = 0.15) +
  facet_wrap(~interaction(channel, color_space)) +
  xlab("Distance from Lesion Boundary [Pixel]") +  ylab("Scaled Pixel Intensity") +
  ggsci::scale_color_npg() +
  ggsci::scale_fill_npg() +
  theme_bw() +
  theme(panel.grid = element_blank())

png("Output/Figures/profiles_classes.png", width = 10, height = 7, units = 'in', res = 300)
plot(plot)
dev.off()

# ======================================================================================================== -
# Model ----
# ======================================================================================================== -

# reshape data for modelling
data_mod <- all_means %>% 
  # do not require standard deviation
  dplyr::select(-sd) %>% 
  # create new variable
  mutate(var = paste(channel, color_space, posY, sep = "_")) %>% 
  # drop old separated variables 
  dplyr::select(-channel, -color_space, -posY) %>% 
  pivot_wider(names_from = "var", values_from = "mean") %>% 
  dplyr::select(-.id)

# resample profiles and remove redudancies
drop <- paste(paste0(as.character(seq(-30, 37, by = 2)), "$"), collapse = "|")
names_drop <- grep(drop, names(data_mod), value = TRUE)
data_mod_red <- data_mod %>% 
  # increase the distance between profile neighbouring pixels
  dplyr::select(-one_of(names_drop)) %>% 
  # these channels appear to be redundant (perfectly correlated with another channel)
  dplyr::select(-starts_with("L_Lab"), -contains("YCC"))

# saveRDS(data_mod_red, "Data_products/data_model_reduced_spl.rds")
# write_csv(data_mod_red, "Data_products/data_model_reduced_spl.csv")

# ======================================================================================================== -

# data_mod_red <- readRDS("Data_products/data_model_reduced_spl.rds")

pred_matrix <- data_mod_red %>% dplyr::select(-label)
M  <- cor(pred_matrix, use = "pairwise.complete.obs", method = "pearson")
summary_stats <- list(mean(M[lower.tri(M)]), min(M[lower.tri(M)]), max(M[lower.tri(M)]))

png("Output/Figures/predictors_structure.png", width = 7, height = 7, units = 'in', res = 300)
corrplot::corrplot(M, method = "color", tl.pos = "n")
dev.off()

# train and validate 
indx <- createMultiFolds(data_mod_red$label, k = 10, times = 5)
ctrl <- caret::trainControl(method = "repeatedcv", 
                            index = indx,
                            classProbs = TRUE,
                            savePredictions = TRUE,
                            verboseIter = TRUE,
                            selectionFunction = "oneSE")

# > random forest ----
#specify model tuning parameters
mtry <- c(1, 2, 5, 9, 14, 20, 30, 45, 70, 100, 200)
min_nodes <- c(1, 2, 5, 10)
tune_grid <- expand.grid(mtry = mtry,
                         splitrule = "gini", # default
                         min.node.size = min_nodes)

rf_ranger <- caret::train(label ~ .,
                          data = data_mod_red,
                          preProc = c("center", "scale"),
                          method = "ranger",
                          tuneGrid = tune_grid,
                          importance = "permutation",
                          num.trees = 200,
                          trControl = ctrl)

saveRDS(rf_ranger, "Output/Models/spl/rf_lesion_clf.rds")

# > support vector machine ----
tune_length = 20
svm <- caret::train(label ~., 
                    data = data_mod_red, 
                    preProc = c("center", "scale"),
                    method = "svmRadial", 
                    trControl = ctrl, preProcess = c("center","scale"), 
                    tuneLength = tune_length)

saveRDS(svm, "Output/Models/spl/svm_lesion_clf.rds")

# > pls ----
plsda <- train(label ~., 
               data = data_mod_red, 
               preProc = c("center", "scale"),
               method = "pls",
               tuneLength = tune_length, 
               trControl = ctrl,
               returnResamp = "all")

plot(plsda)
saveRDS(plsda, "Output/Models/spl/pls_lesion_clf.rds")

pseudo_new <- data_mod_red

write_csv(pseudo_new, "Data_products/pseudo_new.csv")

pred <- predict(plsda, pseudo_new[-1])
actual <- pseudo_new[1]

tb <- tibble(pred = pred, obs = actual)
tb <- tb %>% mutate(match = ifelse(pred==obs$label, 1, 0))
tb %>% filter(match==0)


## SAVE ¨MODEL FOR IMPORT IN PYTHON

MODEL_SAVE_PATH = "Output/Models/spl/pls"
DEP_LIBS = c("C:/Users/anjonas/RLibs/caret", "C:/Users/anjonas/RLibs/pls")

# save
model_rds_path = paste(MODEL_SAVE_PATH, ".rds", sep='')
model_dep_path = paste(MODEL_SAVE_PATH, ".dep", sep='')

# save model
# dir.create(dirname(model_path), showWarnings=FALSE, recursive=TRUE)
saveRDS(plsda, model_rds_path)

# save dependency list
file_conn <- file(model_dep_path)
writeLines(DEP_LIBS, file_conn)
close(file_conn)

pred <- predict(plsda, pseudo_new[-1])

# ======================================================================================================== -

# TEST

pseudo_new <- read_csv("Data_products/pseudo_new.csv")

model <- readRDS("Output/Models/spl/pls.rds")

pred <- predict(model, pseudo_new[-1])

# ======================================================================================================== -

# Model comparison ----

model_files <- list.files("Output/Models", pattern = "_lesion_clf.rds", full.names = TRUE)
models <- lapply(model_files, readRDS)
model_names <- basename(model_files) %>% gsub(".rds", "", .)

miss_pred <- list()
for (i in 1:length(models)){
  print(i)
  model <- models[[i]]
  # extract predicted and observed from cv folds
  predobs_cv <- plyr::match_df(model$pred, model$bestTune, on = names(model$bestTune))
  
  #Average predictions of the held out samples
  predobs <- predobs_cv %>% 
    mutate(check = ifelse(pred == obs, 1, 0)) %>% 
    group_by(rowIndex) %>% 
    summarize(obs = sum(check)) %>% 
    mutate(obs = obs/5)
  
  # misspredicted samples
  misspred <- which(predobs$obs %in% c(0.0, 0.1, 0.2))
  miss_pred[[i]] <- data_reshape[misspred, ".id"]
  
  # stability across resamples
  ggplot(predobs) +
    geom_histogram(aes(x = obs), stat = "count") +
    theme_bw() +
    theme(panel.grid = element_blank())
  
  # class probabilities
  ggplot(predobs_cv) +
    geom_density(aes(x = pos)) +
    theme_bw() +
    theme(panel.grid = element_blank())
}

names(miss_pred) <- c("PLS-DA", "RF", "SVM")
out <- dplyr::bind_rows(miss_pred, .id = "meta_information")
out$model <- strsplit(out$meta_information, "_") %>% lapply("[[", 1) %>% unlist()

COMMON <- intersect(intersect(miss_pred[[1]]$.id, miss_pred[[2]]$.id), miss_pred[[3]]$.id)

data <- miss_pred %>% lapply(., pull, .id)
data = Venn(data)

venn_diag <- ggvenn(data) +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "none")

png("Output/Figures/venn_diag.png", width = 7, height = 7, units = 'in', res = 300)
plot(venn_diag)
dev.off()

predobs_cv <- plyr::match_df(plsda$pred, plsda$bestTune, on = "ncomp")

#Average predictions of the held out samples
predobs <- predobs_cv %>% 
  mutate(check = ifelse(pred == obs, 1, 0)) %>% 
  group_by(rowIndex) %>% 
  summarize(obs = sum(check)) %>% 
  mutate(obs = obs/5)

# misspredicted samples
misspred <- which(predobs$obs %in% c(0.0, 0.1, 0.2))
data_misspred <- data_reshape[misspred, ".id"]


# stability across resamples
ggplot(predobs) +
  geom_histogram(aes(x = obs), stat = "count") +
  theme_bw() +
  theme(panel.grid = element_blank())

# class probabilities
ggplot(predobs_cv) +
  geom_density(aes(x = pos)) +
  theme_bw() +
  theme(panel.grid = element_blank())

set.seed(42)
toy = map(sample(5:25, replace = TRUE, size = 10),
          function(x) sample(letters, size = x))
toy[1:3]

m <- models[[3]]

cm <- confusionMatrix(m)
cm <- confusionMatrix(m$pred$pred,m$pred$obs)

# ======================================================================================================== -
# ======================================================================================================== -

# Extract "higher-level" features ----
# > Breakpoint model for v_Luv ----
## get data
data_v <- all_means %>% 
  filter(channel == "v", color_space == "Luv", type=="sc") %>% 
  group_by(label, .id)

## fit breakpoint models
data_fits <- data_v %>%
  dplyr::select(.id, label, posY, mean) %>% 
  as.data.frame() %>%
  tidyr::nest(data = c(posY, mean)) %>%
  group_by(.id) %>%
  mutate(fit_bp_lm = purrr::map(data, breakpoint_lm)) %>% 
  unnest(fit_bp_lm)

bp_pars_v <- data_fits %>% dplyr::select(.id, slp1:slpdiff)
names(bp_pars_v)[2:length(bp_pars_v)] <- paste0(names(bp_pars_v)[2:length(bp_pars_v)], "_v")

# plot
pd <- data_fits %>% 
  pivot_longer(cols=slp1:slpdiff, names_to = "par", values_to = "value")
ggplot(pd)+
  geom_boxplot(aes(y=value, x=label))+
  facet_wrap(~par, scales = "free")

# ======================================================================================================== -

# > Breakpoint model for R_RGB ----
## get data
data_R <- all_means %>% 
  filter(channel == "R", color_space == "RGB", type == "sc") %>% 
  group_by(label, .id)

## fit breakpoint models
data_fits <- data_R %>%
  dplyr::select(.id, label, posY, mean) %>% 
  as.data.frame() %>%
  tidyr::nest(data = c(posY, mean)) %>%
  group_by(.id) %>%
  mutate(fit_bp_lm = purrr::map(data, breakpoint_lm)) %>% 
  unnest(fit_bp_lm)

bp_pars_R <- data_fits %>% dplyr::select(.id, slp1:slpdiff)
names(bp_pars_R)[2:length(bp_pars_R)] <- paste0(names(bp_pars_R)[2:length(bp_pars_R)], "_R")

# plot
pd <- data_fits %>% 
  pivot_longer(cols=slp1:slpdiff, names_to = "par", values_to = "value")
ggplot(pd)+
  geom_boxplot(aes(y=value, x=label))+
  facet_wrap(~par, scales = "free")

# ======================================================================================================== -

# > Logistic model for H_HSV ----
## get data
data_H <- all_means %>% 
  filter(channel == "H", color_space == "HSV", type == "sc") %>% 
  mutate(posY = posY + 32)

## fit logistic models
data_fits <- data_H %>%
  dplyr::select(.id, label, posY, mean) %>% 
  as.data.frame() %>%
  tidyr::nest(data = c(posY, mean)) %>%
  group_by(.id) %>%
  mutate(fit_log = purrr::map(data,
                              ~ nls_multstart(mean ~ logistic(c, d, b, e, posY = posY),
                                              data = .x,
                                              iter = 750,
                                              start_lower = c(c = -0.5, d = 0.5, e = 20, b = -0.5),
                                              start_upper = c(c = 0.5, d = 1.5, e = 44, b = 0),
                                              convergence_count = 150,
                                              supp_errors = 'Y'))) %>%
  tidyr::gather(met, fit, fit_log:fit_log)

# new data frame of predictions
new_preds <- data_H %>%
  ungroup() %>% 
  do(., data.frame(posY = seq(min(.$posY), max(.$posY), length.out = 100), stringsAsFactors = FALSE))
# max and min for each curve
max_min <- group_by(data_H, .id) %>%
  summarise(., min_gposY = min(posY), max_gposY = max(posY)) %>%
  ungroup()
# create new predictions
preds2 <- data_fits %>%
  mutate(., p = purrr::map(fit, broom::augment, newdata = new_preds)) %>%
  unnest(p) %>%
  merge(., max_min, by = '.id') %>%
  group_by(., .id) %>%
  filter(., posY > unique(min_gposY) & posY < unique(max_gposY)) %>%
  arrange(., .id, posY) %>%
  rename(., value = .fitted) %>%
  ungroup()

# # plot model fits
# ggplot() +
#   geom_point(aes(posY, mean, color = label), size = 0.3, shape = 1, data_H) +
#   geom_line(aes(posY, value, group = met, colour = met), alpha = 0.75, size = 0.3, preds2) +
#   facet_wrap(~ .id, labeller = labeller(.multi_line = FALSE))

# get model parameters
mod_pars <- data_fits %>%
  #reconverto to wide
  dplyr::filter(met == "fit_log") %>% 
  tidyr::spread(met, fit) %>% 
  mutate(p = purrr::map(fit_log, broom::tidy)) %>% 
  unnest(p) %>% 
  dplyr::select(1:6) %>% 
  tidyr::spread(term, estimate) 

log_pars <- mod_pars %>% dplyr::select(.id, b, c, d, e)

# plot parameters
pd <- mod_pars %>% 
  pivot_longer(cols=b:e, names_to = "par", values_to = "value")
ggplot(pd)+
  geom_boxplot(aes(y=value, x=label))+
  facet_wrap(~par, scales = "free")

# ======================================================================================================== -

# Full model ----

# reshape data for modelling
data_reshape <- all_means %>% 
  # do not require standard deviation
  dplyr::select(-sd) %>% 
  # create new variable
  mutate(var = paste(channel, color_space, posY, type, sep = "_")) %>% 
  # drop old separated variables 
  dplyr::select(-channel, -color_space, -posY, -type) %>% 
  pivot_wider(names_from = "var", values_from = "mean")

drop <- paste(paste0(as.character(seq(-30, 37, by = 2)), "_"), collapse = "|")
names_drop <- grep(drop, names(data_reshape), value = TRUE)

data_mod_red <- data_reshape %>% 
  # increase the distance between profile neighbouring pixels
  dplyr::select(-one_of(names_drop)) %>% 
  # these channels appear to be redundant (perfectly correlated with another channel)
  dplyr::select(-starts_with("L_Lab"), -contains("YCC"))

# add model parameters as predictors
# (join by .id)
d_mod <- data_mod_red %>% 
  full_join(., bp_pars_v, by = ".id") %>% 
  full_join(., bp_pars_R, by = ".id") %>% 
  full_join(., log_pars, by = ".id") %>% 
  dplyr::select(-.id)

template <- d_mod %>% dplyr::select(-label) %>% slice(1)
write_csv(template, "Z:/Public/Jonas/001_LesionZoo/TestingData/template_varnames_v4.csv")

# # test without model pars
# d_mod <- data_mod_red %>% 
#   dplyr::select(-.id)

# ======================================================================================================== -

# > pls ----

# train and validate 
indx <- createMultiFolds(d_mod$label, k = 10, times = 5)
ctrl <- caret::trainControl(method = "repeatedcv", 
                            index = indx,
                            classProbs = TRUE,
                            savePredictions = TRUE,
                            verboseIter = TRUE,
                            selectionFunction = "oneSE")
tune_length <- 20
plsda <- train(label ~., 
               data = d_mod, 
               preProc = c("center", "scale"),
               method = "pls",
               tuneLength = tune_length, 
               trControl = ctrl,
               returnResamp = "all")
plot(plsda)
imp <- varImp(plsda)
importance <- imp$importance

## SAVE ¨MODEL FOR IMPORT IN PYTHON

MODEL_SAVE_PATH = "Output/Models/spl/pls_v4"
DEP_LIBS = c("C:/Users/anjonas/RLibs/caret", "C:/Users/anjonas/RLibs/pls")

# save
model_rds_path = paste(MODEL_SAVE_PATH, ".rds", sep='')
model_dep_path = paste(MODEL_SAVE_PATH, ".dep", sep='')

# save model
# dir.create(dirname(model_path), showWarnings=FALSE, recursive=TRUE)
saveRDS(plsda, model_rds_path)

# save dependency list
file_conn <- file(model_dep_path)
writeLines(DEP_LIBS, file_conn)
close(file_conn)

# ======================================================================================================== -

# > random forest ----
#specify model tuning parameters
mtry <- c(3, 5, 7, 9, 12, 15)
min_nodes <- c(2, 3, 5, 8)
tune_grid <- expand.grid(mtry = mtry,
                         splitrule = "gini", # default
                         min.node.size = min_nodes)

rf_ranger <- caret::train(label ~ .,
                          data = d_mod,
                          preProc = c("center", "scale"),
                          method = "ranger",
                          tuneGrid = tune_grid,
                          importance = "permutation",
                          num.trees = 500,
                          trControl = ctrl)
plot(rf_ranger)
imp <- varImp(rf_ranger)
imp$importance

predobs_cv <- plyr::match_df(plsda$pred, plsda$bestTune, on = names(plsda$bestTune))

#Average predictions of the held out samples
predobs <- predobs_cv %>% 
  mutate(check = ifelse(pred == obs, 1, 0)) %>% 
  group_by(rowIndex) %>% 
  summarize(obs = sum(check)) %>% 
  mutate(obs = obs/5)

# misspredicted samples
misspred <- which(predobs$obs %in% c(0.0, 0.1, 0.2))
miss_pred <- data_reshape[misspred, ".id"]

# stability across resamples
ggplot(predobs) +
  geom_histogram(aes(x = obs), stat = "count") +
  theme_bw() +
  theme(panel.grid = element_blank())

# class probabilities
ggplot(predobs_cv) +
  geom_density(aes(x = pos)) +
  theme_bw() +
  theme(panel.grid = element_blank())
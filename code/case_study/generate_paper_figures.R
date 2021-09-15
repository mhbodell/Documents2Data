library(data.table)
library(dplyr)
library(ROCR)
library(pROC)
library(randomForest)
library(xgboost)
library(ggplot2)
library(ggpubr)
#-----------------
# FUNCTIONS

f1 <- function(confusion_matrix, 
               threshold = 0.5,
               var_name= 'IsEditorialContent',
               lexical = FALSE,
               model = 'logit',
               confusion_matrix_name = 'cm_logit_section_pc_lexical'){
  
  if(exists(confusion_matrix_name)){
    precision <- confusion_matrix[2,2]/sum(confusion_matrix[1:2,2]) 
    recall <- confusion_matrix[2,2]/sum(confusion_matrix[2,1:2])
    f1 <- 2 * (precision * recall) / (precision + recall)
    
    res <- data.frame(Model = model,
                      var_name = var_name,
                      lexical = lexical,
                      f1 = f1)
  } else {
    res <- data.frame(Model = model,
                      var_name = var_name,
                      lexical = lexical,
                      f1 = NA)
  }
  
  res$f1 <- ifelse(is.nan(res$f1),0,res$f1)
  
  return(res )
  
}
# accuracy and F1-score funtcion 
acc_f1 <- function(true_labels, 
                   predictions, 
                   plot_roc = FALSE, 
                   threshold = 0.5,
                   var_name= 'IsEditorialContent'){
  predicted_labels <- ifelse(predictions >= threshold,1,0)
  confusion_matrix <- table(true = true_labels, pred = predicted_labels)
  acc <- sum(diag(confusion_matrix))/sum(confusion_matrix)
  precision <- confusion_matrix[2,2]/sum(confusion_matrix[1:2,2]) 
  recall <- confusion_matrix[2,2]/sum(confusion_matrix[2,1:2])
  f1 <- 2 * (precision * recall) / (precision + recall)
  
  
  
  print(paste0('Accuracy: ', round(acc, digits = 3)))
  print(paste0('F1-score: ', round(f1, digits = 3)))
  
  if(plot_roc == TRUE) {
    roc_pred <- ROCR::prediction(predictions = predictions,labels = true_labels)
    perf <- ROCR::performance(roc_pred,"tpr","fpr")
    plot(perf, colorize=TRUE, main = paste0('ROC, ', var_name))
    auc_ROCR <- ROCR::performance(roc_pred, measure = "auc")
    print(paste0('AUC: ', round(auc_ROCR@y.values[[1]], digits = 3)))
  } else {
    roc_pred <- ROCR::prediction(predictions = predictions,labels = true_labels)
    auc_ROCR <- ROCR::performance(roc_pred, measure = "auc")
    print(paste0('AUC: ', round(auc_ROCR@y.values[[1]], digits = 3)))
    
  }
  
  return(confusion_matrix)
  
  
}


# normal approximation
get_error_bands <- function(p_hat, n){
  return(sqrt((p_hat*(1-p_hat))/n))
}

#----------------------------------#
# FIGURE 7D (META-DATA ERROR)   ----
#----------------------------------#

## READ IN DATA

where <- 'miriam'
if(where=='miriam'){
  base_path <- '/Users/mhbodell/Documents/git/Documents2Data/'
  train_data <- fread(paste0(base_path,'data/train_data_kbqual.csv'), stringsAsFactors = TRUE)
  test_data <- fread(paste0(base_path,'data/test_data_kbqual.csv'), stringsAsFactors = TRUE)
  model_path <- paste0(base_path, 'code/models/')
  save_path <- paste0(base_path,'figs/')
}


#---------------------------------------
## READ IN MODELS
i <-10
m_editorial_logistic <- readRDS(paste0(model_path,'logistic_editorial_',i,'.rds'))
m_bodytext_logistic <- readRDS(paste0(model_path,'logistic_bodytext_',i,'.rds'))
m_section_logistic <- readRDS(paste0(model_path,'logistic_section_',i,'.rds'))

m_editorial_rf <- readRDS(paste0(model_path,'rf_editorial_',i,'.rds'))
m_bodytext_rf <- readRDS(paste0(model_path,'rf_bodytext_',i,'.rds'))
m_section_rf <- readRDS(paste0(model_path,'rf_section_',i,'.rds'))


m_editorial_xgb <- xgb.load(paste0(model_path,'xgb_editorial_',i,'.model'))
m_bodytext_xgb <- xgb.load(paste0(model_path,'xgb_bodytext_',i,'.model'))
m_section_xgb <- xgb.load(paste0(model_path,'xgb_section_',i,'.model'))

# PLOT RESULTS 

set.seed(901207)
th <- 0.5


# IsEditorial ----
# test data - evaluation
preds <- predict(m_editorial_logistic, newdata = test_data, type = 'response')
cm_logit_editorial <- acc_f1(true_labels = test_data$IsEditorialContent[!is.na(preds)], 
                             predictions = preds[!is.na(preds)],
                             plot_roc = FALSE,
                             var_name = 'IsEditorialContent',
                             threshold = th)

# RF - linear
# test data - evaluation
preds<- predict(m_editorial_rf, newdata = test_data, type = 'prob')[,2]
cm_rf_editorial_linear <- acc_f1(true_labels = test_data$IsEditorialContent[!is.na(preds)], 
                                 predictions = preds[!is.na(preds)],
                                 plot_roc = FALSE,
                                 var_name = 'IsEditorialContent',
                                 threshold = th)


# XGBOOST -
# train data - evaluation
f <- as.formula(as.factor(IsEditorialContent) ~ numbers10 + capital_character_ratio + x1 + y1 + id_page + number_character_ratio + textblocks_on_page + part + n_words + n_characters + n_numbers  + topy1 + y_dist_up + x_dist_left + x_dist_right + spec_char + paper + year + bold + italic + n_capital_chars + placement +  height + width + underline + words_area_ratio)
inds <- labels(terms(f))

# test data - evaluation
X_test <- data.matrix(test_data[, ..inds])
preds <- predict(m_editorial_xgb, newdata = X_test)
cm_xgb_editorial_linear <- acc_f1(true_labels = test_data$IsEditorialContent, 
                                  predictions = preds,
                                  plot_roc = FALSE,
                                  var_name = 'IsEditorialContent',
                                  threshold = th)




# IsBodyText ----

# logistic 
# test data - evaluation
preds <- predict(m_bodytext_logistic, newdata = test_data, type = 'response')
cm_logit_bodytext <- acc_f1(true_labels = test_data$IsBodyText[!is.na(preds)], 
                            predictions = preds[!is.na(preds)],
                            plot_roc = FALSE,
                            var_name = 'IsBodyText',
                            threshold = th)




# Random Forest 
# test data - evaluation
preds <- predict(m_bodytext_rf, newdata = test_data, type = 'prob')[,2]
cm_rf_bodytext_linear <- acc_f1(true_labels = test_data$IsBodyText[!is.na(preds)], 
                                predictions = preds[!is.na(preds)],
                                plot_roc = FALSE,
                                var_name = 'IsBodyText',
                                threshold = th)





# XGBOOST 
# test data - evaluation
f <- as.formula(as.factor(IsBodyText) ~  numbers10 + capital_character_ratio + x1 + y1 + id_page + number_character_ratio + textblocks_on_page + part + n_words + n_characters + n_numbers  + topy1 + y_dist_up + x_dist_left + x_dist_right + spec_char + paper + year + bold + italic + n_capital_chars + placement +  height + width + underline + words_area_ratio)
inds <- labels(terms(f))
X_test <- data.matrix(test_data[, ..inds])
preds <- predict(m_bodytext_xgb, newdata = X_test)
cm_xgb_bodytext_linear <- acc_f1(true_labels = test_data$IsBodyText, 
                                 predictions = preds,
                                 plot_roc = FALSE,
                                 var_name = 'IsBodyText',
                                 threshold = th)


# IsSection ----

# logistic
# test data - evaluation
preds <- predict(m_section_logistic, newdata = test_data, type = 'response')
cm_logit_section  <- acc_f1(true_labels = test_data$IsSectiontitle[!is.na(preds)], 
                            predictions = preds[!is.na(preds)],
                            plot_roc = FALSE,
                            var_name = 'IsSectiontitle',
                            threshold = th)


# Random Forest
# test data - evaluation
preds <- predict(m_section_rf, newdata = test_data, type = 'prob')[,2]
cm_rf_section_linear <- acc_f1(true_labels = test_data$IsSectiontitle[!is.na(preds)], 
                               predictions = preds[!is.na(preds)],
                               plot_roc = FALSE,
                               var_name = 'IsSectiontitle',
                               threshold = th)


# XGBOOST -
# test data - evaluation
f <- as.formula(as.factor(IsSectiontitle) ~ numbers10 + capital_character_ratio + x1 + y1 + id_page + number_character_ratio + textblocks_on_page + part + n_words + n_characters + n_numbers  + topy1 + y_dist_up + x_dist_left + x_dist_right + spec_char + paper + year + bold + italic + n_capital_chars + placement +  height + width + underline + words_area_ratio)
inds <- labels(terms(f))
X_test <- data.matrix(test_data[, ..inds])
preds <- predict(m_section_xgb, newdata = X_test)
cm_xgb_section_linear <- acc_f1(true_labels = test_data$IsSectiontitle, 
                                predictions = preds,
                                plot_roc = FALSE,
                                var_name = 'IsSectiontitle',
                                threshold = th)





# Non-Lexical
m0 <- f1(cm_logit_editorial, var_name = 'Task 1', lexical = FALSE, model = 'LR', confusion_matrix_name = 'cm_logit_editorial')
m1 <- f1(cm_logit_bodytext, var_name = 'Task 2', lexical = FALSE, model = 'LR', confusion_matrix_name = 'cm_logit_bodytext')
m2 <- f1(cm_logit_section, var_name = 'Task 3', lexical = FALSE, model = 'LR', confusion_matrix_name = 'cm_logit_section')

m3 <- f1(cm_rf_editorial_linear, var_name = 'Task 1', lexical = FALSE, model = 'RF', confusion_matrix_name = 'cm_rf_editorial_linear')
m4 <- f1(cm_rf_bodytext_linear, var_name = 'Task 2', lexical = FALSE, model = 'RF', confusion_matrix_name = 'cm_rf_bodytext_linear')
m5 <- f1(cm_rf_section_linear, var_name = 'Task 3', lexical = FALSE, model = 'RF', confusion_matrix_name = 'cm_rf_section_linear')

m6 <- f1(cm_xgb_editorial_linear, var_name = 'Task 1', lexical = FALSE, model = 'XGBoost', confusion_matrix_name = 'cm_xgb_editorial_linear')
m7 <- f1(cm_xgb_bodytext_linear, var_name = 'Task 2', lexical = FALSE, model = 'XGBoost', confusion_matrix_name = 'cm_xgb_bodytext_linear')
m8 <- f1(cm_xgb_section_linear, var_name = 'Task 3', lexical = FALSE, model = 'XGBoost', confusion_matrix_name = 'cm_xgb_section_linear')


df <- dplyr::bind_rows(m0,m1,m2,m3,m4,m5, m6,m7,m8)

# textblocks independent
df$sd_error <-  mapply(get_error_bands,df$f1, nrow(unique(test_data[,c('packageID','page')])))
df$error_high <- df$f1 + (1.96 * df$sd_error)
df$error_low <- df$f1 - (1.96 * df$sd_error)


df %>%
  filter(lexical == FALSE) %>%
  mutate(var_name= factor(var_name)) %>%
  ggplot(aes(y = f1, 
             x = var_name, 
             fill = Model)) +
  geom_bar(color = 'black',
           stat="identity", 
           position = "dodge") +
  labs(x = '',
       y = '',
       fill = '') + 
  theme_bw(base_size = 32) +
  guides(fill = guide_legend(nrow = 1)) +
  theme(legend.position=c(0.4,0.96),
        legend.background = element_rect(fill = NA)) + 
  scale_y_continuous(limits = c(0,1),
                     breaks = seq(0,1, by = 0.2)) +
  scale_fill_manual(values = c("LR" = 'darkgrey', "RF" = 'grey45',"XGBoost" = 'grey87')) +
  geom_errorbar(aes(x = var_name, 
                    ymin = error_low, 
                    ymax = error_high), 
                width = .1,
                position = position_dodge(width = .9))  -> metadata_error
metadata_error


#--------------------------------------------------#
# FIGURE (8A-C) (TEXT2DOX + IMAGE-2-TEXT ERROR) ----
#--------------------------------------------------#

# put up paths
where <- 'KB'
if(where == 'KB'){
  base_path2 <- '/data/miriam/kb_quality_paper/Miriam_20190815/manual_content/'
} 

count_path <- 'counts/'
#compatison_path <- 
ocr_path <- paste0(base_path2,count_path,'ocr_csv/')
visual_path <- paste0(base_path2,count_path,'visual_csv/')
text_path <- paste0(base_path2, 'comparison/')


# extract files
ocr_files <- list.files(ocr_path)
visual_files <- list.files(visual_path)
text_files <- list.files(text_path)


# Compare text quality

# create empty data frame
df <- data.frame(file_name = rep(NA, length(ocr_files)),
                 n_article = rep(NA, length(ocr_files)),
                 crossing_article_boundry = rep(NA, length(ocr_files)),
                 crossing_stycke_boundry = rep(NA, length(ocr_files)),
                 exact_word_accuracy = rep(NA, length(ocr_files)),
                 full_fuzzy_word_accuracy = rep(NA, length(ocr_files)),
                 preproc_fuzzy_word_accuracy = rep(NA, length(ocr_files)),
                 exact_character_accuracy = rep(NA, length(ocr_files)),
                 raw_levenshtein = rep(NA, length(ocr_files)),
                 relative_levenshtein = rep(NA, length(ocr_files)),
                 relative_preprocessed_levenshtein = rep(NA, length(ocr_files)),
                 manual_full_text = rep(NA, length(ocr_files)),
                 ocr_full_text = rep(NA, length(ocr_files)),
                 manual_preproc_text = rep(NA, length(ocr_files)),
                 ocr_preproc_text = rep(NA, length(ocr_files)),
                 nchar_ocr = rep(NA, length(ocr_files)),
                 nchar_manual = rep(NA, length(ocr_files)),
                 nwords_ocr = rep(NA, length(ocr_files)),
                 nwords_manual = rep(NA, length(ocr_files)),
                 full_fuzzy_word_n = rep(NA, length(ocr_files)),
                 preproc_fuzzy_word_n = rep(NA, length(ocr_files)))

# fill in for each sample
for(i in c(1:length(ocr_files))) {
  print(i)
  visual_file <- read.csv(paste0(visual_path,visual_files[i]),
                          header = TRUE,
                          sep = ';')
  
  ocr_file <- read.csv(paste0(ocr_path,ocr_files[i]),
                       header = TRUE,
                       sep = ';')
  
  multiple_article <- sum(ifelse(stringr::str_detect(ocr_file$ArticleID,'&|-'),1,0), na.rm = TRUE)
  multiple_stycke <- sum(ifelse(stringr::str_detect(ocr_file$StyckeID,'&|-'),1,0), na.rm = TRUE)
  narticle <- length(unique(visual_file$ArticleID[!is.na(visual_file$ArticleID)]))
  #nparagraph <- length(unique(visual_file$StyckeID[!is.na(visual_file$StyckeID)]))
  
  if(where == 'home' & text_files[i]=="DAGENS NYHETER 1945-12-16_upplaga-0_pagenr-28_paragraph13.csv"){
    text_file <- read.table(text = readLines(paste0(text_path,text_files[i]), warn = FALSE), 
                            sep = ';', 
                            header = TRUE,
                            stringsAsFactors = FALSE)
    
  } else {
    text_file <- read.csv(paste0(text_path,text_files[i]),
                          header = TRUE,
                          sep = ',',
                          stringsAsFactors = FALSE)
    
  }
  
  # full text
  manual_full_text <- text_file$manual_text
  ocr_full_text <- text_file$ocr_text
  nchar_manual <- nchar(manual_full_text)
  nchar_ocr <- nchar(ocr_full_text)
  nwords_manual <- length(stringr::str_split(manual_full_text, ' ')[[1]])
  nwords_ocr <- length(stringr::str_split(ocr_full_text, ' ')[[1]])
  
  
  # remove special characters
  manual_preproc_text <- gsub("\\s+"," ",stringr::str_replace_all(manual_full_text, "[[:punct:]]", " "))
  ocr_preproc_text <- gsub("\\s+"," ",stringr::str_replace_all(ocr_full_text, "[[:punct:]]", " "))
  nchar_preproc_manual <- nchar(manual_preproc_text)
  nchar_preproc_ocr <- nchar(ocr_preproc_text)
  
  # calculate levenshtein for full text
  raw_levenshtein <- adist(manual_full_text,
                           ocr_full_text)
  relative_levenshtein <- raw_levenshtein/max(nchar_manual,nchar_ocr)
  
  # calculate levenshtein for preprocessed text
  raw_preproc_levenshtein <- adist(manual_preproc_text,
                                   ocr_preproc_text)
  relative_preproc_levenshtein <- raw_preproc_levenshtein/max(nchar_preproc_manual,nchar_preproc_ocr)
  
  # right word, wrong position - full
  split_manual <- stringr::str_split(manual_full_text,' ')[[1]]
  split_manual <- split_manual[split_manual!='']
  split_ocr <- stringr::str_split(ocr_full_text,' ')[[1]]
  split_ocr <- split_ocr[split_ocr!='']
  full_fuzzy_word_accuracy <- length(which(split_ocr%in%split_manual))/max(length(split_ocr), length(split_manual))
  full_fuzzy_word_n <- length(which(split_ocr%in%split_manual))
  
  # right word, wrong position - preprocessed
  preproc_split_manual <- stringr::str_split(manual_preproc_text,' ')[[1]]
  preproc_split_manual <- preproc_split_manual[preproc_split_manual!='']
  preproc_split_ocr <- stringr::str_split(ocr_preproc_text,' ')[[1]]
  preproc_split_ocr <- preproc_split_ocr[preproc_split_ocr!='']
  
  preproc_fuzzy_word_accuracy <- length(which(preproc_split_ocr%in%preproc_split_manual))/max(length(preproc_split_ocr), length(preproc_split_manual))
  
  preproc_fuzzy_word_n <- length(which(preproc_split_ocr%in%preproc_split_manual))
  
  df$file_name[i] <- ocr_files[i]
  df$n_article[i] <- narticle
  df$crossing_article_boundry[i] <-  multiple_article
  df$crossing_stycke_boundry[i] <- multiple_stycke
  df$exact_word_accuracy[i] <- text_file$word_accuracy
  df$preproc_fuzzy_word_accuracy[i] <- preproc_fuzzy_word_accuracy 
  df$full_fuzzy_word_accuracy[i] <- full_fuzzy_word_accuracy 
  df$full_fuzzy_word_n[i] <- full_fuzzy_word_n
  df$preproc_fuzzy_word_n[i] <- preproc_fuzzy_word_n
  #df$nparagrap
  
  if('charachter_accuracy'%in%names(text_file)){
    df$exact_character_accuracy[i] <- text_file$charachter_accuracy
  } else {
    df$exact_character_accuracy[i] <- text_file$character_accuracy
  }
  df$raw_levenshtein[i] <- raw_levenshtein[1,1]
  df$relative_levenshtein[i] <- relative_levenshtein[1,1]
  df$relative_preprocessed_levenshtein[i] <- relative_preproc_levenshtein[1,1]
  df$manual_full_text[i] <- manual_full_text
  df$ocr_full_text[i] <- ocr_full_text
  df$manual_preproc_text[i] <- manual_preproc_text
  df$ocr_preproc_text[i] <- ocr_preproc_text
  df$nchar_ocr[i] <- nchar_ocr
  df$nchar_manual[i] <- nchar_manual
  df$nwords_ocr[i]  <- nwords_ocr
  df$nwords_manual[i]  <- nwords_manual
}

df$year <- unlist(lapply(df$file_name, function(x) regmatches(x, gregexpr("[[:digit:]]+", x))[[1]][1]))
df

df %>%
  mutate(newspaper = substring(file_name,1,2)) -> df



# textual quality - Levenshtein (Fig 7B)
ggplot(df) +
  geom_point(aes(x = as.numeric(year), 
                 y = relative_levenshtein, 
                 fill = 'Raw'), 
             size = 4, pch = 21)  +
  scale_x_continuous(breaks=seq(1945,2019,20), 
                     labels = seq(1945,2019,20)) +
  scale_y_continuous(labels = scales::percent) +
  labs(y = '',  fill = '',  x = '') + 
  theme_bw() +
  theme(legend.position="none",
        text = element_text(size=32)) +
  scale_fill_manual(values=c("Raw"="lightgrey", "Preprocessed"="black"), 
                    labels=c("Raw", "Preprocessed")) -> ocr_error_levenshtein

# textual quality - Word recognition (Fig 7C)
ggplot(df) +
  geom_point(aes(x = as.numeric(year), 
                 y = full_fuzzy_word_accuracy, fill = 'Raw'), 
             size = 4, pch = 21)  +
  scale_x_continuous(breaks=seq(1945,2019,20), 
                     labels = seq(1945,2019,20)) +
  scale_y_continuous(labels = scales::percent) +
  labs( y = '', fill = '',  x = '') + 
  theme_bw() +
  theme(legend.position="none",
        text = element_text(size=32)) +
  scale_fill_manual(values=c("Raw"="lightgrey", "Preprocessed"="black"), 
                    labels=c("Raw", "Preprocessed")) -> ocr_error_word_recognition


# Text-to-documents error (Fig 7A)
ggplot(df) +
  geom_point(aes(x = as.numeric(year), 
                 y = crossing_article_boundry), 
             size = 4)  +
  scale_x_continuous(breaks=seq(1945,2019,20), 
                     labels = seq(1945,2019,20)) +
  labs(y = '', fill = '', x = '') +
  theme_bw() +
  geom_vline(xintercept = 1970, col = 'red', linetype = 'dashed', size = 1) +
  theme(legend.position="bottom",
        text = element_text(size=32)) -> segmentation_error

# combine plots
tre_plot <-  ggarrange(segmentation_error,
                       ocr_error_levenshtein,
                       ocr_error_word_recognition,
                       metadata_error,
                       labels = c('A','B',
                                  'C','D'),
                       nrow = 2, ncol = 2,
                       font.label = list(size = 30))
tre_plot
ggsave(tre_plot, 
       file = paste0(save_path, 'text_representation_errors.png'),
       height = 40,
       width = 40,
       units = 'cm')


#---------------------------------------------------#
#                 FIGURE 8                       ----
#---------------------------------------------------#


fig_data_path <- paste0(base_path,'data/')

#----------------------------------#
# FIGURE 8A (COVERAGE ERROR)    ----
#----------------------------------#

# number of text blocks
raw0_dn_tb <- fread(paste0(fig_data_path, 'raw0_n_per_title_dn.csv'))
raw0_svd_tb <- fread(paste0(fig_data_path, 'raw0_n_per_title_svd.csv'))
raw0_afb_tb <- fread(paste0(fig_data_path, 'raw0_n_per_title_afb.csv'))
raw0_exp_tb <- fread(paste0(fig_data_path, 'raw0_n_per_title_exp.csv'))

raw0_tb <- rbind(raw0_dn_tb,raw0_svd_tb,raw0_afb_tb,raw0_exp_tb)
raw0_tb[, year:=lubridate::year(date)]
raw0_tb <- raw0_tb[,sum(N), by = c('year','paper')]
raw0_tb[,paper:=ifelse(substr(paper,start = 1, stop = 1)=='A','Aftonbladet',
                       ifelse(substr(paper,start = 1, stop = 1)=='E', 'Expressen',
                              ifelse(substr(paper,start = 1, stop = 1)=='S','Svenska Dagbladet','Dagens Nyheter')))]
names(raw0_tb) <- c('year','paper','N_tb')

ggplot(raw0_tb, aes(x = year, y = N_tb/1000000, col = paper)) +
  geom_line(size = 3) +
  theme_bw(base_size = 24) +
  scale_color_grey() +
  guides(linetype = FALSE) +
  scale_y_continuous(labels = scales::comma) +
  theme(legend.position="bottom") + 
  labs(x = '',
       y = '',
       color = '') -> coverage_error_textblocks
coverage_error_textblocks

# OPERATIONALIZATION ERROR
salience_strict <- fread(paste0(fig_data_path, 'salience_strict.csv'))
salience_weak <- fread(paste0(fig_data_path, 'salience_weak.csv'))

strict <- salience_strict[,sum(N), by = 'year']
strict[,type := 'Strict']
weak <- salience_weak[,sum(N), by = 'year']
weak[,type := 'Weak']
df <- data.table(dplyr::bind_rows(strict,weak))

ggplot(df, aes(x = year, y =V1/1000, col = type)) +
  geom_line(size = 3) +
  theme_bw(base_size = 24) +
  scale_color_grey() +
  #scale_y_continuous(labels = scales::percent) + 
  theme(legend.position=c(0.2,0.9),
        legend.background = element_rect(fill = NA)) + 
  labs(x = '',
       y = '', #Topic salience (counts)
       color = '') -> operationalization_error
operationalization_error


# peak change
strict[,lag1 := dplyr::lag(V1, n = 1)]
weak[,lag1 := dplyr::lag(V1, n = 1)]

strict[,diff := (V1-lag1)/lag1]
weak[, diff := (V1-lag1)/lag1]
rbind(strict,weak) %>%
  ggplot(aes(x = year, y = diff, col = type)) +
  geom_line(size = 2, type = 'dotted') +
  theme_bw(base_size = 30) +
  scale_color_grey() +
  #scale_y_continuous(labels = scales::percent) + 
  theme(legend.position=c(0.2,0.9),
        legend.background = element_rect(fill = NA)) + 
  labs(x = '',
       y = '', #Topic salience (counts)
       color = '') -> operationalization_error2


rie_plot <- ggarrange(coverage_error_textblocks + theme(legend.position = 'bottom'),
                      operationalization_error  + theme(legend.position = 'bottom'),
                      labels = c('A','B'),
                      nrow = 1, ncol = 2,
                      font.label = list(size = 28),
                      common.legend = FALSE, 
                      legend="bottom")
rie_plot
ggsave(rie_plot, 
       file = paste0(save_path, 'research_inference_errors.png'),
       height = 30,
       width = 55,
       units = 'cm')



#-------------------------------------------#
#----               FIGURE 9             ----
#-------------------------------------------#


#   LOAD MODELS      
where <- 'miriam'
if(where=='miriam'){
  base_path <- '/Users/mhbodell/Documents/git/Documents2Data/'
  model_path <- paste0(base_path,'code/models/')
  train_data <- fread(paste0(base_path,'data/train_data_kbqual.csv'), stringsAsFactors = TRUE)
  test_data <- fread(paste0(base_path,'data/test_data_kbqual.csv'), stringsAsFactors = TRUE)
  save_path <- paste0(base_path,'figs/')
  
}


# read in models
m_logistic <- readRDS(paste0(model_path,'logistic_editorial_10.rds'))
rf <- readRDS(paste0(model_path, 'rf_editorial_10.rds'))
xgb <- xgb.load(paste0(model_path, 'xgb_editorial_10.model'))

m_logistic_bodytext <- readRDS(paste0(model_path, 'logistic_bodytext_10.rds'))
rf_bodytext_linear <- readRDS(paste0(model_path, 'rf_bodytext_10.rds'))
xgb_bodytext <- xgb.load(paste0(model_path, 'xgb_bodytext_10.model'))

m_logistic_section <- readRDS(paste0(model_path, 'logistic_section_10.rds'))
rf_section_linear <- readRDS(paste0(model_path, 'rf_section_10.rds'))
xgb_section <- xgb.load(paste0(model_path, 'xgb_section_10.model'))



# F1-scores  

floor_decade <- function(value){ return(value - value %% 10) }
# accuracy and F1-score funtcion 
acc_f1_t <- function(true_data = test_data, 
                     predictions, 
                     threshold = 0.5,
                     var_name= 'IsEditorialContent'){
  predicted_labels <- ifelse(predictions >= threshold,1,0)
  true_labels <-  pull(true_data[,..var_name])
  true_data[, decade:=floor_decade(year)]
  
  f1_years <- list()
  for(y in 1:length(unique(true_data$decade))){
    tmp_data <- true_data
    tmp_data[, pred_labels := predicted_labels]
    tmp_data[, true_labels := true_labels]
    tmp_year <- unique(true_data$decade)[y]
    tmp_data <- tmp_data[decade==tmp_year,]
    
    
    tmp_data[, tp := ifelse(pred_labels==true_labels & true_labels == 1,1,0)]
    tmp_data[, fp := ifelse(pred_labels>true_labels & true_labels == 0,1,0)]
    tmp_data[, tn := ifelse(pred_labels==true_labels & true_labels == 0,1,0)]
    tmp_data[, fn := ifelse(pred_labels<true_labels & true_labels == 1,1,0)]
    
    acc <- sum(tmp_data$tp,tmp_data$tn)/nrow(tmp_data)
    precision <- sum(tmp_data$tp)/sum(tmp_data$tp, tmp_data$fp)
    recall <- sum(tmp_data$tp)/sum(tmp_data$tp, tmp_data$fn)
    precision <-ifelse(is.nan(precision),0,precision)
    recall <- ifelse(is.nan(recall),0,recall)
    f1 <- 2 * (precision * recall) / (precision + recall)
    f1_years[[y]] <- data.table(year = tmp_year, 
                                f1 =  ifelse(is.nan(f1),0, f1), 
                                recall = recall, 
                                precision = precision,
                                acc = acc)
    
  }
  
  f1_years <- rbindlist(f1_years)
  f1_years <- f1_years[order(year),]
  f1_years[, type:=ifelse(var_name%like%'Section', 'Task 3',
                          ifelse(var_name%like%'Body','Task 2', 'Task 1'))]
  return(f1_years)
  
}

# user 0.5 as threshold
th <- 0.5


# section predictions
inds <- c('numbers10', 'capital_character_ratio', 'x1', 'y1', 'id_page', 'number_character_ratio', 'textblocks_on_page', 'part', 'n_words', 'n_characters', 'n_numbers', 'topy1', 'y_dist_up', 'x_dist_left', 'x_dist_right', 'spec_char', 'paper', 'year', 'bold', 'italic', 'n_capital_chars', 'placement', 'height', 'width', 'underline', 'words_area_ratio')
X_test <- data.matrix(test_data[, ..inds])
preds <- predict(xgb_section, newdata = X_test)

f1_section_xgb_t <- acc_f1_t(true_data = test_data[!is.na(preds),], 
                             predictions = preds,
                             var_name = 'IsSectiontitle',
                             threshold = 0.5)
plot(x = f1_section_xgb_t$year, f1_section_xgb_t$f1, type = 'l')

# editorial body text
preds <- predict(xgb_bodytext, newdata = X_test)
f1_bodytext_xgb_t <- acc_f1_t(true_data = test_data[!is.na(preds),], 
                              predictions = preds,
                              var_name = 'IsBodyText',
                              threshold = 0.5)
lines(x = f1_bodytext_xgb_t$year, f1_bodytext_xgb_t$f1, col = 'red', lty = 2)


# editorial content
preds <- predict(xgb, newdata = X_test)
f1_editorial_xgb_t <- acc_f1_t(true_data = test_data[!is.na(preds),], 
                               predictions = preds,
                               var_name = 'IsEditorialContent',
                               threshold = 0.5)
lines(x = f1_editorial_xgb_t$year, f1_editorial_xgb_t$f1, col = 'blue', lty = 3)

f1_t_plot <- dplyr::bind_rows(f1_editorial_xgb_t,f1_bodytext_xgb_t,f1_section_xgb_t)

ggplot(f1_t_plot, aes(x = year, y = f1, col = type, linetype = type)) +
  geom_line(size = 3) +
  labs(y = 'F1',
       x = 'Year',
       color = '') +
  scale_colour_grey() +
  theme_bw(base_size = 42) +
  theme(legend.position="bottom") +
  guides(linetype=FALSE) -> meta_data_error_t
meta_data_error_t

ggsave(filename = paste0(save_path,'meta_data_error_t.png'),
       plot = meta_data_error_t,
       height = 350, width = 500, units = 'mm')





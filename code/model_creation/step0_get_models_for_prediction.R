library(data.table)
library(dplyr)

library(randomForest)
library(xgboost)


# read in tagged data
base_path <- '/data/miriam/data_media_group_threat/' 
save_path <- '/data/miriam/data_watershed/models/'

# all labled files
files <- list.files(paste0(base_path,'classifier_csv/all_labeled/'))




#
files_train <- list.files(paste0(base_path,'classifier_csv/raw/'))
files_test <- list.files(paste0(base_path,'classifier_csv/raw_test/'))

files_train_gs <- paste0(base_path,'classifier_csv/all_labeled/',files[which(files%in%files_train)])
files_test_gs <-  paste0(base_path,'classifier_csv/all_labeled/',files[which(files%in%files_test)])

dts <- lapply(files_train_gs, function(f) read.csv(f, stringsAsFactors = FALSE))
dtsT <- lapply(files_test_gs, function(f) read.csv(f, stringsAsFactors = FALSE))


# remove file if has other structure
if(length(which(unlist(lapply(dts, function(x) ncol(x)))==1))>0){
  dts <- dts[-which(unlist(lapply(dts, function(x) ncol(x)))==1)]
}


# function for creating features
feature_extraction <- function(data){
  
  #  print(i)
  
  print(unique(data$newspaper_day_x))
  #  print(names(data))
  ymiddle <- max(data$y1)/2 * -1
  xmiddle <- max(data$x1)/2
  na_editorials <- which(is.na(data$IsEditorialContent))
  if(length(na_editorials)!=0){
    data <- data[-which(is.na(data$IsEditorialContent)),]
  }
  
  data %>%
    rename(title = newspaper_day_x) %>%
    mutate(x2 = x1 + width,
           y1 = y1 * -1 ,
           y2 = y1 - height,
           area = height*width,
           year = lubridate::year(as.Date(gsub(x = title, pattern = '[A-Z]', replacement = '')))) %>%
    mutate(n_characters = stringi::stri_count(content, regex = '[a-öA-Ö]'),
           n_capital_chars = stringi::stri_count(content, regex = '[[:upper:]]'),
           n_numbers = stringi::stri_count(content, regex = '[0-9]')) %>%
    rowwise() %>%
    mutate(n_words = length(stringr::str_split(string = content, pattern = ' ')[[1]]),
           placement = ifelse(x1 <= xmiddle & y1 <= ymiddle, 'upper_left','unknown'),
           placement = ifelse(x1 >= xmiddle & y1 <= ymiddle, 'upper_right',placement),
           placement = ifelse(x1 <= xmiddle & y1 >= ymiddle, 'lower_left',placement),
           placement = ifelse(x1 >= xmiddle & y1 >= ymiddle,'lower_right',placement),
           placement_p4 = ifelse(x2 <= xmiddle & y2 <= ymiddle, 'upper_left','unknown'),
           placement_p4 = ifelse(x2 > xmiddle & y2 <= ymiddle, 'upper_right',placement_p4),
           placement_p4 = ifelse(x2 <= xmiddle & y2 > ymiddle, 'lower_left',placement_p4),
           placement_p4 = ifelse(x2 > xmiddle & y2 > ymiddle,'lower_right',placement_p4),
           
           paper = ifelse(stringr::str_detect(string = title, pattern = 'AFTONBLADET'), 'afb', 
                          ifelse(stringr::str_detect(string = title, pattern = 'DAGENS NYHETER'),'dn',
                                 ifelse(stringr::str_detect(string = title, pattern = 'EXPRESSEN'),'exp','svd'))),
           paper = as.factor(paper),
           first5page = ifelse(part == 0 & page <= 5,1,0),
           editorial_bodytext = ifelse(IsEditorialContent==1 & IsBodyText == 1,1,0)) %>%
    ungroup() %>%
    tidyr::replace_na(list(IsSectiontitle = 0,
                           IsSubsectionTitle = 0,
                           IsEditorialContent = 0,
                           IsBodyText = 0,
                           IsTitle = 0,
                           IsOther = 0,
                           n_characters = 0,
                           n_numbers = 0,
                           n_words = 0,
                           n_capital_chars = 0,
                           editorial_bodytext = 0)) -> d
  
  files_fonts <- list.files(paste0(base_path,'classifier_csv_flerge/raw2/'))
  font_match <- files_fonts[which(files_fonts%like%unique(d$title))]
  if(length(font_match)!=0) {
    dt_font <- read.csv(paste0(base_path,'classifier_csv_flerge/raw2/',font_match), stringsAsFactors = FALSE) %>% select(-title, -X)
    d2 <- merge(x = d, y = dt_font, by.x = 'alto_zoneID', by.y = 'id')# %>% #select(-x1.y, -y1.y, -width.y, - heigth, -content.y) %>%
     if(nrow(d2)!=0){
      d <- d2
      d$font_larger_12 <- ifelse(unlist(lapply(d$font_size, function(x) sum(ifelse(as.numeric(stringr::str_extract_all(x, pattern = '[:digit:]{2}')[[1]])>12,1,0))))!=0,1,0)      
      d$font_larger_20 <- ifelse(unlist(lapply(d$font_size, function(x) sum(ifelse(as.numeric(stringr::str_extract_all(x, pattern = '[:digit:]{2}')[[1]])>20,1,0))))!=0,1,0)      
      d$bold <- ifelse(stringr::str_detect(tolower(d$font_style), pattern = 'bold'),1,0)
      d$italic <- ifelse(stringr::str_detect(tolower(d$font_style), pattern = 'italic'),1,0)
      d$underline <- ifelse(stringr::str_detect(tolower(d$font_style), pattern = 'underline'),1,0)
      d$sansserif <- ifelse(stringr::str_detect(tolower(d$font_type), pattern = 'sans-serif'),1,0)
    }
    
  }
  
  id_page <- stringr::str_match(d$alto_filename, "_(.*)_alto.xml")[,2]
  d$id_page <- as.numeric(substring(id_page, first = nchar(id_page)-3, last = nchar(id_page)))
  d$textblocks_on_page <- nrow(d)
  
  # lexical content --
  
  # commercial
  d$price_words <- unlist(lapply(d$content, function(ss) ifelse(sum(stringr::str_detect(strsplit(ss , ' ')[[1]], pattern =  stringr::regex('rea$', ignore_case = T)))>0,1,0)))
  d$price_words <- unlist(lapply(d$content, function(ss) ifelse(sum(stringr::str_detect(strsplit(ss , ' ')[[1]], pattern =  stringr::regex('rabatt$', ignore_case = T)))>0,1, d$price_words)))
  d$price_words <- unlist(lapply(d$content, function(ss) ifelse(sum(stringr::str_detect(strsplit(ss , ' ')[[1]], pattern =  stringr::regex('\\:\\-', ignore_case = T)))>0,1, d$price_words)))
  d$price_words <- unlist(lapply(d$content, function(ss) ifelse(sum(stringr::str_detect(strsplit(ss , ' ')[[1]], pattern =  stringr::regex('kr', ignore_case = T)))>0,1, d$price_words)))
  
  # special characters
  #d$spec_char <- unlist(lapply(d$content, function(ss) stringr::str_count(ss , pattern =  '\\*|\\•\\»|\\~|\\[|\\]|\\{|\\}|\\#|\\%|\\£')))
  spec_char <-  c(',', '.', ';', '?', '/', '\\', '`', '[', ']', '"', ':', '>', '<', '|', '-', '_', '=', '+', '(', ')', '^', '{', '}', '~', '\'', '*', '&', '%', '$', '!', '@', '#', '•','”','—','«','■','»')
  d$spec_char <- unlist(lapply(d$content, function(ss) sum(stringr::str_count(ss , pattern =  stringr::fixed(spec_char)))))
  
  
  # entertainment
  d$entertainment_words <- unlist(lapply(d$content, function(ss) ifelse(sum(stringr::str_detect(strsplit(ss , ' ')[[1]], pattern =  stringr::regex('PREMIÄR', ignore_case = F)))>0,1,0)))
  d$entertainment_words <- unlist(lapply(d$content, function(ss) ifelse(sum(stringr::str_detect(strsplit(ss , ' ')[[1]], pattern =  stringr::regex('biljetter', ignore_case = T)))>0,1, d$entertainment_words)))
  
  d$tv_words <- unlist(lapply(d$content, function(ss) ifelse(sum(stringr::str_detect(strsplit(ss , ' ')[[1]], pattern =  stringr::regex('TV', ignore_case = F)))>0,1,0)))
  
  # full ad page
  d$payed_page <- unlist(lapply(d$content, function(ss) ifelse(sum(stringr::str_detect(ss, pattern =  stringr::regex('hela sidan är betalad', ignore_case = T)))>0,1,0)))
  d$payed_page <- unlist(lapply(d$content, function(ss) ifelse(sum(stringr::str_detect(ss, pattern =  stringr::regex('hela sidan är en annons', ignore_case = T)))>0,1,d$payed_page)))
  d$payed_page <- unlist(lapply(d$content, function(ss) ifelse(sum(stringr::str_detect(ss, pattern =  stringr::regex('är en annons från', ignore_case = T)))>0,1,d$payed_page)))
  d$payed_page <- unlist(lapply(d$content, function(ss) ifelse(sum(stringr::str_detect(ss, pattern =  stringr::regex('hela denna tematidning är en annons från', ignore_case = T)))>0,1,d$payed_page)))
  d$payed_page <- unlist(lapply(d$content, function(ss) ifelse(sum(stringr::str_detect(ss, pattern =  stringr::regex('hela denna bilaga är en annons från', ignore_case = T)))>0,1,d$payed_page)))

  
  # section titles
  d$section_words <- unlist(lapply(d$content, function(ss) ifelse(sum(stringr::str_detect(strsplit(ss , ' ')[[1]], pattern =  stringr::regex('sport', ignore_case = T)))>0,1,0)))
  d$section_words <- unlist(lapply(d$content, function(ss) ifelse(sum(stringr::str_detect(strsplit(ss , ' ')[[1]], pattern =  stringr::regex('salu', ignore_case = T)))>0,1,d$section_words)))
  d$section_words <- unlist(lapply(d$content, function(ss) ifelse(sum(stringr::str_detect(strsplit(ss , ' ')[[1]], pattern =  stringr::regex('nöje', ignore_case = T)))>0,1,d$section_words)))
  d$section_words <- unlist(lapply(d$content, function(ss) ifelse(sum(stringr::str_detect(strsplit(ss , ' ')[[1]], pattern =  stringr::regex('börs', ignore_case = T)))>0,1,d$section_words)))
  d$section_words <- unlist(lapply(d$content, function(ss) ifelse(sum(stringr::str_detect(strsplit(ss , ' ')[[1]], pattern =  stringr::regex('ekonomi', ignore_case = T)))>0,1,d$section_words)))
  d$section_words <- unlist(lapply(d$content, function(ss) ifelse(sum(stringr::str_detect(strsplit(ss , ' ')[[1]], pattern =  stringr::regex('scen', ignore_case = T)))>0,1,d$section_words)))
  d$section_words <- unlist(lapply(d$content, function(ss) ifelse(sum(stringr::str_detect(strsplit(ss , ' ')[[1]], pattern =  stringr::regex('stan', ignore_case = T)))>0,1,d$section_words)))
  d$section_words <- unlist(lapply(d$content, function(ss) ifelse(sum(stringr::str_detect(strsplit(ss , ' ')[[1]], pattern =  stringr::regex('marknad', ignore_case = T)))>0,1,d$section_words)))
  d$section_words <- unlist(lapply(d$content, function(ss) ifelse(sum(stringr::str_detect(strsplit(ss , ' ')[[1]], pattern =  stringr::regex('nyheter', ignore_case = T)))>0,1,d$section_words)))
  d$section_words <- unlist(lapply(d$content, function(ss) ifelse(sum(stringr::str_detect(strsplit(ss , ' ')[[1]], pattern =  stringr::regex('kultur', ignore_case = T)))>0,1,d$section_words)))
  d$section_words <- unlist(lapply(d$content, function(ss) ifelse(sum(stringr::str_detect(strsplit(ss , ' ')[[1]], pattern =  stringr::regex('TV', ignore_case = T)))>0,1,d$section_words)))
  d$section_words <- unlist(lapply(d$content, function(ss) ifelse(sum(stringr::str_detect(strsplit(ss , ' ')[[1]], pattern =  stringr::regex('radio', ignore_case = T)))>0,1,d$section_words)))

  
  # closest y-dist, up
  d$y_dist_up <- 0
  for(i in 1:nrow(d)){
    neg_inds <- which(c(d$y1[i] - d$y2[-i])<0)
    if(length(neg_inds)>0){
      d$y_dist_up[i] <- max(c(d$y1[i] - d$y2[-i])[neg_inds])
    }
    neg_inds <- NULL
  }
  
  # closest x-dist, left
  d$x_dist_left <- 0
  for(i in 1:nrow(d)){
    pos_inds <- which(c(d$x1[i] - d$x2[-i])>0)
    if(length(pos_inds)>0){
      d$x_dist_left[i] <- min(c(d$x1[i] - d$x2[-i])[pos_inds])
    }
    pos_inds <- NULL
  }  
  
  # closest x-dist, rights
  d$x_dist_right <- 0
  for(i in 1:nrow(d)){
    neg_inds <- which(c(d$x1[i] - d$x2[-i])<0)
    if(length(neg_inds)>0){
      d$x_dist_right[i] <- max(c(d$x1[i] - d$x2[-i])[neg_inds])
    }
    pos_inds <- NULL
  }  
  
  
  #---

  d %>% mutate(x1 = as.numeric(x1),
               y1 =  as.numeric(y1),
               area = as.numeric(area),
               n_characters = as.numeric(n_characters),
               id_page = as.numeric(id_page),
               textblocks_on_page = as.numeric(textblocks_on_page),
               part = as.numeric(part),
               n_numbers = as.numeric(n_numbers),
               n_characters = as.numeric(n_characters),
               n_words = as.numeric(n_words),
               words_area_ratio = n_words/area,
               number_character_ratio = n_numbers/n_characters,
               number_character_ratio = ifelse(number_character_ratio==Inf, 10, number_character_ratio),
               capital_character_ratio = n_capital_chars/n_characters,
               capital_character_ratio = ifelse(capital_character_ratio==Inf, 0, capital_character_ratio),
               words_area_ratio = ifelse(words_area_ratio==Inf, 0, words_area_ratio),
               placement = as.factor(placement)) -> d
  
  # replace more NA
  d %>%
    tidyr::replace_na(list(capital_character_ratio = 0,
                           number_character_ratio = 0,
                           font_larger_12 = 12,
                           font_larger_20 = 20)) 
  
  return(d)
  
}



# create features for all observations
# train
fdts <- lapply(dts, function(x) feature_extraction(x))
data <- data.table::rbindlist(fdts, fill = TRUE)
raw_data <- data.table::rbindlist(dts, fill = TRUE)  %>% filter(!is.na(IsEditorialContent))


# test
fdtsT <- lapply(dtsT, function(x) feature_extraction(x))
dataT <- data.table::rbindlist(fdtsT, fill = TRUE)



# some data fix
data$bold <- ifelse(is.na(data$bold),0,data$bold)
data$sansserif <- ifelse(is.na(data$sansserif),0,data$sansserif)
data$italic <- ifelse(is.na(data$italic),0,data$italic)
data$topy1 <- ifelse(data$y1> -300, 1,0)
data$numbers10 <- ifelse(data$n_numbers>10,1,0)
data$section <- ifelse(data$IsSectiontitle == 1,1, ifelse(data$IsSubsectionTitle == 1,1,0))


dataT$bold <- ifelse(is.na(dataT$bold),0,dataT$bold)
dataT$sansserif <- ifelse(is.na(dataT$sansserif),0,dataT$sansserif)
dataT$italic <- ifelse(is.na(dataT$italic),0,dataT$italic)
dataT$topy1 <- ifelse(dataT$y1> -300, 1,0)
dataT$numbers10 <- ifelse(dataT$n_numbers>10,1,0)
dataT$section <- ifelse(dataT$IsSectiontitle == 1,1, ifelse(data$IsSubsectionTitle == 1,1,0))

# create train and test data
train_data <- data
test_data <- dataT
test_data[which(!test_data$IsEditorialContent%in%c(0,1)),] %>% pull(title) %>% unique()


f1 <- function(true_labels, 
               predictions, 
               threshold = 0.5){
  predicted_labels <- ifelse(predictions >= threshold,1,0)
  confusion_matrix <- table(true = true_labels, pred = predicted_labels)
  if(dim(confusion_matrix)[1]==2 & dim(confusion_matrix)[2] == 2){
    acc <- sum(diag(confusion_matrix))/sum(confusion_matrix)
    precision <- confusion_matrix[2,2]/sum(confusion_matrix[1:2,2]) 
    recall <- confusion_matrix[2,2]/sum(confusion_matrix[2,1:2])
    f1 <- 2 * (precision * recall) / (precision + recall)
  } else {
    f1 <- 0
  }
  
  
  return(f1)
  
  
}

datasize_fit <- function(train_data, 
                         test_data,
                         partitions = c(0.25,0.5,0.75,1),
                         model = 'logistic',
                         formula,
                         threshold = 0.5,
                         variable_name = 'IsEditorialContent'){
  
  res <- data.frame(partition = rep(NA, length(partitions)),
                    f1 = rep(NA, length(partitions)),
                    model_type = rep(model,length(partitions)))
  models <- vector('list', length(partitions))
  
  for(i in 1:length(partitions)){
    
    # create train data
    set.seed(123)
    train_ind <- sample(1:nrow(train_data),nrow(train_data)*partitions[i])
    tmp_data <- train_data[train_ind,]
    
    #
    if(model == 'logistic'){
      m <- glm(formula,
               data = tmp_data, 
               family = "binomial", 
               control = list(maxit = 1000)) 
      
      pred <- predict(m, newdata = test_data, type = 'response')
    }
    else if(model == 'random forest'){
      m <- randomForest(formula,
                        data = tmp_data, 
                        importance = TRUE,
                        ntree = 500,
                        na.action=na.exclude)
      
      pred <- predict(m, newdata = test_data, type = 'prob')[,2]
    }
    
    else if(model == 'random forest regularized'){
      m <- randomForest(formula,
                        data = tmp_data , 
                        importance = TRUE,
                        ntree = 500,
                        na.action=na.exclude,
                        sampsize = ceiling(.33*nrow(train_data)),
                        maxnodes = 100,
                        mtry = floor(sqrt(length(labels(terms(formula))))))
      
      pred <- predict(m, newdata = test_data, type = 'prob')[,2]     
      
    }
    
    else if(model == 'xgb'){
      # train data - evaluation
      inds <- labels(terms(formula))
      X <- data.matrix(tmp_data[, ..inds])
      X_test <- data.matrix(test_data[, ..inds])
      y <- as.integer(tmp_data[, ..variable_name] %>% pull(variable_name))
      
      m <- xgboost(data = data.matrix(X),
                   label = as.integer(y),
                   nrounds = 500,
                   nthread = 5,
                   max_depth = 6,
                   objective = "binary:logistic",
                   colsample = 0.5)
      
      
      
      pred <- predict(m, newdata = X_test)    
      
    }
    
    f1_tmp <- f1(true_labels = test_data %>% pull(variable_name), 
                 predictions = pred, 
                 threshold = threshold)
    
    res[i,1:2] <- c(partitions[i], f1_tmp)
    models[[i]] <- m
  }
  
  return(list(res,models))
  
}



#####################################
##    WITHOUT LEXICAL              ##
#####################################


# Editorial Content
logistic_formula_editorial <- as.formula(IsEditorialContent ~  spec_char + numbers10:y1 + numbers10 + n_words:area +  bold:font_larger_20 + n_numbers:capital_character_ratio:number_character_ratio + year:paper:n_words + capital_character_ratio + n_numbers:capital_character_ratio:textblocks_on_page:number_character_ratio + first5page:paper + id_page:textblocks_on_page + x1 + y1 + id_page + id_page:paper:year + x1:y1 + number_character_ratio + textblocks_on_page + part + n_words + n_numbers + textblocks_on_page:area:placement + x1:y1:font_larger_12 + topy1 + y_dist_up + x_dist_left + x_dist_right)
rf_formula_editorial <- as.formula(as.factor(IsEditorialContent) ~ numbers10 + capital_character_ratio + x1 + y1 + id_page + number_character_ratio + textblocks_on_page + part + n_words + n_characters + n_numbers  + topy1 + y_dist_up + x_dist_left + x_dist_right + spec_char + paper + year + bold + italic + n_capital_chars + placement +  height + width + underline + words_area_ratio)



editorial_logistic_res <- datasize_fit(train_data = train_data,
                                       test_data = test_data,
                                       partitions = seq(0.1,1.0, 0.1),
                                       model = 'logistic',
                                       formula = logistic_formula_editorial,
                                       threshold = 0.5,
                                       variable_name = 'IsEditorialContent')

editorial_rf_res <- datasize_fit(train_data = train_data,
                                 test_data = test_data,
                                 partitions = seq(0.1,1.0, 0.1),
                                 model = 'random forest',
                                 formula = rf_formula_editorial,
                                 threshold = 0.5,
                                 variable_name = 'IsEditorialContent')

editorial_xgb_res <- datasize_fit(train_data = train_data,
                                  test_data = test_data,
                                  partitions = seq(0.1,1.0, 0.5),
                                  #partitions = c(0.1,1),
                                  model = 'xgb',
                                  formula = rf_formula_editorial,
                                  threshold = 0.5,
                                  variable_name = 'IsEditorialContent') 


# EditorialBodyText
logistic_formula_bodytext <- as.formula(IsBodyText ~  spec_char + numbers10:y1 + numbers10 + n_words:area +  bold:font_larger_20 + n_numbers:capital_character_ratio:number_character_ratio + year:paper:n_words + capital_character_ratio + n_numbers:capital_character_ratio:textblocks_on_page:number_character_ratio + first5page:paper + id_page:textblocks_on_page + x1 + y1 + id_page + id_page:paper:year + x1:y1 + number_character_ratio + textblocks_on_page + part + n_words + n_numbers + textblocks_on_page:area:placement + x1:y1:font_larger_12 + topy1 + y_dist_up + x_dist_left + x_dist_right)
rf_formula_bodytext <- as.formula(as.factor(IsBodyText) ~  numbers10 + capital_character_ratio + x1 + y1 + id_page + number_character_ratio + textblocks_on_page + part + n_words + n_characters + n_numbers  + topy1 + y_dist_up + x_dist_left + x_dist_right + spec_char + paper + year + bold + italic + n_capital_chars + placement +  height + width + underline + words_area_ratio)


bodytext_logistic_res <- datasize_fit(train_data = train_data,
                                      test_data = test_data,
                                      partitions = seq(0.1,1.0, 0.1),
                                      model = 'logistic',
                                      formula = logistic_formula_bodytext,
                                      threshold = 0.5,
                                      variable_name = 'IsBodyText')

bodytext_rf_res <- datasize_fit(train_data = train_data,
                                test_data = test_data,
                                partitions = seq(0.1,1.0, 0.1),
                                model = 'random forest',
                                formula = rf_formula_bodytext,
                                threshold = 0.5,
                                variable_name = 'IsBodyText')


bodytext_xgb_res <- datasize_fit(train_data = train_data,
                                 test_data = test_data,
                                 partitions = seq(0.1,1.0, 0.1),
                                 #partitions = c(0.1,1),
                                 model = 'xgb',
                                 formula = rf_formula_bodytext,
                                 threshold = 0.5,
                                 variable_name = 'IsBodyText')

# IsSectionTitle
logistic_formula_section <- as.formula(IsSectiontitle ~ spec_char + numbers10:y1 + numbers10 + n_words:area +  bold:font_larger_20 + n_numbers:capital_character_ratio:number_character_ratio + year:paper:n_words + capital_character_ratio + n_numbers:capital_character_ratio:textblocks_on_page:number_character_ratio + first5page:paper + id_page:textblocks_on_page + x1 + y1 + id_page + id_page:paper:year + x1:y1 + number_character_ratio + textblocks_on_page + part + n_words + n_numbers + textblocks_on_page:area:placement + x1:y1:font_larger_12 + topy1 + y_dist_up + x_dist_left + x_dist_right)
rf_formula_section <- as.formula(as.factor(IsSectiontitle) ~ numbers10 + capital_character_ratio + x1 + y1 + id_page + number_character_ratio + textblocks_on_page + part + n_words + n_characters + n_numbers  + topy1 + y_dist_up + x_dist_left + x_dist_right + spec_char + paper + year + bold + italic + n_capital_chars + placement +  height + width + underline + words_area_ratio)



section_logistic_res <- datasize_fit(train_data = train_data,
                                     test_data = test_data,
                                     partitions = seq(0.1,1.0, 0.1),
                                     model = 'logistic',
                                     formula = logistic_formula_section,
                                     threshold = 0.5,
                                     variable_name = 'IsSectiontitle')

section_rf_res <- datasize_fit(train_data = train_data,
                               test_data = test_data,
                               partitions = seq(0.1,1.0, 0.1),
                               model = 'random forest',
                               formula = rf_formula_section,
                               threshold = 0.5,
                               variable_name = 'IsSectiontitle')



section_xgb_res <- datasize_fit(train_data = train_data,
                                test_data = test_data,
                                partitions = seq(0.1,1.0, 0.1),
                                model = 'xgb',
                                formula = rf_formula_section,
                                threshold = 0.5,
                                variable_name = 'IsSectiontitle')







#----------------
# SAVE MODELS
#----------------

# Logit
lapply(1:length(editorial_logistic_res[[2]]), function(i) saveRDS(editorial_logistic_res[[2]][[i]], paste0(save_path,'datasize_test/logistic_editorial_',i,'.rds')))
lapply(1:length(bodytext_logistic_res[[2]]), function(i) saveRDS(bodytext_logistic_res[[2]][[i]], paste0(save_path,'datasize_test/logistic_bodytext_',i,'.rds')))
lapply(1:length(section_logistic_res[[2]]), function(i) saveRDS(section_logistic_res[[2]][[i]], paste0(save_path,'datasize_test/logistic_section_',i,'.rds')))



# RF
lapply(1:length(editorial_rf_res[[2]]), function(i) saveRDS(editorial_rf_res[[2]][[i]], paste0(save_path,'datasize_test/rf_editorial_',i,'.rds')))
lapply(1:length(bodytext_rf_res[[2]]), function(i) saveRDS(bodytext_rf_res[[2]][[i]], paste0(save_path,'datasize_test/rf_bodytext_',i,'.rds')))
lapply(1:length(section_rf_res[[2]]), function(i) saveRDS(section_rf_res[[2]][[i]], paste0(save_path,'datasize_test/rf_section_',i,'.rds')))


# XGBoost
lapply(1:length(editorial_xgb_res[[2]]), function(i) xgb.save(editorial_xgb_res[[2]][[i]], paste0(save_path,'datasize_test/xgb_editorial_',i,'.model')))
lapply(1:length(bodytext_xgb_res[[2]]), function(i) xgb.save(bodytext_xgb_res[[2]][[i]], paste0(save_path,'datasize_test/xgb_bodytext_',i,'.model')))
lapply(1:length(section_xgb_res[[2]]), function(i) xgb.save(section_xgb_res[[2]][[i]], paste0(save_path,'datasize_test/xgb_section_',i,'.model')))




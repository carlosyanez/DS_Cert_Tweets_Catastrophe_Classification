#####################################################################################################
#####################################################################################################
# PH125.9x - Data Science : Capstone
# Own Project - Tweet Classifier - Catastrophe or not?
# C Yáñez Santibáñez
#####################################################################################################
#####################################################################################################
# This file contains code used to generate movie rating predictions from the movielens dataset.
#This file is structured in the following sections:
# 1. Code to load all require libraries
# 2. Functions
#### 2.1 Prepare Data
#### 2.2 Tokenise Data
#### 2.3 Calculate Scores
#### 2.4 Get Domains
#### 2.5 Score Tweets
#### 2.6 Fit Model
#### 2.7 Predict Values

#####################################################################################################
#####################################################################################################
# 1. Code to load all require libraries

if (!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require(caret)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require(tidytext)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require(scales)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require(textdata)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require(useful)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require(textclean)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require(lexicon)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require(stringi)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require(longurl)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require(xgboost)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require(onehot)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require(caTools)) install.packages("caTools", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(tidytext)
library(scales)
library(textdata)
library(useful)
library(textclean)
library(lexicon)
library(longurl)
library(xgboost)
library(onehot)
library(caTools)

#####################################################################################################
#####################################################################################################
# 2. Machine learning model and evaluation functions


#### 2.1 Prepare Data - convert data loaded from Internet and split into clean
#        text, hashtag, mention and link observations. Generates both normal and "sanitised" output
#        Inputs:
##             twitter_df : dataset with tweets - direct load of data obtained from the Internet
##             profanity_clean : 0 - Output without profanity removal, 1 - Remove profanity from output
prepare_data <- function(twitter_df, profanity_clean=0) {
  
  # extract hashtags
  
  twitter_df$hashtag <-str_extract_all(twitter_df$text, "#\\S+")
  twitter_df <-twitter_df %>%
    mutate(hashtag = gsub(x = hashtag, pattern = "character\\(0)", replacement = "")) %>%
    mutate(hashtag = gsub(x = hashtag, pattern = "c\\(", replacement = "")) %>%
    mutate(hashtag = gsub(x = hashtag, pattern = "\"", replacement = "")) %>%
    mutate(hashtag = gsub(x = hashtag, pattern = ")", replacement = "")) %>%
    mutate(hashtag = gsub(x = hashtag, pattern = ",", replacement = ""))

  
  #extract links
  
  twitter_df$link <-str_extract_all(twitter_df$text,
                                     "(http|ftp|https)://([\\w_-]+(?:(?:\\.[\\w_-]+)+))([\\w.,@?^=%&:/~+#-]*[\\w@?^=%&/~+#-])?")
  twitter_df <-twitter_df %>%
    mutate(link = gsub(x = link, pattern = "character\\(0)", replacement = "")) %>%
    mutate(link = gsub(x = link, pattern = "c\\(", replacement = "")) %>%
    mutate(link = gsub(x = link, pattern = "\"", replacement = "")) %>%
    mutate(link = gsub(x = link, pattern = ")", replacement = "")) %>%
    mutate(link = gsub(x = link, pattern = ",", replacement = ""))
  
  # extract mentions
  
  twitter_df$mention <-str_extract_all(twitter_df$text, "@\\S+")
  twitter_df <-twitter_df %>%
    mutate(mention = gsub(x = mention, pattern = "character\\(0)", replacement = "")) %>%
    mutate(mention = gsub(x = mention, pattern = "c\\(", replacement = "")) %>%
    mutate(mention = gsub(x = mention, pattern = "\"", replacement = "")) %>%
    mutate(mention = gsub(x = mention, pattern = ")", replacement = "")) %>%
    mutate(mention = gsub(x = mention, pattern = ",", replacement = ""))
  
  #remove mentions and links from text, save original text in different observation
  
  twitter_df$original_text <-twitter_df$text
  twitter_df <-twitter_df %>%
    mutate(text = gsub(x = text, pattern = "#", replacement = "")) %>%
    mutate(text = gsub(x = text, pattern = "@\\S+", replacement = "")) %>%
    mutate(text = gsub(x = text,
                       pattern = "(s?)(f|ht)tp(s?)://\\S+\\b",
                       replacement = ""))
  
  # #further clean text with textclean package
  
  twitter_df$text <-replace_contraction(twitter_df$text)
  twitter_df$text <-replace_incomplete(twitter_df$text)
  twitter_df$text <-replace_word_elongation(twitter_df$text)
  twitter_df$text <-add_comma_space(twitter_df$text)
  twitter_df$text <-replace_emoji(twitter_df$text)
  twitter_df$text <-replace_emoticon(twitter_df$text)
  twitter_df$text <-replace_non_ascii(twitter_df$text, impart.meaning = TRUE)
  twitter_df$text <-replace_internet_slang(twitter_df$text)
  twitter_df$text <-replace_html(twitter_df$text)
  twitter_df$text <-replace_money(twitter_df$text)
  twitter_df$text <-replace_number(twitter_df$text)
  twitter_df$text <-replace_ordinal(twitter_df$text)
  twitter_df$text <-replace_time(twitter_df$text)
  
  #profanity removal
  
  if (profanity_clean == 1) {
    
    #download profane words and prep for matching
    data(profanity_zac_anger)
    data(profanity_alvarez)
    data(profanity_arr_bad)
    data(profanity_banned)
    data(profanity_racist)
    Sys.sleep(100)
    
    special_chars <-as_tibble(c("\\!", "\\@", "\\#", "\\$", "\\&", "\\(", "\\)",
                                 "\\-", "\\‘", "\\.", "\\/", "\\+", '\\"', '\\“'))
    special_chars$replacement <-paste0("st", stri_rand_strings(nrow(special_chars), 3, '[a-zA-Z0-9]'))
    
    profanity <-as_tibble(c(profanity_zac_anger, profanity_alvarez,
                             profanity_arr_bad, profanity_banned,
                             profanity_racist))
    profanity <-unique(profanity)
    profanity$replacement <-paste0("pr", stri_rand_strings(nrow(profanity), 7, '[a-zA-Z0-9]'))
    
    for (i in 1: nrow(special_chars)) {
      profanity <-profanity %>%
        mutate(value = gsub(special_chars[i, ] $value, special_chars[i, ] $replacement,
                            value))
    }
    profanity <-profanity %>% mutate(value = paste0('\\b', value, '\\b'))
    
    rm(profanity_zac_anger, profanity_alvarez, profanity_arr_bad,
       profanity_banned, profanity_racist)
    
    #clean up profane words, replace by random string
    
    for (i in 1: nrow(special_chars)) {
      twitter_df <-twitter_df %>%
        mutate(text = gsub(special_chars[i, ] $value, special_chars[i, ] $replacement,
                           text))
    }
    
    for (i in 1: nrow(profanity)) {
      twitter_df <-twitter_df %>%
        mutate(text = gsub(profanity[i, ] $value, profanity[i, ] $replacement,
                           text, ignore.case = TRUE))
    }
    
    
    twitter_df <-twitter_df %>%
      mutate(text = gsub("st[a-zA-Z0-9][a-zA-Z0-9][a-zA-Z0-9]", "", text, ignore.case = TRUE))
    
    
    
    rm(profanity, special_chars)
  }
  
  
  twitter_df
  
}

#### 2.2 Tokenise Data - Take "prepared" data frame and convert into collection of words, hashtags, handle
#        and domain tokens for scoring
#        Inputs:
##             twitter_df : output of prepare_data
##             extra_stop_words : dataset with additional stop words (not contained in preloaded libraries)
##             domain_data : output of get_domains function
##             training_flag : TRUE - add true value in output (for training) - FALSE - don't add (e.g. if not present)
tokenise_data <- function(twitter_df, extra_stop_words, domain_data, training_flag = FALSE) {
  
  output <-vector(mode = "list", length = 0)
  
  
  data("stop_words")
  extra_stop_words$lexicon <-"EXTRA"
  stop_words <-rbind(stop_words, extra_stop_words)
  rm(extra_stop_words)
  
  
  output$tokenised_words <-twitter_df %>% unnest_tokens(word, text)
  output$tokenised_words <-output$tokenised_words %>%
    anti_join(stop_words, by = "word")
  
  output$tokenised_hashtags <-twitter_df %>% filter(!nchar(hashtag) == 0) %>% unnest_tokens(word, hashtag)
  output$tokenised_hashtags <-output$tokenised_hashtags %>%
    anti_join(stop_words, by = "word")
  
  output$tokenised_handles <-twitter_df %>% filter(!nchar(mention) == 0) %>% unnest_tokens(word, mention)
  output$tokenised_handles <-output$tokenised_handles %>%
    anti_join(stop_words, by = "word")
  
  domain_data$word <-domain_data$link
  
  output$tokenised_links <-twitter_df %>%
    filter(!nchar(link) == 0) %>%
    unnest_tokens(word, link, to_lower = FALSE) %>%
    filter(!(word == "t.co")) %>%
    filter(!(word == "http")) %>%
    filter(!(word == "https")) %>%
    mutate(word = paste("https://t.co/", word, sep = "")) %>%
    left_join(domain_data, by = "word") %>%
    filter(!is.na(domain))
  
  
  
  if (training_flag == TRUE) {
    output$tokenised_words <-output$tokenised_words %>% select(id, word, target)
    output$tokenised_hashtags <-output$tokenised_hashtags %>% select(id, word, target)
    output$tokenised_handles <-output$tokenised_handles %>% select(id, word, target)
    output$tokenised_links <-output$tokenised_links %>% select(id, word = domain, target)
    
  } else {
    output$tokenised_words <-output$tokenised_words %>% select(id, word)
    output$tokenised_hashtags <-output$tokenised_hashtags %>% select(id, word)
    output$tokenised_handles <-output$tokenised_handles %>% select(id, word)
    output$tokenised_links <-output$tokenised_links %>% select(id, word = domain)
    
  }
  
  output
  
}

#### 2.3 Calculate Scores - Take "prepared" data frame and convert into collection of words, hashtags, handle
#        and domain tokens for scoring
#        Inputs:
##           tokenised_training_data : output  of tokenise_data
##           word_parameters : array with variables for word scoring c(a, b, c, d, e) (refer to RMD formulas):
###                          a : exp/sin - expontential or sinus function
###                          b : lambda_2
###                          c : lamdba_1
###                          d : lambda_4
###                          e : lambda_3
##           hashtag_parameters : array with variables for hashtag scoring c(a, b, c, d) (refer to RMD formulas):
###                          a : exp/sin - expontential or sinus function
###                          b : lambda_2
###                          c : lamdba_1
###                          d : lambda_3
##           handle_parameters : array with variables for handle scoring c(a, b, c, d) (refer to RMD formulas):
###                          a : exp/sin - expontential or sinus function
###                          b : lambda_2
###                          c : lamdba_1
###                          d : lambda_3
##           link_parameters : array with variables for link scoring c(a, b, c, d) (refer to RMD formulas):
###                          a : exp/sin - expontential or sinus function
###                          b : lambda_2
###                          c : lamdba_1
###                          d : lambda_3
##           manual_scores : dataframe with manual scores for hashtags, handle, links

calculate_scores <- function(tokenised_training_data,
                              word_parameters = c("sin", 1, 0, 10, 0),
                              hashtag_parameters = c("sin", 1, 0, 5),
                              handle_parameters = c("sin", 1, 0, 5),
                              link_parameters = c("sin", 1, 0, 5), manual_scores = 1) {
  
  output <-vector(mode = "list", length = 0)
  
  #Words
  
  words_freq <-tokenised_training_data$tokenised_words %>% count(word, sort = TRUE)
  
  freq_pos_words <-tokenised_training_data$tokenised_words %>%
    filter(target == 1) %>% count(word, sort = TRUE) %>%
    mutate(nn = n / sum(n)) %>% mutate(nn = nn / max(nn)) %>%
    select(word, pos_n = n, pos_nn = nn)
  
  freq_neg_words <-tokenised_training_data$tokenised_words %>%
    filter(target == 0) %>% count(word, sort = TRUE) %>%
    mutate(nn = n / sum(n)) %>% mutate(nn = nn / max(nn)) %>%
    select(word, neg_n = n, neg_nn = nn)
  
  positive_words <-(freq_pos_words %>% select(word, pos_n, pos_nn)) %>%
    anti_join(freq_neg_words, by = "word") %>%
    mutate(neg_n = 0, neg_nn = 0)
  
  words <-(freq_neg_words %>% select(word, neg_n, neg_nn)) %>%
    anti_join(positive_words, by = "word") %>%
    left_join((freq_pos_words %>% select(word, pos_n, pos_nn)), by = "word")
  
  words <-rbind(words, positive_words)
  words[2: ncol(words)][is.na(words[2: ncol(words)])] <-0
  
  words <-words %>% mutate(pos_proportion = pos_n / (pos_n + neg_n), neg_proportion = neg_n / (pos_n + neg_n))
  words$theta_nn <-cart2pol(words$neg_nn, words$pos_nn, degrees = TRUE) $theta
  
  if (word_parameters[1] == "sin") {
    
    words <-words %>% mutate(score = sin(pos_proportion) ^ (as.numeric(word_parameters[2])) * theta_nn / 90)
    
  } else {
    
    words <-words %>% mutate(score = (pos_proportion / as.numeric(word_parameters[3])) ^ (as.numeric(word_parameters[2])) * theta_nn / 90)
    
  }
  words <-words %>% mutate(score = score / max(words$score) - as.numeric(word_parameters[5]))
  words <-words %>% left_join(words_freq, by = "word") %>%
    select(word, score, pos_proportion, theta_nn, n) %>%
    filter(n > word_parameters[4])
  
  output$words <-words
  rm(words_freq, freq_pos_words, freq_neg_words, positive_words, words)
  
  
  # hashtags
  
  hashtags_freq <-tokenised_training_data$tokenised_hashtags %>% count(word, sort = TRUE)
  
  freq_pos_hashtags <-tokenised_training_data$tokenised_hashtags %>%
    filter(target == 1) %>% count(word, sort = TRUE) %>%
    mutate(nn = n / sum(n)) %>% mutate(nn = nn / max(nn)) %>%
    select(word, pos_n = n, pos_nn = nn)
  
  freq_neg_hashtags <-tokenised_training_data$tokenised_hashtags %>%
    filter(target == 0) %>% count(word, sort = TRUE) %>%
    mutate(nn = n / sum(n)) %>% mutate(nn = nn / max(nn)) %>%
    select(word, neg_n = n, neg_nn = nn)
  
  positive_hashtags <-(freq_pos_hashtags %>% select(word, pos_n, pos_nn)) %>%
    anti_join(freq_neg_hashtags, by = "word") %>%
    mutate(neg_n = 0, neg_nn = 0)
  
  hashtags <-(freq_neg_hashtags %>% select(word, neg_n, neg_nn)) %>%
    anti_join(positive_hashtags, by = "word") %>%
    left_join((freq_pos_hashtags %>% select(word, pos_n, pos_nn)), by = "word")
  
  hashtags <-rbind(hashtags, positive_hashtags)
  hashtags[2: ncol(hashtags)][is.na(hashtags[2: ncol(hashtags)])] <-0
  
  hashtags <-hashtags %>% mutate(pos_proportion = pos_n / (pos_n + neg_n), neg_proportion = neg_n / (pos_n + neg_n))
  hashtags$theta_nn <-cart2pol(hashtags$neg_nn, hashtags$pos_nn, degrees = TRUE) $theta
  
  if (hashtag_parameters[1] == "sin") {
    
    hashtags <-hashtags %>% mutate(score = sin(pos_proportion) ^ (as.numeric(hashtag_parameters[2])) * theta_nn / 90)
    
  } else {
    
    hashtags <-hashtags %>%
      mutate(score = (pos_proportion / as.numeric(hashtag_parameters[3])) ^ (as.numeric(hashtag_parameters[2])) * theta_nn / 90)
    
  }
  
  hashtags <-hashtags %>% left_join(hashtags_freq, by = "word") %>%
    select(word, score, pos_proportion, n) %>%
    filter(n > hashtag_parameters[4])
  
  if (!(manual_scores == 1)) {
    
    if (nrow(hashtags) == 0) {
      pre_score <-5
    } else {
      pre_score <-max(hashtags$score)
    }
    
    m_scores <-manual_scores %>% filter(type == "hashtag") %>%
      mutate(score = value * pre_score, pos_proportion = 1, n = 0) %>%
      select(word = element, score, pos_proportion, n)
    
    hashtags <-hashtags %>% anti_join(m_scores, by = "word")
    hashtags <-rbind(hashtags, m_scores)
    
    
  }
  
  output$hashtags <-hashtags
  rm(hashtags_freq, freq_pos_hashtags, freq_neg_hashtags, positive_hashtags, hashtags)
  
  #Handles
  
  handles_freq <-tokenised_training_data$tokenised_handles %>% count(word, sort = TRUE)
  
  freq_pos_handles <-tokenised_training_data$tokenised_handles %>%
    filter(target == 1) %>% count(word, sort = TRUE) %>%
    mutate(nn = n / sum(n)) %>% mutate(nn = nn / max(nn)) %>%
    select(word, pos_n = n, pos_nn = nn)
  
  freq_neg_handles <-tokenised_training_data$tokenised_handles %>%
    filter(target == 0) %>% count(word, sort = TRUE) %>%
    mutate(nn = n / sum(n)) %>% mutate(nn = nn / max(nn)) %>%
    select(word, neg_n = n, neg_nn = nn)
  
  positive_handles <-(freq_pos_handles %>% select(word, pos_n, pos_nn)) %>%
    anti_join(freq_neg_handles, by = "word") %>%
    mutate(neg_n = 0, neg_nn = 0)
  
  handles <-(freq_neg_handles %>% select(word, neg_n, neg_nn)) %>%
    anti_join(positive_handles, by = "word") %>%
    left_join((freq_pos_handles %>% select(word, pos_n, pos_nn)), by = "word")
  
  handles <-rbind(handles, positive_handles)
  handles[2: ncol(handles)][is.na(handles[2: ncol(handles)])] <-0
  
  handles <-handles %>% mutate(pos_proportion = pos_n / (pos_n + neg_n), neg_proportion = neg_n / (pos_n + neg_n))
  handles$theta_nn <-cart2pol(handles$neg_nn, handles$pos_nn, degrees = TRUE) $theta
  
  if (handle_parameters[1] == "sin") {
    
    handles <-handles %>% mutate(score = sin(pos_proportion) ^ (as.numeric(handle_parameters[2])) * theta_nn / 90)
    
  } else {
    
    handles <-handles %>% mutate(score = (pos_proportion / as.numeric(handle_parameters[3])) ^ (as.numeric(handle_parameters[2])) * theta_nn / 90)
    
  }
  
  handles <-handles %>% left_join(handles_freq, by = "word") %>%
    select(word, score, pos_proportion, n) %>%
    filter(n > handle_parameters[4])
  
  output$handles <-handles
  rm(handles_freq, freq_pos_handles, freq_neg_handles, positive_handles, handles)
  
  # Domains
  
  links_freq <-tokenised_training_data$tokenised_links %>% count(word, sort = TRUE)
  
  freq_pos_links <-tokenised_training_data$tokenised_links %>%
    filter(target == 1) %>% count(word, sort = TRUE) %>%
    mutate(nn = n / sum(n)) %>% mutate(nn = nn / max(nn)) %>%
    select(word, pos_n = n, pos_nn = nn)
  
  freq_neg_links <-tokenised_training_data$tokenised_links %>%
    filter(target == 0) %>% count(word, sort = TRUE) %>%
    mutate(nn = n / sum(n)) %>% mutate(nn = nn / max(nn)) %>%
    select(word, neg_n = n, neg_nn = nn)
  
  positive_links <-(freq_pos_links %>% select(word, pos_n, pos_nn)) %>%
    anti_join(freq_neg_links, by = "word") %>%
    mutate(neg_n = 0, neg_nn = 0)
  
  links <-(freq_neg_links %>% select(word, neg_n, neg_nn)) %>%
    anti_join(positive_links, by = "word") %>%
    left_join((freq_pos_links %>% select(word, pos_n, pos_nn)), by = "word")
  
  links <-rbind(links, positive_links)
  links[2: ncol(links)][is.na(links[2: ncol(links)])] <-0
  
  links <-links %>% mutate(pos_proportion = pos_n / (pos_n + neg_n), neg_proportion = neg_n / (pos_n + neg_n))
  links$theta_nn <-cart2pol(links$neg_nn, links$pos_nn, degrees = TRUE) $theta
  
  if (link_parameters[1] == "sin") {
    
    links <-links %>% mutate(score = sin(pos_proportion) ^ (as.numeric(link_parameters[2])) * theta_nn / 90)
    
  } else {
    
    links <-links %>% mutate(score = (pos_proportion / as.numeric(link_parameters[3])) ^ (as.numeric(link_parameters[2])) * theta_nn / 90)
    
  }
  
  links <-links %>% left_join(links_freq, by = "word") %>%
    select(word, score, pos_proportion, n) %>%
    filter(n > link_parameters[4])
  
  output$links <-links
  rm(links_freq, freq_pos_links, freq_neg_links, positive_links, links)
  
  
  output
}

#### 2.4 Get Domains - Obtain actual "source" domain names for a list of t.co URLs
#Inputs
##             prepared_data : output of prepare_data
##             anonimised : 0 - output actual domain names, 1 - output sequential id
get_domains <- function(prepared_data, anonimised = 0) {
  
  #prepared_data <-processed_source
  # anonimised <-1
  
  tokenised_links <-prepared_data %>% filter(!nchar(link) == 0) %>%
    unnest_tokens(word, link, to_lower = FALSE)
  
  tokenised_links <-tokenised_links %>% filter(!(word == "t.co")) %>%
    filter(!(word == "http")) %>%
    filter(!(word == "https")) %>%
    mutate(word = paste("https://t.co/", word, sep = "")) %>%
    select(link = word)
  # tokenised_links <-as.tibble(tokenised_links[1: 20, ])
  # tokenised_links$link <-tokenised_links$value
  results <-expand_urls(tokenised_links$link)
  
  #remove NAs
  results <-results %>% filter(!is.na(expanded_url))
  
  #extract domains from results
  
  results$domain <-str_extract_all(results$expanded_url, ".*\\b(\\w+\\.\\w+)", "\\1")
  results <-results %>%
    mutate(domain = gsub(x = domain, pattern = "(http|ftp|https)://", replacement = "")) %>%
    mutate(domain = gsub(x = domain, pattern = "www.", replacement = ""))
  
  #if anonimised, collate domains and create replacements
  
  if (anonimised == 1) {
    domain_list <-results %>% select(domain) %>% unique(.)
    domain_list$domain_key <-paste0("do_", stri_rand_strings(nrow(domain_list), 9, '[a-zA-Z0-9]'))
    
    results <-results %>% left_join(domain_list, by = "domain")
    
  }
  
  results
  
  
}

#### 2.5 Score Tweets - Produce a score for each tweet, for each of the key attributes
#       Inputs
##             tweets_df : output of prepare_data
##             training_tokenised : 0 - output of tokenise_data
##             scores:  output of calculate_scores
score_tweets <- function(tweets_df, training_tokenised, scores) {
  
  output <-vector(mode = "list", length = 0)
  
  
  word_score <-training_tokenised$tokenised_words %>%
    left_join(scores$words, by = "word") %>% filter(!is.na(score)) %>%
    group_by(id) %>% summarise(word_score = sum(score)) %>%
    ungroup()
  
  hashtag_score <-training_tokenised$tokenised_hashtags %>%
    left_join(scores$hashtags, by = "word") %>% filter(!is.na(score)) %>%
    group_by(id) %>% summarise(hashtag_score = sum(score)) %>%
    ungroup()
  
  handle_score <-training_tokenised$tokenised_handle %>%
    left_join(scores$handles, by = "word") %>% filter(!is.na(score)) %>%
    group_by(id) %>% summarise(handle_score = sum(score)) %>%
    ungroup()
  
  link_score <-training_tokenised$tokenised_links %>%
    left_join(scores$links, by = "word") %>% filter(!is.na(score)) %>%
    group_by(id) %>% summarise(link_score = sum(score)) %>%
    ungroup()
  
  
  score_vector <-tweets_df %>% select(id, target) %>%
    left_join(word_score, by = "id") %>%
    left_join(hashtag_score, by = "id") %>%
    left_join(handle_score, by = "id") %>%
    left_join(link_score, by = "id")
  
  # score_vector[is.na(score_vector)] <- -1
  score_vector$target <-as.factor(score_vector$target)
  score_vector <-score_vector %>%
    mutate(type = paste(gsub(x = !is.na(word_score),
                             pattern = "TRUE", replacement = "word"),
                        gsub(x = !is.na(hashtag_score),
                             pattern = "TRUE", replacement = "hashtag"),
                        gsub(x = !is.na(handle_score),
                             pattern = "TRUE", replacement = "handle"),
                        gsub(x = !is.na(link_score),
                             pattern = "TRUE", replacement = "link"),
                        sep = ",")) %>%
    mutate(type = gsub(x = type,
                       pattern = "FALSE", replacement = "")) %>%
    mutate(type = gsub(x = type,
                       pattern = ",,,", replacement = ",")) %>%
    mutate(type = gsub(x = type,
                       pattern = ",,", replacement = ",")) %>%
    mutate(type = gsub(x = type,
                       pattern = ",$", replacement = "")) %>%
    mutate(type = gsub(x = type,
                       pattern = "^,", replacement = "")) %>%
    mutate(type2 = gsub(x = (nchar(type) == 0),
                        pattern = "TRUE", replacement = "none")) %>%
    mutate(type2 = gsub(x = type2,
                        pattern = "FALSE", replacement = "")) %>%
    mutate(type = paste(type, type2, sep = "")) %>% select(-type2)
  
  output$vector <-score_vector
  output$split <-score_vector %>% group_by(type) %>% summarise(n = n()) %>%
    ungroup() %>% mutate(n_perc = 100 * n / sum(n)) %>% arrange(-n)
  
  output$split$id <-seq.int(nrow(output$split))
  output
}

#### 2.6 Fit Model - Fit combined Machine Learning Model, using caret train function
#       Inputs
##             score_vector : output of score_tweets
##             cv_number : cross validation number for each use of caret's train
##             method_ABC:  machine learning method used for each attribute and for overall model. 
##                          Can be used with any method supported by caret, but additional libraries may be required.
##                          This file has been setup to used with glm, LogitBoost and xgbTree
##            family_ABC : parameter used in glm to choose regression family. Use "binomial" for Logistic Classification
##            tuneGrid : caret's tuneGrid parameter

fit_model <- function(score_vector, cv_number = 20,
                       method_word = "glm", family_word = "binomial", tuneGrid_word = 1,
                       method_hashtag = "glm", family_hashtag = "binomial", tuneGrid_hashtag = 1,
                       method_link = "glm", family_link = "binomial", tuneGrid_link = 1,
                       method_handle = "glm", family_handle = "binomial", tuneGrid_handle = 1,
                       method_aggregate = "xgbTree", family_aggregate = "binomial", tuneGrid_aggregate = 1) {
  

  fitting_model <-vector(mode = "list", length = 0)
  
  
  #word
  score_vector_i <-score_vector$vector %>% filter(!is.na(word_score)) %>% select(target,word_score)
  if (!nrow(score_vector_i) == 0) {
    word_exist <-1
    if (tuneGrid_word == 1) {
      if (method_word == "glm") {
        fitting_model$word <-train(target~word_score, data = score_vector_i,
                                    trControl = trainControl(method = "cv", number = cv_number),
                                    method = method_word, family = family_word)
      } else {
        fitting_model$word <-train(target~word_score, data = score_vector_i,
                                    trControl = trainControl(method = "cv", number = cv_number),
                                    method = method_word)
        
      }
    } else {
      if (method_word == "glm") {
        fitting_model$word <-train(target~word_score, data = score_vector_i,
                                    trControl = trainControl(method = "cv", number = cv_number),
                                    tuneGrid = tuneGrid_word,
                                    method = method_word, family = family_word)
      } else {
        fitting_model$word <-train(target~word_score, data = score_vector_i,
                                    trControl = trainControl(method = "cv", number = cv_number),
                                    tuneGrid = tuneGrid_word,
                                    method = method_word)
        
      }
    }
  }
  

  #hashtag
  # i <-i + 1
  score_vector_i <-score_vector$vector %>% filter(!is.na(hashtag_score))
  if (!(nrow(score_vector_i) == 0)) {
    hashtag_exist <-1
    if (tuneGrid_hashtag == 1) {
      if (method_hashtag == "glm") {
        fitting_model$hashtag <-train(target~hashtag_score, data = score_vector_i,
                                       trControl = trainControl(method = "cv", number = cv_number),
                                       method = method_hashtag, family = family_hashtag)
      } else {
        fitting_model$hashtag <-train(target~hashtag_score, data = score_vector_i,
                                       trControl = trainControl(method = "cv", number = cv_number),
                                       method = method_hashtag)
        
      }
    } else {
      if (method_hashtag == "glm") {
        fitting_model$hashtag <-train(target~hashtag_score, data = score_vector_i,
                                       trControl = trainControl(method = "cv", number = cv_number),
                                       tuneGrid = tuneGrid_hashtag,
                                       method = method_hashtag, family = family_hashtag)
      } else {
        fitting_model$hashtag <-train(target~hashtag_score, data = score_vector_i,
                                       trControl = trainControl(method = "cv", number = cv_number),
                                       tuneGrid = tuneGrid_hashtag,
                                       method = method_hashtag)
        
        
      }
      
    }
  }

  #handle
  score_vector_i <-score_vector$vector %>% filter(!is.na(handle_score))
  if (!nrow(score_vector_i) == 0) {
    handle_exist <-1
    if (tuneGrid_handle == 1) {
      if (method_handle == "glm") {
        fitting_model$handle <-train(target~handle_score, data = score_vector_i,
                                      trControl = trainControl(method = "cv", number = cv_number),
                                      method = method_handle, family = family_handle)
        
      } else {
        fitting_model$handle <-train(target~handle_score, data = score_vector_i,
                                      trControl = trainControl(method = "cv", number = cv_number),
                                      method = method_handle)
        
        
      }
    } else {
      if (method_handle == "glm") {
        fitting_model$handle <-train(target~handle_score, data = score_vector_i,
                                      trControl = trainControl(method = "cv", number = cv_number),
                                      tuneGrid = tuneGrid_handle,
                                      method = method_handle, family = family_handle)
      } else {
        fitting_model$handle <-train(target~handle_score, data = score_vector_i,
                                      trControl = trainControl(method = "cv", number = cv_number),
                                      tuneGrid = tuneGrid_handle,
                                      method = method_handle)
        
      }
      
    }
  }

  #link
  score_vector_i <-score_vector$vector %>% filter(!is.na(link_score))
  if (!nrow(score_vector_i) == 0) {
    link_exist <-1
    if (tuneGrid_link == 1) {
      if (method_link == "glm") {
        fitting_model$link <-train(target~link_score, data = score_vector_i,
                                    trControl = trainControl(method = "cv", number = cv_number),
                                    method = method_link, family = family_link)
      } else {
        fitting_model$link <-train(target~link_score, data = score_vector_i,
                                    trControl = trainControl(method = "cv", number = cv_number),
                                    method = method_link)
      }
    } else {
      if (method_link == "glm") {
        fitting_model$link <-train(target~link_score, data = score_vector_i,
                                    trControl = trainControl(method = "cv", number = cv_number),
                                    tuneGrid = tuneGrid_link,
                                    method = method_link, family = family_link)
      } else {
        fitting_model$link <-train(target~link_score, data = score_vector_i,
                                    trControl = trainControl(method = "cv", number = cv_number),
                                    tuneGrid = tuneGrid_link,
                                    method = method_link)
        
      }
      
    }
  }
  
  
  
  predictions <-tibble(id = integer(), type = character(), prediction = integer())
  
  #word model
  if (word_exist == 1) {
    term <-"word"
    
    model_i <-fitting_model$word
    scores_i <-score_vector$vector %>% filter(!is.na(word_score))
    # # # #
    scores_i_v <-scores_i %>% select(word_score)
    # # # # #
    scores_i$prediction <-predict(model_i, scores_i_v)
    scores_i <-scores_i %>% select(id, type, prediction) %>% mutate(type = term)
    predictions <-rbind(predictions, scores_i)
  }
  #hashtag model
  if (hashtag_exist == 1) {
    term <-"hashtag"
    
    model_i <-fitting_model$hashtag
    scores_i <-score_vector$vector %>% filter(!is.na(hashtag_score))

    scores_i_v <-scores_i %>% select(hashtag_score)

    scores_i$prediction <-predict(model_i, scores_i_v)
    scores_i <-scores_i %>% select(id, type, prediction) %>% mutate(type = term)
    predictions <-rbind(predictions, scores_i)
  }
  #handle model
  if (handle_exist == 1) {
    term <-"handle"
    
    model_i <-fitting_model$handle
    scores_i <-score_vector$vector %>% filter(!is.na(handle_score))
    # # # #
    scores_i_v <-scores_i %>% select(handle_score)
    # # # # #
    scores_i$prediction <-predict(model_i, scores_i_v)
    scores_i <-scores_i %>% select(id, type, prediction) %>% mutate(type = term)
    predictions <-rbind(predictions, scores_i)
  }
  #link model
  
  if (link_exist == 1) {
    term <-"link"
    
    model_i <-fitting_model$link
    scores_i <-score_vector$vector %>% filter(!is.na(link_score))
    # # # #
    scores_i_v <-scores_i %>% select(link_score)
    # # # # #
    scores_i$prediction <-predict(model_i, scores_i_v)
    scores_i <-scores_i %>% select(id, type, prediction) %>% mutate(type = term)
    predictions <-rbind(predictions, scores_i)
  }
  predictions <-predictions %>% spread(type, prediction)
  predictions_orig <-predictions
  
  p <-onehot(predictions)
  predictions <-predict(p, predictions)
  
  
  predictions <-score_vector$vector %>% left_join(as_tibble(predictions), by = "id") %>%
    select(-id, -type) %>% select(-contains("_score")) %>% filter(!is.na(`word=1`))
  
  if (tuneGrid_aggregate == 1) {
    if (method_aggregate == "glm") {
      fitting_model$aggregate <-train(target~., data = predictions,
                                       trControl = trainControl(method = "cv", number = cv_number),
                                       method = method_aggregate, family = family_aggregate)
    } else {
      fitting_model$aggregate <-train(target~., data = predictions,
                                       trControl = trainControl(method = "cv", number = cv_number),
                                       method = method_aggregate)
      
    }
  } else {
    if (method_aggregate == "glm") {
      fitting_model$aggregate <-train(target~., data = predictions,
                                       trControl = trainControl(method = "cv", number = cv_number),
                                       tuneGrid = tuneGrid_aggregate,
                                       method = method_aggregate, family = family_aggregate)
    } else {
      fitting_model$aggregate <-train(target~., data = predictions,
                                       trControl = trainControl(method = "cv", number = cv_number),
                                       tuneGrid = tuneGrid_aggregate,
                                       method = method_aggregate)
    }
  }
  
  fitting_model
}

#### 2.7 Predict Values - Used machine learning model to predict valies
#       Inputs
##             data : pre-processed by prepare_data
##             model : output of fit_model
##             scores : output of score_tweets (with training data)
##             extra_stop_words : extra stop words, in addition of pre-loaded libraries
##             domain_data : output of get_domains
predict_values <- function(data, model, scores, extra_stop_words, domain_data) {
  
  data <-test
  model <-fitting_model
  
  
  tokenised_test <-tokenise_data(test, extra_stop_words, domains_data)
  scored_tweets_test <-score_tweets(test, tokenised_test, scores)
  
  score_vector <-scored_tweets_test
  
  predictions <-tibble(id = integer(), type = character(), prediction = integer())
  
  #word model
  term <-"word"
  
  model_i <-fitting_model$word
  scores_i <-score_vector$vector %>% filter(!is.na(word_score))
  # # # #
  scores_i_v <-scores_i %>% select(word_score)
  # # # # #
  scores_i$prediction <-predict(model_i, scores_i_v)
  scores_i <-scores_i %>% select(id, type, prediction) %>% mutate(type = term)
  predictions <-rbind(predictions, scores_i)
  
  #hashtag model
  term <-"hashtag"
  
  model_i <-fitting_model$hashtag
  scores_i <-score_vector$vector %>% filter(!is.na(hashtag_score))
  # # # #
  scores_i_v <-scores_i %>% select(hashtag_score)
  # # # # #
  scores_i$prediction <-predict(model_i, scores_i_v)
  scores_i <-scores_i %>% select(id, type, prediction) %>% mutate(type = term)
  predictions <-rbind(predictions, scores_i)
  
  #handle model
  term <-"handle"
  
  model_i <-fitting_model$handle
  scores_i <-score_vector$vector %>% filter(!is.na(handle_score))
  # # # #
  scores_i_v <-scores_i %>% select(handle_score)
  # # # # #
  scores_i$prediction <-predict(model_i, scores_i_v)
  scores_i <-scores_i %>% select(id, type, prediction) %>% mutate(type = term)
  predictions <-rbind(predictions, scores_i)
  
  #link model
  term <-"link"
  
  model_i <-fitting_model$link
  scores_i <-score_vector$vector %>% filter(!is.na(link_score))
  # # # #
  scores_i_v <-scores_i %>% select(link_score)
  # # # # #
  scores_i$prediction <-predict(model_i, scores_i_v)
  scores_i <-scores_i %>% select(id, type, prediction) %>% mutate(type = term)
  predictions <-rbind(predictions, scores_i)
  
  predictions <-predictions %>% spread(type, prediction)
  predictions_orig <-predictions
  p <-onehot(predictions)
  predictions <-predict(p, predictions)
  
  predictions <-as_tibble(predictions) %>% select(-id)
  
  predictions_orig$aggregate <-predict(fitting_model$aggregate, predictions)
  
  remnant <-data %>% select(id) %>% filter(!(id %in% predictions_orig$id))
  
  remnant$aggregate <-0
  remnant$handle <- -1
  remnant$hashtag <- -1
  remnant$link <- -1
  remnant$word <- -1
  predictions_orig <-rbind(predictions_orig, remnant)
  
  
  
  
  results1 <-predictions_orig %>% left_join(score_vector$vector, by = "id")
  
  
  
  results_eval <-tibble(eval = character(), result_method = numeric(), result_agg = numeric())
  
  results_eval <-add_row(results_eval, eval = "word",
                          result_method = mean(results1 %>% filter(!is.na(word)) %>%
                                                 mutate(hit = (target == word)) %>%
                                                 pull(hit)),
                          result_agg = mean(results1 %>% filter(!is.na(word)) %>%
                                              mutate(hit = (target == aggregate)) %>% pull(hit)))
  results_eval <-add_row(results_eval, eval = "hashtag",
                          result_method = mean(results1 %>% filter(!is.na(hashtag)) %>%
                                                 mutate(hit = (target == hashtag)) %>% pull(hit)),
                          result_agg = mean(results1 %>% filter(!is.na(hashtag)) %>%
                                              mutate(hit = (target == aggregate)) %>% pull(hit)))
  results_eval <-add_row(results_eval, eval = "link",
                          result_method = mean(results1 %>% filter(!is.na(link)) %>%
                                                 mutate(hit = mean(target == link)) %>% pull(hit)),
                          result_agg = mean(results1 %>% filter(!is.na(link)) %>%
                                              mutate(hit = (target == aggregate)) %>% pull(hit)))
  
  results_eval <-add_row(results_eval, eval = "handle",
                          result_method = mean(results1 %>% filter(!is.na(handle)) %>%
                                                 mutate(hit = (target == handle)) %>% pull(hit)),
                          result_agg = mean(results1 %>% filter(!is.na(handle)) %>%
                                              mutate(hit = (target == aggregate)) %>% pull(hit)))
  
  
  
  results_eval <-add_row(results_eval, eval = "aggregate",
                          result_method = mean(results1 %>%
                                                 mutate(hit = (target == aggregate)) %>% pull(hit)),
                          result_agg = mean(results1 %>%
                                              mutate(hit = (target == aggregate)) %>% pull(hit)))
  
  
  output <-vector(mode = "list", length = 0)
  output$results <-results1
  output$results_extended <-predictions_orig %>% left_join(test, by = "id") %>%
    left_join(scored_tweets_test$vector, by = "id") %>%
    mutate(hit = ifelse(aggregate == target.x, "T", "F")) %>%
    mutate(category = ifelse(hit == "F",
                             ifelse(target.x == 1, "FN", "FP"),
                             ifelse(target.x == 1, "TP", "TN"))) %>% select(-hit) %>%
    select(id, category, target = target.x, aggregate, word_pred = word, hashtag_pred = hashtag.x,
           handle_pred = handle, link_pred = link.x,
           text, hashtag = hashtag.y, mention, link = link.y,
           word_score, hashtag_score, handle_score, link_score,
           type, category, original_text, location, keyword)
  
  output$eval <-results_eval
  
  output
  
}
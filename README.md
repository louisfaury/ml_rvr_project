# AML_RVR-SVR_project

## Running the project 
The MATLAB project is decomposed in 9 sections. Simply comment / uncomment them regarding on which you desired to run. 
### S1 : Loads datasets 
Change the name of the dataset (*sinc* or *airfoils*) regarding which you desire to load.  
### S2 : Run SVR
Runs the SVR with the kernel, the SVR method you choose (nu,epsilon SVR), the hyperparameters indicated.
### S3 : Run RVR
Runs the RVR with the indicated kernel width. 
### S4 : BICSR validation 
Runs cross-validation on several models (arbitrary model as well as the *optimal models* according to different metrics). Plots the result on a MSE/sparsity graph. 
### S5 : CV for nu-SVR
Runs f-fold cross-validation for the nu-SVR (with RBF Gaussian kernel), with grid search over the hyperparameters. 
### S6 : CV for eps-SVR
Runs f-fold cross-validation for the eps-SVR (with RBF Gaussian kernel), with grid search over the hyperparameters.
### S7 : CV for RVR
Runs f-fold cross-validation for the RVR (with RBF Gaussian kernel), with grid search over the hyperparameters.
### S8 : BICSR validation (2)
Basically performs the same as S4, without different penalizing terms (stays with klnN)
### S9 : Model comparison 
Plots the different optimal models (according to BICSR and MSE) on the MSE / sparsity graph. 

## Library gestion
* Put libsvm source code in libsvm/
* Put sparseBayes source code in sparseBayes/ 

## Online popularity dataset
 [link to dataset] (http://archive.ics.uci.edu/ml/datasets/Online+News+Popularity)
 
### Ouput

* Number of shares

### List of features :

1. n_tokens_title: Number of words in the title 
1. n_tokens_content: Number of words in the content 
1. n_unique_tokens: Rate of unique words in the content 
1. n_non_stop_words: Rate of non-stop words in the content 
1. n_non_stop_unique_tokens: Rate of unique non-stop words in the content 
1. num_hrefs: Number of links 
1. num_self_hrefs: Number of links to other articles published by Mashable 
1. num_imgs: Number of images 
1. num_videos: Number of videos 
1. average_token_length: Average length of the words in the content 
1. num_keywords: Number of keywords in the metadata 
1. data_channel_is_lifestyle: Is data channel 'Lifestyle'? 
1. data_channel_is_entertainment: Is data channel 'Entertainment'? 
1. data_channel_is_bus: Is data channel 'Business'? 
1. data_channel_is_socmed: Is data channel 'Social Media'? 
1. data_channel_is_tech: Is data channel 'Tech'? 
1. data_channel_is_world: Is data channel 'World'? 
1. kw_min_min: Worst keyword (min. shares) 
1. kw_max_min: Worst keyword (max. shares) 
1. kw_avg_min: Worst keyword (avg. shares) 
1. kw_min_max: Best keyword (min. shares) 
1. kw_max_max: Best keyword (max. shares) 
1. kw_avg_max: Best keyword (avg. shares) 
1. kw_min_avg: Avg. keyword (min. shares) 
1. kw_max_avg: Avg. keyword (max. shares) 
1. kw_avg_avg: Avg. keyword (avg. shares) 
1. self_reference_min_shares: Min. shares of referenced articles in Mashable 
1. self_reference_max_shares: Max. shares of referenced articles in Mashable 
1. self_reference_avg_sharess: Avg. shares of referenced articles in Mashable 
1. weekday_is_monday: Was the article published on a Monday? 
1. weekday_is_tuesday: Was the article published on a Tuesday? 
1. weekday_is_wednesday: Was the article published on a Wednesday? 
1. weekday_is_thursday: Was the article published on a Thursday? 
1. weekday_is_friday: Was the article published on a Friday? 
1. weekday_is_saturday: Was the article published on a Saturday? 
1. weekday_is_sunday: Was the article published on a Sunday? 
1. is_weekend: Was the article published on the weekend? 
1. LDA_00: Closeness to LDA topic 0 
1. LDA_01: Closeness to LDA topic 1 
1. LDA_02: Closeness to LDA topic 2 
1. LDA_03: Closeness to LDA topic 3 
1. LDA_04: Closeness to LDA topic 4 
1. global_subjectivity: Text subjectivity 
1. global_sentiment_polarity: Text sentiment polarity 
1. global_rate_positive_words: Rate of positive words in the content 
1. global_rate_negative_words: Rate of negative words in the content 
1. rate_positive_words: Rate of positive words among non-neutral tokens 
1. rate_negative_words: Rate of negative words among non-neutral tokens 
1. avg_positive_polarity: Avg. polarity of positive words 
1. min_positive_polarity: Min. polarity of positive words 
1. max_positive_polarity: Max. polarity of positive words 
1. avg_negative_polarity: Avg. polarity of negative words 
1. min_negative_polarity: Min. polarity of negative words 
1. max_negative_polarity: Max. polarity of negative words 
1. title_subjectivity: Title subjectivity 
1. title_sentiment_polarity: Title polarity 
1. abs_title_subjectivity: Absolute subjectivity level 
1. abs_title_sentiment_polarity: Absolute polarity level 

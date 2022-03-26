# +
"""
4 - run models.py

Build and test multiple machine learning models and optimize for performance

"""

## Managining dependencies ##
import numpy as np
import pandas as pd

## Model selection ##
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


## Import in complete dataset and convert to numpy array this excludes ANY drugs with Nans ##
drug_df = pd.read_pickle('master_data_with_sentiment.pk1')
drug_df_1_year = pd.read_pickle('master_data_with_sentiment_1_year.pk1')
drug_df_3_year = pd.read_pickle('master_data_with_sentiment_3_year.pk1')
drug_df_5_year = pd.read_pickle('master_data_with_sentiment_5_year.pk1')


# drop any NA rows
drug_df = drug_df.dropna()

# get array of word frequencies for each year
word_frequencies_year_0 = drug_df.iloc[:,12:9613].to_numpy()
word_frequencies_year_1 = drug_df_1_year.iloc[:,12:8340].to_numpy()
word_frequencies_year_3 = drug_df_3_year.iloc[:,12:6329].to_numpy()
word_frequencies_year_5 = drug_df_5_year.iloc[:,12:4686].to_numpy()

# get array of sentiment scores for each year
sentiment_scores_year_0 = drug_df.iloc[:,[4,6,8,10]].to_numpy()
sentiment_scores_year_1 = drug_df_1_year.iloc[:,[4,6,8,10]].to_numpy()
sentiment_scores_year_3 = drug_df_3_year.iloc[:,[4,6,8,10]].to_numpy()
sentiment_scores_year_5 = drug_df_5_year.iloc[:,[4,6,8,10]].to_numpy()

# get array of other drug features
drug_features = drug_df.iloc[:,9614:9638].to_numpy()
# get vector of labels
drug_labels = drug_df.iloc[:,11].to_numpy()

# create feature sets
fs1  = np.concatenate((drug_features, sentiment_scores_year_0, word_frequencies_year_0), axis=1)
fs2  = np.concatenate((drug_features, sentiment_scores_year_1, word_frequencies_year_1), axis=1)
fs3  = np.concatenate((drug_features, sentiment_scores_year_3, word_frequencies_year_3), axis=1)
fs4  = np.concatenate((drug_features, sentiment_scores_year_5, word_frequencies_year_5), axis=1)
fs5  = drug_features
fs6  = np.concatenate((drug_features,sentiment_scores_year_0), axis=1)
fs7  = np.concatenate((drug_features,sentiment_scores_year_1), axis=1)
fs8  = np.concatenate((drug_features,sentiment_scores_year_3), axis=1)
fs9  = np.concatenate((drug_features,sentiment_scores_year_5), axis=1)
fs10 = word_frequencies_year_0
fs11 = word_frequencies_year_1
fs12 = word_frequencies_year_3
fs13 = word_frequencies_year_5

feature_sets = [fs1,fs2,fs3,fs4,fs5,fs6,fs7,fs8,fs9,fs10,fs11,fs12,fs13]
fs_columns = ['fs1','fs2','fs3','fs4','fs5','fs6','fs7','fs8','fs9','fs10','fs11','fs12','fs13']

##############################################################################
# Analysis Functions
##############################################################################

# metrics on which to score models
# available scores: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
metrics = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']


# Function to calculate the k-fold cross validation scores for all metrics
#   - loops through various metrics
#   - calculates the score from k-fold cross validation
#   - returns the mean and standard deviation of scores for each metric

def cross_validation_scores(drug_X, drug_y, model, k_fold):
    
    # lists to score scores
    scores_mean = []
    scores_std = []
    
    # loop though all scoring metrics
    for metric in ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']:
        
        # get k-fold cross validation scores for given metric
        scores = cross_val_score(model, drug_X, drug_y, cv=k_fold, scoring=metric)
        
        # calculate mean and SD for each fold
        scores_mean.append(np.mean(scores))
        scores_std.append(np.std(scores))
        
    return scores_mean, scores_std


# Function to do k-fold cross validation with hyperparameter grid search
#   - creates grid search object using sklearn's GridSearchCV
#   - fits to the given data and performs grid search over parameters
#   - returns the mean and std score for various metrics with optimal parameters

def grid_search_cv_scores(drug_X, drug_y, model, parameter_grid, k_fold=5):

    # create grid search model object
    gso = GridSearchCV(estimator=model,
                       param_grid=parameter_grid,
                       scoring=['accuracy', 'f1', 'precision', 'recall', 'roc_auc'],
                       refit = 'accuracy',  # optimize for accuracy metric
                       n_jobs=-1,           # use all CPU cores
                       cv=k_fold)           # 5-fold CV
    
    # fit grid search object with data
    gso.fit(drug_X, drug_y)
    
    # get mean and std of cross validation results for given scoring
    # gso object contains CV results from all combinations of parameters in grid search (gso.cv_results)
    # gso.best_index_ returns the index of the results for optimal parameter values
    
    scores_mean = [gso.cv_results_['mean_test_accuracy'][gso.best_index_],
                   gso.cv_results_['mean_test_f1'][gso.best_index_],
                   gso.cv_results_['mean_test_precision'][gso.best_index_],
                   gso.cv_results_['mean_test_recall'][gso.best_index_],
                   gso.cv_results_['mean_test_roc_auc'][gso.best_index_]]
    
    scores_std = [gso.cv_results_['std_test_accuracy'][gso.best_index_],
                  gso.cv_results_['std_test_f1'][gso.best_index_],
                  gso.cv_results_['std_test_precision'][gso.best_index_],
                  gso.cv_results_['std_test_recall'][gso.best_index_],
                  gso.cv_results_['std_test_roc_auc'][gso.best_index_]]
    
    # list of optimal parameters
    optimal_parameters = gso.best_params_
    
    return scores_mean, scores_std, optimal_parameters


##############################################################################
# 1. Gaussian Naive Bayes
##############################################################################

# https://scikit-learn.org/stable/modules/naive_bayes.html
from sklearn.naive_bayes import GaussianNB

# model object
gnb = GaussianNB()

# sample run
# fit model 
gnb.fit(fs1, drug_labels)
# predict labels
gnb.predict(fs1)
# get scores from k-fold CV
gnb_scores_means, gnb_scores_sd = cross_validation_scores(fs1, drug_labels, gnb, 5)


# test model on all feature sets ---------------------------------------------

list_of_means = []
list_of_sds = []

# loop though all feature sets
for feature_set in feature_sets:
    
    # user defined k-fold cross validation function
    score_means, score_sds = cross_validation_scores(feature_set, drug_labels, gnb, 5)
   
    list_of_means.append(score_means)
    list_of_sds.append(score_sds)

# build table of mean scores for all metrics and all feature sets
gnb_mean_df = pd.DataFrame(list_of_means).T
gnb_mean_df.index = metrics
gnb_mean_df.columns = fs_columns

# build table of score standard deviations for all metrics and all feature sets
gnb_sd_df = pd.DataFrame(list_of_sds).T
gnb_sd_df.index = metrics
gnb_sd_df.columns = fs_columns
    


##############################################################################
# 2. Multinomial Naive Bayes
##############################################################################

# https://scikit-learn.org/stable/modules/naive_bayes.html
from sklearn.naive_bayes import MultinomialNB

# model object
mnb = MultinomialNB()

# sample run
# fit model 
mnb.fit(fs1, drug_labels)
# predict labels
mnb.predict(fs1)
# get scores from k-fold CV
mnb_scores_means, mnb_scores_sd = cross_validation_scores(fs1, drug_labels, mnb, 5)


# test model on all feature sets  --------------------------------------------

list_of_means = []
list_of_sds = []

# loop though all feature sets
for feature_set in feature_sets:
    
    # user defined k-fold cross validation function
    score_means, score_sds = cross_validation_scores(feature_set, drug_labels, mnb, 5)
   
    list_of_means.append(score_means)
    list_of_sds.append(score_sds)

# build table of mean scores for all metrics and all feature sets
mnb_mean_df = pd.DataFrame(list_of_means).T
mnb_mean_df.index = metrics
mnb_mean_df.columns = fs_columns

# build table of score standard deviations for all metrics and all feature sets
mnb_sd_df = pd.DataFrame(list_of_sds).T
mnb_sd_df.index = metrics
mnb_sd_df.columns = fs_columns



##############################################################################
# 3. Support Vector Machines
##############################################################################

# https://scikit-learn.org/stable/modules/svm.html
from sklearn.svm import SVC

# model object
svc = SVC()

# sample run
svc_sr = SVC(kernel='linear', C=0.1, gamma=1)
# fit model 
svc_sr.fit(fs1, drug_labels)
# predict labels
svc_sr.predict(fs1)
# get scores from k-fold CV and grid search
svc_scores_means, svc_scores_sd, svc_parameters = grid_search_cv_scores(fs1, drug_labels, SVC(), parameters, k_fold=5)

# set grid of parameters
parameters = {'kernel':('linear', 'rbf'),
              'C':[0.001,0.01,0.1,1,10,1000,10000],
              'gamma':[0.001,0.01,0.1,1,10,1000,10000]}


# test model on all feature sets ---------------------------------------------

list_of_means = []
list_of_sds = []
list_of_params = []

# loop though all feature sets
for feature_set in feature_sets:
    
    # user defined k-fold cross validation and grid search function
    score_means, score_sds, best_params = grid_search_cv_scores(feature_set, drug_labels, SVC(), parameters, k_fold=5)
   
    list_of_means.append(score_means)
    list_of_sds.append(score_sds)
    list_of_params.append(best_params)

# build table of mean scores for all metrics and all feature sets
svc_mean_df = pd.DataFrame(list_of_means).T
svc_mean_df.index = metrics
svc_mean_df.columns = fs_columns

# build table of score standard deviations for all metrics and all feature sets
svc_sd_df = pd.DataFrame(list_of_sds).T
svc_sd_df.index = metrics
svc_sd_df.columns = fs_columns

# build table of best parameters
svc_param_df = pd.DataFrame(list_of_params).T
svc_param_df.index = parameters.keys()
svc_param_df.columns = fs_columns


##############################################################################
# 4. Linear Discriminant Analysis
##############################################################################

# https://scikit-learn.org/stable/modules/lda_qda.html#lda-qda
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# model object
lda = LDA()

# sample run
# fit model
lda.fit(fs1, drug_labels)
# predict labels
lda.predict(fs1)
# get scores from k-fold CV
lda_scores_means, lda_scores_sd = cross_validation_scores(fs1, drug_labels, lda, 5)


# test model on all feature sets ---------------------------------------------

list_of_means = []
list_of_sds = []

# loop though all feature sets
for feature_set in feature_sets:
    
    # user defined k-fold cross validation function
    score_means, score_sds = cross_validation_scores(feature_set, drug_labels, LDA(), 5)
   
    list_of_means.append(score_means)
    list_of_sds.append(score_sds)

# build table of mean scores for all metrics and all feature sets
lda_mean_df = pd.DataFrame(list_of_means).T
lda_mean_df.index = metrics
lda_mean_df.columns = fs_columns

# build table of score standard deviations for all metrics and all feature sets
lda_sd_df = pd.DataFrame(list_of_sds).T
lda_sd_df.index = metrics
lda_sd_df.columns = fs_columns



##############################################################################
# 5. Quadratic Discriminant Analysis
##############################################################################

# https://scikit-learn.org/stable/modules/lda_qda.html#lda-qda
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

# model object
qda = QDA()

# sample run
# fit model
qda.fit(fs1, drug_labels)
# predict labels
qda.predict(fs1)
# get scores from k-fold CV
qda_scores_means, qda_scores_sd = cross_validation_scores(fs1, drug_labels, qda, 5)


# test model on all feature sets ---------------------------------------------

list_of_means = []
list_of_sds = []

# loop though all feature sets
for feature_set in feature_sets:
    
    # user defined k-fold cross validation function
    score_means, score_sds = cross_validation_scores(feature_set, drug_labels, QDA(), 5)
   
    list_of_means.append(score_means)
    list_of_sds.append(score_sds)

# build table of mean scores for all metrics and all feature sets
qda_mean_df = pd.DataFrame(list_of_means).T
qda_mean_df.index = metrics
qda_mean_df.columns = fs_columns

# build table of score standard deviations for all metrics and all feature sets
qda_sd_df = pd.DataFrame(list_of_sds).T
qda_sd_df.index = metrics
qda_sd_df.columns = fs_columns


##############################################################################
# 6. K-nearest Neighbors
##############################################################################

# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
from sklearn.neighbors import KNeighborsClassifier as KNC

# model object
knn = KNC()

# sample run
knn = KNC(n_neighbors=1)
# fit model
knn.fit(fs1, drug_labels)
# predict labels
knn.predict(fs1)
# get scores from k-fold CV and grid search
knn_scores_means, knn_scores_sd, knn_parameters = grid_search_cv_scores(fs1, drug_labels, KNC(), parameters, k_fold=5)

# set grid of parameters
parameters = {'n_neighbors':[n for n in range(1,101)]}
# test n_neighbors from 1 to 100. 


# test model on all feature sets ---------------------------------------------

list_of_means = []
list_of_sds = []
list_of_params = []

# loop though all feature sets
for feature_set in feature_sets:
    
    # user defined k-fold cross validation and grid search function
    score_means, score_sds, best_params = grid_search_cv_scores(feature_set, drug_labels, KNC(), parameters, k_fold=5)
   
    list_of_means.append(score_means)
    list_of_sds.append(score_sds)
    list_of_params.append(best_params)

# build table of mean scores for all metrics and all feature sets
knn_mean_df = pd.DataFrame(list_of_means).T
knn_mean_df.index = metrics
knn_mean_df.columns = fs_columns

# build table of score standard deviations for all metrics and all feature sets
knn_sd_df = pd.DataFrame(list_of_sds).T
knn_sd_df.index = metrics
knn_sd_df.columns = fs_columns

# build table of best parameters
knn_param_df = pd.DataFrame(list_of_params).T
knn_param_df.index = parameters.keys()
knn_param_df.columns = fs_columns


##############################################################################
# 7. Logistic Regression
##############################################################################

# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
from sklearn.linear_model import LogisticRegression

# model object
lr = LogisticRegression()

# sample run
# fit model
lr.fit(fs1, drug_labels)
# predict labels
lr.predict(fs1)
# get scores from k-fold CV
lr_scores_means, lr_scores_sd = cross_validation_scores(fs1, drug_labels, lr, 5)


# test model on all feature sets ---------------------------------------------

list_of_means = []
list_of_sds = []

# loop though all feature sets
for feature_set in feature_sets:
    
    # user defined k-fold cross validation function
    score_means, score_sds = cross_validation_scores(feature_set, drug_labels, LogisticRegression(), 5)
   
    list_of_means.append(score_means)
    list_of_sds.append(score_sds)

# build table of mean scores for all metrics and all feature sets
lr_mean_df = pd.DataFrame(list_of_means).T
lr_mean_df.index = metrics
lr_mean_df.columns = fs_columns

# build table of score standard deviations for all metrics and all feature sets
lr_sd_df = pd.DataFrame(list_of_sds).T
lr_sd_df.index = metrics
lr_sd_df.columns = fs_columns



##############################################################################
# 8. Random Forest
##############################################################################

from sklearn.ensemble import RandomForestClassifier

# model object
rf = RandomForestClassifier()

# sample run
rf = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=10, max_features=None)
# fit model
rf.fit(fs1, drug_labels)
# predict labels
rf.predict(fs1)
# get scores from k-fold CV and grid search
rf_scores_means, rf_scores_sd, rf_parameters = grid_search_cv_scores(fs1, drug_labels, rf, parameters, k_fold=5)

# set grid of parameters
parameters = {'n_estimators':[10,50,100,300,500],
              'criterion':('entropy', 'gini'),
              'max_depth':[1,2,3,4,5,7,10],
              'max_features':('sqrt',None)}


# test model on all feature sets ---------------------------------------------

list_of_means = []
list_of_sds = []
list_of_params = []

# loop though all feature sets
for feature_set in feature_sets:
    
    # user defined k-fold cross validation and grid search function
    score_means, score_sds, best_params = grid_search_cv_scores(feature_set, drug_labels, RandomForestClassifier(), parameters, k_fold=5)
   
    list_of_means.append(score_means)
    list_of_sds.append(score_sds)
    list_of_params.append(best_params)

# build table of mean scores for all metrics and all feature sets
rf_mean_df = pd.DataFrame(list_of_means).T
rf_mean_df.index = metrics
rf_mean_df.columns = fs_columns

# build table of score standard deviations for all metrics and all feature sets
rf_sd_df = pd.DataFrame(list_of_sds).T
rf_sd_df.index = metrics
rf_sd_df.columns = fs_columns

# build table of best parameters
rf_param_df = pd.DataFrame(list_of_params).T
rf_param_df.index = parameters.keys()
rf_param_df.columns = fs_columns


#!/usr/bin/env python
# coding: utf-8

# # COMP9417 20T2  Homework 1: Applying Machine Learning

# _Last revision: Wed 17 Jun 2020 23:07:10 AEST_
# 
# _Python version: Python 3.7.3_

# The aim of this homework is to enable you to **apply** different classifier learning algorithms implemented in the Python [scikit-learn](http://scikit-learn.org/stable/index.html) machine learning library on a variety of datasets and answer questions based on your **analysis** and **interpretation** of the empirical results, using your knowledge of machine learning.
# 
# After completing this homework you will be able to:
# 
# - set up replicated $k$-fold cross-validation experiments to obtain average
#   performance measures of algorithms on datasets
# - compare the performance of different algorithms against a base-line
#   and each other
# - aggregate comparative performance scores for algorithms over a range
#   of different datasets
# - propose properties of algorithms and their parameters, or datasets, which
#   may lead to performance differences being observed
# - suggest reasons for actual observed performance differences in terms of
#   properties of algorithms, parameter settings or datasets.
# - apply methods for data transformations and parameter search
#   and evaluate their effects on the performance of algorithms
# 
# There are a total of *10 marks* available.
# Each homework mark is worth *0.5 course mark*, i.e., homework marks will be scaled
# to a **course mark out of 5** to contribute to the course total.
# 
# #### Deadline: 10:59:59, Monday June 29, 2020.
# 
# Submission will be via the CSE *give* system (see below).
# 
# Late penalties: one mark will be deducted from the total for each day late, up to a total of five days. If six or more days late, no marks will be given.
# 
# Recall the guidance regarding plagiarism in the course introduction: this applies to this homework and if evidence of plagiarism is detected it may result in penalties ranging from loss of marks to suspension.
# 
# ### Format of the questions
# 
# There are 2 questions in this homework. Each question has two parts: the Python code which must be run to generate the output results on the given datasets, and the responses you give in the file [*answers.txt*](http://www.cse.unsw.edu.au/~cs9417/20T2/hw1/answers.txt) on your analysis and interpretation of the results  produced by running these learning algorithms for the question. Marks are given for both parts: submitting correct output from the code, and giving correct responses. For each question, you will need to save the output results from running the code to a separate plain text file. There will also be a plain text file containing the questions which you will need to edit to specify your answers. These files will form your submission.
# 
# In summary, your submission will comprise a total of 3 (three) files which should be named as follows:
# ```
# q1.out
# q2.out
# answers.txt
# ```
# Please note: files in any format other than plain text **cannot be accepted**.
# 
# Submit your files using ```give```. On a CSE Linux machine, type the following on the command-line:
# ```
# $ give cs9417 hw1 q1.out q2.out answers.txt
# ```
# 
# Alternatively, you can submit using the web-based interface to ```give```.
# 
# ### Datasets
# 
# You can download the datasets required for the homework as the file [*datasets.zip*](http://www.cse.unsw.edu.au/~cs9417/20T2/hw1/datasets.zip).
# Note: you will need to ensure the dataset files are in the same directory from which you are started the notebook.
# 
# **Please Note**: this homework uses datasets in the Attribute-Relation File Format (.arff). To load datasets from '.arff' formatted files, you will need to have installed the ```liac-arff``` package. You can do this using ```pip``` at the command-line, as follows:
# 
# ```
# $ pip install liac-arff
# ```
# 

# ## Question 1
# 
# For this question the objective is to run two different learning algorithms on a range of different sample sizes taken from the same training set to assess the effect of training sample size on classification error as estimated by cross-validation . You will use the nearest neighbour classifier and the decision tree classifier to generate two different sets of "learning curves" on 8 real-world datasets:
# 
# ```
# anneal.arff
# audiology.arff
# autos.arff
# credit-a.arff
# hypothyroid.arff
# letter.arff
# microarray.arff
# vote.arff
# ```
# 
# 
# ### Running the classifiers  [2 marks]
# You will run the following code section, and save the results to a plain text file "q1.out". Note that this may take a little time to run, so be patient ! Code has been added to save the output for you. However, you will need to write your own code to compute the error reduction for question 1(b).
# 
# The output of the code section comprises two tables, which represent the percentage error of classification for the nearest neighbour and the decision tree algorithm respectively. The first column contains the result of the baseline classifier, which simply predicts the majority class. From the second column on, the results are obtained by running the nearest neighbour or decision tree algorithms on $10\%$, $25\%$, $50\%$, $75\%$, and $100\%$ of the data. The standard deviation are shown in brackets, and where an asterisk is present, it indicates that the result is significantly different from the baseline.
# 
# 
# ### Result interpretation  [5 marks]
# Answer these questions in the file called [*answers.txt*](http://www.cse.unsw.edu.au/~cs9417/20T2/hw1/answers.txt). Your answers must be based on the results you saved in "q1.out". **_Please note_**: the goal of these questions is to attempt to **_explain why_** you think the results you obtained are as they are.
# 
# **1(a). [1 mark]** Refer to [*answers.txt*](http://www.cse.unsw.edu.au/~cs9417/20T2/hw1/answers.txt).
# 
# **1(b). [4 marks]** For each algorithm over all of the datasets, find the average change in error when moving from the default prediction to learning from 10% of the training set as follows.
# Let the error on the base line be err<sub>0</sub> and the error on 10% of the training set be error<sub>10</sub>.
# For each algorithm, calculate the percentage reduction in error relative to the default on each dataset as:
# 
# \begin{equation*}
# \frac{err_0 - err_{10}}{err_{0}} \times 100.
# \end{equation*}
# 
# Now repeat exactly the same process by comparing the two classifiers over all of the datasets, learning from $100\%$ of the training set, compared to default. Organise your results by grouping them into a 2 by 2 table, 
# like this:
# 
# <table style="width:64%">
#     <caption>Mean error reduction relative to default:</caption>
#   <tr>
#     <th>Algorithm</td>
#     <th>After 10% training</td> 
#     <th>After 100% training</td>
#   </tr>
#   <tr>
#     <th>Nearest Neighbour</td>
#     <td>Your result</td> 
#     <td>Your result</td>
#   </tr>
#   <tr>
#     <th>Decision Tree</td>
#     <td>Your result</td> 
#     <td>Your result</td>
#   </tr>
# </table>
# 
# 
# The "Your result" entries from this table should now be inserted into the correct places in your file [*answers.txt*](http://www.cse.unsw.edu.au/~cs9417/20T2/hw1/answers.txt).
# 
# Once you have done this, complete the rest of the answers for Question 1(b) in your file [*answers.txt*](http://www.cse.unsw.edu.au/~cs9417/20T2/hw1/answers.txt).
# 

# In[1]:


# Code for Question 1

import arff
import numpy as np
from itertools import product
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score, KFold
from sklearn.utils import resample
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import ttest_ind

seeds = [2, 3, 5, 7, 11, 13, 17, 23, 29, 31, 37]
score_list = []

dir_name = 'datasets/'
for fname in ["anneal.arff", "audiology.arff", "autos.arff", "credit-a.arff",               "hypothyroid.arff", "letter.arff", "microarray.arff", "vote.arff"]:
    dataset = arff.load(open(dir_name + fname), 'r')
    data = np.array(dataset['data'])

    X = np.array(data)[:, :-1]
    Y = np.array(data)[:, -1]

    # turn unknown/none/? into a separate value
    for i, j in product(range(len(data)), range(len(data[0]) - 1)):
        if X[i, j] is None:
            X[i, j] = len(dataset['attributes'][j][1])

    # a hack to turn negative categories positive for autos.arff
    for i in range(Y.shape[0]):
        if Y[i] < 0:
            Y[i] += 7

    # identify and extract categorical/non-categorical features
    categorical, non_categorical = [], []
    for i in range(len(dataset['attributes']) - 1):
        if isinstance(dataset['attributes'][i][1], str):
            non_categorical.append(X[:, i])
        else:
            categorical.append(X[:, i])

    categorical = np.array(categorical).T
    non_categorical = np.array(non_categorical).T

    if categorical.shape[0] == 0:
        transformed_X = non_categorical
    else:
        # encode categorical features
        # encoder = OneHotEncoder(n_values = 'auto',
        #                        categorical_features = 'all',
        #                        dtype = np.int32,
        #                        sparse = False,
        #                        handle_unknown = 'error')
        encoder = OneHotEncoder(categories = 'auto',
                                dtype = np.int32,
                                sparse = False,
                                handle_unknown = 'error')
        encoder.fit(categorical)
        categorical = encoder.transform(categorical)
        if non_categorical.shape[0] == 0:
            transformed_X = categorical
        else:
            transformed_X = np.concatenate((categorical, non_categorical), axis = 1)

    # concatenate the feature array and the labels for resampling purpose
    Y = np.array([Y], dtype = np.int)
    input_data = np.concatenate((transformed_X, Y.T), axis = 1)

    # build the models
    models = [DummyClassifier(strategy = 'most_frequent')]               + [KNeighborsClassifier(n_neighbors = 1, algorithm = "brute")] * 5               + [DecisionTreeClassifier()] * 5

    # resample and run cross validation
    portion = [1.0, 0.1, 0.25, 0.5, 0.75, 1.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    sample, scores = [None] * 11, [None] * 11
    for i in range(11):
        sample[i] = resample(input_data,
                             replace = False,
                             n_samples = int(portion[i] * input_data.shape[0]),
                             random_state = seeds[i])
        score = [None] * 10
        for j in range(10):
            score[j] = np.mean(cross_val_score(models[i],
                                               sample[i][:, :-1],
                                               sample[i][:, -1].astype(np.int),
                                               scoring = 'accuracy',
                                               cv = KFold(10, True, seeds[j])))
        scores[i] = score

    score_list.append((fname[:-5], 1 - np.array(scores)))

# print the results
header = ["{:^123}".format("Nearest Neighbour Results") + '\n' + '-' * 123  + '\n' +           "{:^15} | {:^10} | {:^16} | {:^16} | {:^16} | {:^16} | {:^16}"           .format("Dataset", "Baseline", "10%", "25%", "50%", "75%", "100%"),
          "{:^123}".format("Decision Tree Results") + '\n' + '-' * 123  + '\n' + \
          "{:^15} | {:^10} | {:^16} | {:^16} | {:^16} | {:^16} | {:^16}" \
          .format("Dataset", "Baseline", "10%", "25%", "50%", "75%", "100%")]
offset = [1, 6]


for k in range(2):
    print(header[k])
    for i in range(8):
        scores = score_list[i][1]
        p_value = [None] * 5
        for j in range(5):
            _, p_value[j] = ttest_ind(scores[0], scores[j + offset[k]], equal_var = False)

        print("{:<15} | {:>10.2%}".format(score_list[i][0], np.mean(scores[0])), end = '')
        for j in range(5):
            print(" | {:>6.2%} ({:>5.2%}) {}" .format(np.mean(scores[j + offset[k]]),
                                                      np.std(scores[j + offset[k]]),
                                                      '*' if p_value[j] < 0.05 else ' '), end = '')
        print()
    print()

with open('q1.out', 'w') as f1:
    for k in range(2):
        print(header[k], file=f1)
        for i in range(8):
            scores = score_list[i][1]
            p_value = [None] * 5
            for j in range(5):
                _, p_value[j] = ttest_ind(scores[0], scores[j + offset[k]], equal_var = False)

            print("{:<15} | {:>10.2%}".format(score_list[i][0], np.mean(scores[0])), end = '', file=f1)
            for j in range(5):
                print(" | {:>6.2%} ({:>5.2%}) {}" .format(np.mean(scores[j + offset[k]]),
                                                          np.std(scores[j + offset[k]]),
                                                          '*' if p_value[j] < 0.05 else ' '), end = '', file=f1)
            print(file=f1)
        print(file=f1)


# ## Question 2
# 
# This question involves mining text data, for which machine learning algorithms typically use a transformation into a dataset of "word counts". In the original dataset each text example is a string of words with a class label, and the sklearn transform converts this to a vector of word counts.
# 
# The dataset contains "snippets", short sequences of words taken from Google searches, each of which has been labelled with one of 8 classes, referred to as "sections", such as business, sports, etc. The dataset is provided already split into a training set of $10,060$ snippets and a test set of $2,280$ snippets (for convenience, the combined dataset is also provided as a separate file). 
# 
# Using a vector representation for text data means that we can use many of the standard classifier learning methods. However, such datasets are often highly _sparse_ in the sense that, for any example (i.e., piece of text), nearly all of its feature values are zero. To tackle this problem, we typically apply methods of feature selection (or dimensionality reduction). In this question you will investigate the effect of using the [SelectKBest](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html) method to select the $K$ best features (words or tokens in this case) that appear to help classification accuracy.
# 
# ### Running the classifier  [1 mark]
# You will run the following code section, which will show the results and save them to a plain text file "q2.out". 
# 
# The output of the code section is 5 lines of output, each of which represents the percentage accuracy of classification on training and test set for different amounts of feature selection.
# The first such line represents the "default", i.e., using all features. The remaining 4 lines show the effect of learning and predicting on text data where only the top $K$ features are being used.
# 
# 
# ### Result interpretation  [2 marks]
# Answer this question in the file called [*answers.txt*](http://www.cse.unsw.edu.au/~cs9417/20T2/hw1/answers.txt). Your answers must be based on the results you saved in "q2.out".
# 
# **2. [2 marks]** Refer to [*answers.txt*](http://www.cse.unsw.edu.au/~cs9417/20T2/hw1/answers.txt).

# In[36]:


# Code for Question 2

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, chi2

dir_name = 'datasets/'
df_trte = pd.read_csv(dir_name + 'snippets_all.csv')
df_tr = pd.read_csv(dir_name + 'snippets_train.csv')
df_te = pd.read_csv(dir_name + 'snippets_test.csv')

# Set up the vocabulary (the global set of "words" or tokens) for training and test datasets
vectorizer = CountVectorizer()
vectorizer.fit(df_trte.snippet)

# Apply this vocabulary to transform the text snippets to vectors of word counts
X_train = vectorizer.transform(df_tr.snippet)
X_test = vectorizer.transform(df_te.snippet)
y_train = df_tr.section
y_test = df_te.section

# See dimensions of training and test datasets
print("X train: ", X_train.shape)
print("X test: ", X_test.shape)
print("Y train: ", y_train.shape)
print("Y test: ", y_test.shape)

# Learn a Naive Bayes classifier on the training set
clf = MultinomialNB(alpha=0.5)
MultinomialNB(alpha=0.5, class_prior=None, fit_prior=True)
clf.fit(X_train, y_train)
pred_train = clf.predict(X_train)
score_train = metrics.accuracy_score(y_train, pred_train)
pred_test = clf.predict(X_test)
score_test = metrics.accuracy_score(y_test, pred_test)

f2 = open('q2.out', 'w')
print("Train/test accuracy using all features: ", score_train, score_test)
print("Train/test accuracy using all features: ", score_train, score_test, file=f2)

# Use Chi^2 criterion to select top 10000 features
ch2_10000 = SelectKBest(chi2, k=10000)
ch2_10000.fit(X_train, y_train)
# Project training data onto top 10000 selected features
X_train_kbest_10000 = ch2_10000.transform(X_train)
# Train NB Classifier using top 10 selected features
clf_kbest_10000 = MultinomialNB(alpha=0.5, class_prior=None, fit_prior=True)
clf_kbest_10000.fit(X_train_kbest_10000,y_train)
# Predictive accuracy on training set
pred_train_kbest_10000 = clf_kbest_10000.predict(X_train_kbest_10000)
score_train_kbest_10000 = metrics.accuracy_score(y_train,pred_train_kbest_10000)
# Project test data onto top 10000 selected features
X_test_kbest_10000 = ch2_10000.transform(X_test)
# Predictive accuracy on test set
pred_test_kbest_10000 = clf_kbest_10000.predict(X_test_kbest_10000)
score_test_kbest_10000 = metrics.accuracy_score(y_test,pred_test_kbest_10000)

print("Train/test accuracy for top 10K features", score_train_kbest_10000, score_test_kbest_10000)
print("Train/test accuracy for top 10K features", score_train_kbest_10000, score_test_kbest_10000, file=f2)

# Use Chi^2 criterion to select top 1000 features
ch2_1000 = SelectKBest(chi2, k=1000)
ch2_1000.fit(X_train, y_train)
# Project training data onto top 1000 selected features
X_train_kbest_1000 = ch2_1000.transform(X_train)
# Train NB Classifier using top 1000 selected features
clf_kbest_1000 = MultinomialNB(alpha=0.5, class_prior=None, fit_prior=True)
clf_kbest_1000.fit(X_train_kbest_1000,y_train)
# Predictive accuracy on training set
pred_train_kbest_1000 = clf_kbest_1000.predict(X_train_kbest_1000)
score_train_kbest_1000 = metrics.accuracy_score(y_train,pred_train_kbest_1000)
# Project test data onto top 1000 selected features
X_test_kbest_1000 = ch2_1000.transform(X_test)
# Predictive accuracy on test set
pred_test_kbest_1000 = clf_kbest_1000.predict(X_test_kbest_1000)
score_test_kbest_1000 = metrics.accuracy_score(y_test,pred_test_kbest_1000)

print("Train/test accuracy for top 1K features", score_train_kbest_1000, score_test_kbest_1000)
print("Train/test accuracy for top 1K features", score_train_kbest_1000, score_test_kbest_1000, file=f2)

# Use Chi^2 criterion to select top 100 features
ch2_100 = SelectKBest(chi2, k=100)
ch2_100.fit(X_train, y_train)
# Project training data onto top 100 selected features
X_train_kbest_100 = ch2_100.transform(X_train)
# Train NB Classifier using top 100 selected features
clf_kbest_100 = MultinomialNB(alpha=0.5, class_prior=None, fit_prior=True)
clf_kbest_100.fit(X_train_kbest_100,y_train)
# Predictive accuracy on training set
pred_train_kbest_100 = clf_kbest_100.predict(X_train_kbest_100)
score_train_kbest_100 = metrics.accuracy_score(y_train,pred_train_kbest_100)
# Project test data onto top 100 selected features
X_test_kbest_100 = ch2_100.transform(X_test)
# Predictive accuracy on test set
pred_test_kbest_100 = clf_kbest_100.predict(X_test_kbest_100)
score_test_kbest_100 = metrics.accuracy_score(y_test,pred_test_kbest_100)

print("Train/test accuracy for top 100 features", score_train_kbest_100, score_test_kbest_100)
print("Train/test accuracy for top 100 features", score_train_kbest_100, score_test_kbest_100, file=f2)

# Use Chi^2 criterion to select top 10 features
ch2_10 = SelectKBest(chi2, k=10)
ch2_10.fit(X_train, y_train)
# Project training data onto top 10 selected features
X_train_kbest_10 = ch2_10.transform(X_train)
# Train NB Classifier using top 10 selected features
clf_kbest_10 = MultinomialNB(alpha=0.5, class_prior=None, fit_prior=True)
clf_kbest_10.fit(X_train_kbest_10,y_train)
# Predictive accuracy on training set
pred_train_kbest_10 = clf_kbest_10.predict(X_train_kbest_10)
score_train_kbest_10 = metrics.accuracy_score(y_train,pred_train_kbest_10)
# Project test data onto top 10 selected features
X_test_kbest_10 = ch2_10.transform(X_test)
# Predictive accuracy on test set
pred_test_kbest_10 = clf_kbest_10.predict(X_test_kbest_10)
score_test_kbest_10 = metrics.accuracy_score(y_test,pred_test_kbest_10)

print("Train/test accuracy for top 10 features", score_train_kbest_10, score_test_kbest_10)
print("Train/test accuracy for top 10 features", score_train_kbest_10, score_test_kbest_10, file=f2)
f2.close()


# In[ ]:





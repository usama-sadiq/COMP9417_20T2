# COMP9417      HOMEWORK 1      Year: 2020 Term: 2

## AIM
- Apply different classifier learning algorithms implemented in the Python scikit-learn machine learning library on a variety of datasets and answer questions based on anaylsis and intrepetation of the empirical results, using my knowledge of machine learning.

## LEARNING OUTCOMES
- set up replicated k-fold cross-validation experiments to obtain average performance measures of algorithms on datasets.

- compare the performance of different algorithms against a base-line and each other.

- aggregate comparative performance scores for algorithms over a range of different datasets.

- propose properties of algorithms and their parameters, or datasets, which may lead to performance differences being observed.

- suggest reasons for actual observed performance differences in terms of properties of algorithms, parameter settings or datasets.

- apply methods for data transformations and parameter search and evaluate their effects on the performance of algorithms.

## OBJECTIVES

### QUESTION 1

- Run nearest neigbour classifier and the decision tree classifier on a range of different sample sizes taken from the same training set to assess the effect of training sample size on classification error as estimated by cross-validation.

### QUESTION 2

- Mining text data, for which machine learning algorithms use a transformation of dataset into a dataset of "word counts".

- Using a vector representation for text data enables us to use many of the standard classifier learning methods.

- Problem with this approach is datasets like this are often highly sparse. For example, for any example (piece of text), nearly all of its feature values are zero.

- To solve this problem, we apply methods of feature selection (or dimensionality reduction).

- Investigate the effect of using the SelectKBest method to select the K best features (words or tokens in this case) that appear to help classification accuracy.

#### DATASET

- Each text example  is a string of words with a class label and sklearn transform converts this to a vector of word counts.

- The dataset contains "snippets", short sequence of words taken from Google searches, each of which has been labelled with one of the 8 classes, referred to as "sections", such as business, spoorts, etc.

- The dataset is split into a training set of 10,060 snippets and a test set of 2,280 snippets.


## IMPLEMENTATION

### QUESTION 1

#### PART A

- Run the code section and save the results to a text file "q1.out"

#### PART B

- Write a code for each machine learning algorithm to calculate the precentage reduction in error relative to the base on each dataset and find average of all the 8 datasets.

### QUESTION 2

- Run the code section and save the results to a text file "q2.out"


## RESULTS

### QUESTION 1

#### PART A

<table style="width:100%">
  <tr>
    <th colspan="7">Nearest Neighbour Results</th>
  </tr>
  <tr>
    <td>Dataset</td>
    <td>Baseline</td>
    <td>10%</td>
    <td>25%</td>
    <td>50%</td>
    <td>75%</td>
    <td>100%</td>
  </tr>
</table>
                                                 Nearest Neighbour Results                                                 
---------------------------------------------------------------------------------------------------------------------------
    Dataset     |  Baseline  |       10%        |       25%        |       50%        |       75%        |       100%      
anneal          |     23.83% | 20.31% (0.94%) * | 18.00% (1.33%) * | 11.18% (0.77%) * |  9.11% (0.37%) * |  7.44% (0.44%) *
audiology       |     74.77% | 60.17% (2.17%) * | 42.00% (2.56%) * | 31.85% (2.13%) * | 29.62% (1.78%) * | 26.47% (1.81%) *
autos           |     67.35% | 64.50% (1.50%) * | 61.40% (2.21%) * | 65.96% (2.02%)   | 52.92% (2.39%) * | 57.37% (0.95%) *
credit-a        |     44.49% | 39.98% (1.05%) * | 41.35% (0.99%) * | 32.04% (1.50%) * | 34.63% (0.79%) * | 34.71% (0.73%) *
hypothyroid     |      7.71% |  8.27% (0.52%) * |  7.33% (0.18%) * |  4.74% (0.14%) * |  5.01% (0.13%) * |  4.79% (0.10%) *
letter          |     96.26% | 16.86% (0.35%) * |  9.61% (0.20%) * |  6.05% (0.08%) * |  4.71% (0.06%) * |  3.93% (0.07%) *
microarray      |     50.20% | 59.47% (2.55%) * | 49.58% (2.36%)   | 42.45% (0.83%) * | 50.71% (0.95%)   | 50.88% (0.60%)  
vote            |     38.63% |  6.45% (1.01%) * | 10.42% (1.16%) * |  8.26% (0.55%) * |  7.12% (0.19%) * |  7.91% (0.39%) *

                                                   Decision Tree Results                                                   
---------------------------------------------------------------------------------------------------------------------------
    Dataset     |  Baseline  |       10%        |       25%        |       50%        |       75%        |       100%      
anneal          |     23.83% |  9.32% (1.05%) * |  4.02% (0.68%) * |  1.40% (0.46%) * |  1.35% (0.33%) * |  0.81% (0.26%) *
audiology       |     74.77% | 61.83% (4.25%) * | 47.40% (4.43%) * | 30.48% (1.63%) * | 21.39% (1.22%) * | 23.09% (2.37%) *
autos           |     67.35% | 71.50% (3.91%) * | 48.03% (3.55%) * | 34.18% (2.16%) * | 29.12% (2.81%) * | 20.43% (3.17%) *
credit-a        |     44.49% | 19.81% (2.66%) * | 12.59% (1.93%) * | 20.02% (1.58%) * | 19.50% (1.20%) * | 19.19% (1.12%) *
hypothyroid     |      7.71% |  2.92% (0.43%) * |  1.52% (0.23%) * |  0.61% (0.05%) * |  0.74% (0.11%) * |  0.62% (0.07%) *
letter          |     96.26% | 29.07% (0.29%) * | 21.40% (0.39%) * | 16.45% (0.32%) * | 13.31% (0.15%) * | 11.80% (0.14%) *
microarray      |     50.20% | 52.90% (4.12%)   | 52.06% (2.86%)   | 50.35% (2.42%)   | 46.09% (2.41%) * | 49.25% (1.74%)  
vote            |     38.63% | 14.05% (3.67%) * |  5.52% (1.34%) * |  7.19% (0.96%) * |  3.29% (0.56%) * |  5.84% (0.43%) *

- The output of the code section comprises two tables.
- Each table represents the percentage error of classification for the nearest neighbour and the decision tree algorithm respectively.
- The first column contains the result of the baseline classifier, which simply predicts the majority class.
- From the second column on, the results are obtained by running the nearest neighbour or decision tree algorithms on 10%, 25%, 50%, 75%, and 100% of the data.
- The standard deviation are shown in brackets.
- Asterisk represents a result significantly different from the baseline.


#### PART B

********************************************************************************
***                 Mean error reduction relative to default                    
********************************************************************************
*** Algorithm            After 10% training	After 100% training                
********************************************************************************
*** Nearest Neighbour      23.60                    47.77                       
*** Decision Tree          39.71                    69.84                       
********************************************************************************

### QUESTION 2

Train/test accuracy using all features:  0.979324055666004 0.8052631578947368
Train/test accuracy for top 10K features 0.9717693836978131 0.8052631578947368
Train/test accuracy for top 1K features 0.9538767395626243 0.6942982456140351
Train/test accuracy for top 100 features 0.7187872763419483 0.4631578947368421
Train/test accuracy for top 10 features 0.4092445328031809 0.18903508771929825

- Each line represents the percentage accuracy of classification on training and test set for different amounts of feature selection.
- The first line represents the "default" i.e. using all the features.
- The remaining 4 lines show the effect of learning and predicting on text data where only the top $K$ features are being used.

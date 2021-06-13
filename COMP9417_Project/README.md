# COMP9417   PROJECT   Year: 2020      TERM: 02


# AIM

- Implement a ‘Transaction Fraud Prevention System’ leveraging machine learning models,
which aims to predict whether a given financial transaction is ‘Fraudulent’ or ‘Valid’.

# DATASET

- The dataset for the model was taken from the Kaggle competition: https://www.kaggle.com/c/ieee-fraud-detection and
was provided via the collaboration of IEEE and Vesta Corporation.

- Give a short description about the dataset.

# IMPLEMENTATION

## SUMMARY

- After solving class imbalance, leveraging feature selection and Exploratory Data Analysis, we executed tested the
following models for the given data:
    1) Decision Tree: This was our baseline model.
    2) Bernoulli Naive Bayes.
    3) K-Nearest Neighbour.
    4) SVM: We could not get the conclusive answer via the SVM.
    5) Random Forest.
    6) Light Gradient Boost.
    7) Integrated Stacked Model.

- The final model is an LGB model with hyper parameter tuning giving the Kaggle Score of 93.

## EDA


# RESULTS

<table style="width:100%">
  
  <tr>
    <th>Model</th>
    <th>Parameters</th>
    <th>Kaggle Score</th>
  </tr>

  <tr>
    <td>Decision Tree</td>
    <td>random_state = 0, criterion = 'entropy', max_depth = 30, splitter = 'best', min_samples_split = 30</td>
    <td>0.70</td>
  </tr>

  <tr>
    <td>Naive Bayes</td>
    <td>Alpha = 0.01, prior_class = True</td>
    <td>0.75</td>
  </tr>

  <tr>
    <td>K Nearest Neighbour</td>
    <td></td>
    <td>0.67</td>
  </tr>

  <tr>
    <td>Random Forest</td>
    <td>n_estimators = 1000, random_state = 121, min_samples_split = 2, bootstrap = False, max_depth = 5</td>
    <td>0.87</td>
  </tr>

  <tr>
    <td><b>Light Gradient Boosting Machine</b></td>
    <td><b>objective = binary, n_estimators = 700, learning_rate = 0.1, num_leaves = 50, max_depth = 7, subsample = 0.9, colsample_bytree = 0.9, random_state = 108</b></td>
    <td><b>0.92</b></td>
  </tr>

  <tr>
    <td>Integrated Stacked Model</td>
    <td>Decision Tree + Naive Bayes + K-Nearest Neighbour + Random Forest + Light Gradient Boosting Machine</td>
    <td>0.77</td>
  </tr>

</table>

<p style="font-size=11px"><b>Light Gradient Boost Machine was chosen as the final model with the final prediction acurracy of 0.92</b></p>
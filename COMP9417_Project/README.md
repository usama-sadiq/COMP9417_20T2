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
        <td>random_state=0, <br> criterion='entropy', <br> max_depth=30, <br> splitter='best', <br> min_samples_split=30</td>
    </tr>
</table>
# COMP9417   PROJECT   Year: 2020      TERM: 02


# AIM

- Implement a ‘Transaction Fraud Prevention System’ leveraging machine learning models,
which aims to predict whether a given financial transaction is ‘Fraudulent’ or ‘Valid’.

# DATASET

- The dataset for the model was taken from the Kaggle competition: https://www.kaggle.com/c/ieee-fraud-detection and
was provided via the collaboration of IEEE and Vesta Corporation.

## TRANSACTION TABLE

- TransactionDT: timedelta from a given reference datetime (not an actual timestamp).
- TransactionAMT: transaction payment amount in USD.
- (*) ProductCD: product code -> the product for each transaction. (categorical feature)
- (*) [card1, card2, card3, card4, card5, card6] : payment card information For example card type, card category, issue bank, country, etc. (categorical feature)
- (*) addr1: address. (categorical feature)
- (*) addr2: address. (categorical feature)
- dist: distance.
- (*) P_emaildomain: Purchaser email domain. (categorical feature)
- (*) R_emaildomain: Recipient email domain. (categorical feature)
- [C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14]: The actual meaning is masked but can be said as a count such as how many addresses are found to associated with the payment card.
- [D1,D2,D3,D4,D5,D6,D7,D8,D9,D10,D11,D12,D13,D14,D15]: timedelta in days between previous transaction.
- (*) [M1,M2,M3,M4,M5,M6,M7,M8,M9]: match such as names on card and address etc. (categorical feature)
- Vxxx: Vesta engineered rich features such as ranking, counting and other entity relations.


## IDENTITY TABLE

- The field names are masked for privacy protection and contract agreement as part fo Vesta's policies.

- Mostly fields are related to identity information such as network connection information.

### CATEGORICAL FEATURES
- DeviceType.
- DeviceInfo.
- id_12 - id_38.

<p><b>Note:</b> All data description are provided by the competition host (Vesta) at https://www.kaggle.com/c/ieee-fraud-detection/discussion/101203</p>

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

## MACHINE LEARNING MODELS

<table style="width:100%">

</table>


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

<p><b>* Light Gradient Boost Machine was chosen as the final model with the final prediction score of 0.92</b></p>

# CONTRIBUTORS

- Usama Sadiq. (Github Profile: https://github.com/usama-sadiq)
- Mohit Khanna. (Github Profile: https://github.com/mohitKhanna1411)
- Uttkarsh Sharma. (Github Profile: https://github.com/khaamosh)
- Sibo Zhang. (Github Profile: https://github.com/sibozhang400)
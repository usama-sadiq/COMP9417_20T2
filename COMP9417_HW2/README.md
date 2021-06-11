# COMP9417      HOMEWORK 2      Year: 2020 Term: 2

## AIM
- Apply parameter search for machine learning algorithms implemented in the Python scikit-learn machine learning library.
- Answer questions based on your analysis and interpretation of the empirical results of such applications, using your knowledge of machine learning.
- Complete an implementation of a different version of a learning algorithm you have previously seen.

## LEARNING OUTCOMES
- Set up a simple grid search over different hyper-parameter settings based on  𝑘 -fold cross-validation to obtain performance measures on different datasets.
- Compare the performance measures of different algorithm settings.
- Propose properties of algorithms and their hyper-parameters, or datasets, which may lead to performance differences being observed.
- Suggest reasons for actual observed performance differences in terms of properties of algorithms, parameter settings or datasets.
- Read and understand incomplete code for a learning algorithm to the point of being able to complete the implementation and run it successfully on a dataset.

## OBJECTIVES

### QUESTION 1

- Dealing with noisy data is a key issue in machine learning.
- Alogirthms that have noise-handling mechanisms built-in, like decision trees, can overfit noisy data, unless their "overfitting avoidance" or regularization hyper-parameters are set properly.
- Use datasets that have had various amounts of "class noise" added by randomly changing the actual class value to a different one for a specified percentage of the training data.
- Three arbitrarily chosen levels of noise specified are: low ( 20% ), medium ( 50% ) and high ( 80% ).
- The learning algorithm must try to learn the best model it can, which is later  evaluated on the test data.
- Test data does not contain noise.
- The algorithm is evaluated on how well it has avoided fitting the noise.
- Let the algorithm do a limited grid search using cross-validation for the best over-fitting avoidance parameter settings on each training set

### QUESTION 2

- Implement a simple recurrent neural network (RNN) to predict the last character in the word.
- Input to RNN: First 9 characters of a word
- Predict the 10th character.
- If there are fewer than 10 characters iin a word, spaces are used to pad it.
- Implement the back-propagation through time.

#### DATASET

- Dataset consists of partial words (wrods without the last character).


## IMPLEMENTATION

### QUESTION 1

#### PART A

- Run the code section and insert the table of results in the answer.txt file.


### QUESTION 2

- Insert the correct code in the TO DO sections to complete the implementation of the RNN.

--- TO DO: setup for the current step  [2 marks]
last_input = outputs.pop()
layer_input = numpy.concatenate((last_input,X[:,step,:]),axis=1)
s = self.input_size - 1
if (step == s):
	weight = self.weights[1][:64,:]
else:
        weight = self.weights[0][:64,:]


--- TO DO: calculate gradients  [1 mark]
gradients, dW, db = self.derivatives_of_hidden_layer(previous_gradients,layer_output,layer_input,weight)

--- TO DO: update weights  [2 marks]
self.weights[0] += self.learning_rate / X.shape[0] * dW
self.biases[0] += self.learning_rate / X.shape[0] * db

--- TO DO: setup for the next step  [2 marks]
previous_gradients = gradients
layer_output = last_input


## RESULTS

### QUESTION 1

#### PART A

<table style="width:100%">
  
  <tr>
    <th colspan="6">Decision Tree Results</th>
  </tr>

  <tr>
    <td>Dataset</td>
    <td>Default</td>
    <td>0%</td>
    <td>20%</td>
    <td>50%</td>
    <td>80%</td>
  </tr>

  <tr>
    <td>balance-scale</td>
    <td>36.70% ( 2)</td>
    <td>76.06% ( 2)</td>
    <td>71.28% (12)</td>
    <td>65.43% (27)</td>
    <td>18.09% (27)</td>
  </tr>

  <tr>
    <td>primary-tumor</td>
    <td>25.49% ( 2)</td>
    <td>37.25% (12)</td>
    <td>42.16% (12)</td>
    <td>43.14% (12)</td>
    <td>26.47% ( 7)</td>
  </tr>

  <tr>
    <td>glass</td>
    <td>44.62% ( 2)</td>
    <td>69.23% ( 7)</td>
    <td>66.15% (22)</td>
    <td>35.38% (17)</td>
    <td>29.23% (17)</td>
  </tr>

  <tr>
    <td>heart-h</td>
    <td>35.96% ( 2)</td>
    <td>67.42% ( 7)</td>
    <td>78.65% (22)</td>
    <td>56.18% (17)</td>
    <td>20.22% (27)</td>
  </tr>

</table>

- The output of the code section is a table, which represents the percentage accuracy of classification for the decision tree algorithm.
- The first column contains the result of the "Default" classifier, which is the decision tree algorithm with default parameter settings running on each of the datasets which have had  50%  noise added.
- From the second column on, in each column the results are obtained by running the decision tree algorithm on  0% ,  20% ,  50%  and  80%  noise added to each of the datasets.
- The number in the barackets represents the best value of min_samples leaf i.e. the minimum number of examples that can be used to make a prediction in the tree, on that dataset which is a result of a grid search.


### QUESTION 2

<p>In iteration 0, training accuracy is 0.005393258426966292.</p>
<p>In iteration 100, training accuracy is 0.0.</p>
<p>In iteration 500, training accuracy is 0.3096629213483146.</p>
<p>In iteration 1000, training accuracy is 0.42067415730337077.</p>
<p>In iteration 1500, training accuracy is 0.46831460674157305.</p>
<p>In iteration 2000, training accuracy is 0.5523595505617978.</p>
<p>In iteration 2500, training accuracy is 0.5797752808988764.</p>
<p>In iteration 3000, training accuracy is 0.6107865168539326.</p>
<p>In iteration 3500, training accuracy is 0.6426966292134831.</p>
<p>In iteration 4000, training accuracy is 0.6656179775280899.</p>
<p>In iteration 4500, training accuracy is 0.706067415730337.</p>
<p>In iteration 5000, training accuracy is 0.717752808988764.</p>
<p>Finished training in 349.1479959487915 seconds.</p>

<p>Testing accuracy: 0.6265060240963856</p>
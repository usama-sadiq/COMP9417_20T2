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
<p>In iteration 200, training accuracy is 0.0.</p>
<p>In iteration 300, training accuracy is 0.0.</p>
<p>In iteration 400, training accuracy is 0.0.</p>
<p>In iteration 500, training accuracy is 0.3096629213483146.</p>
<p>In iteration 600, training accuracy is 0.3649438202247191.</p>
<p>In iteration 700, training accuracy is 0.37258426966292135.</p>
<p>In iteration 800, training accuracy is 0.39280898876404496.</p>
<p>In iteration 900, training accuracy is 0.4049438202247191.</p>
<p>In iteration 1000, training accuracy is 0.42067415730337077.</p>
<p>In iteration 1100, training accuracy is 0.4341573033707865.</p>
<p>In iteration 1200, training accuracy is 0.4310112359550562.</p>
<p>In iteration 1300, training accuracy is 0.4543820224719101.</p>
<p>In iteration 1500, training accuracy is 0.46831460674157305.</p>
<p>In iteration 1400, training accuracy is 0.46202247191011236.</p>
<p>In iteration 1600, training accuracy is 0.4898876404494382.</p>
<p>In iteration 1700, training accuracy is 0.5092134831460674.</p>
<p>In iteration 1800, training accuracy is 0.5123595505617977.</p>
<p>In iteration 1900, training accuracy is 0.529438202247191.</p>
<p>In iteration 2000, training accuracy is 0.5523595505617978.</p>
<p>In iteration 2100, training accuracy is 0.5608988764044944.</p>
<p>In iteration 2200, training accuracy is 0.5730337078651685.</p>
<p>In iteration 2300, training accuracy is 0.5582022471910112.</p>
<p>In iteration 2400, training accuracy is 0.5635955056179776.</p>
<p>In iteration 2500, training accuracy is 0.5797752808988764.</p>
<p>In iteration 2600, training accuracy is 0.5829213483146067.</p>
<p>In iteration 2700, training accuracy is 0.5950561797752809.</p>
<p>In iteration 2800, training accuracy is 0.6035955056179775.</p>
<p>In iteration 2900, training accuracy is 0.6049438202247192.</p>
<p>In iteration 3000, training accuracy is 0.6107865168539326.</p>
<p>In iteration 3100, training accuracy is 0.6206741573033708.</p>
<p>In iteration 3200, training accuracy is 0.6229213483146068.</p>
<p>In iteration 3300, training accuracy is 0.6256179775280899.</p>
<p>In iteration 3400, training accuracy is 0.6408988764044944.</p>
<p>In iteration 3500, training accuracy is 0.6426966292134831.</p>
<p>In iteration 3600, training accuracy is 0.6507865168539326.</p>
<p>In iteration 3700, training accuracy is 0.6575280898876404.</p>
<p>In iteration 3800, training accuracy is 0.6683146067415731.</p>
<p>In iteration 3900, training accuracy is 0.6669662921348315.</p>
<p>In iteration 4000, training accuracy is 0.6656179775280899.</p>
<p>In iteration 4100, training accuracy is 0.6768539325842696.</p>
<p>In iteration 4200, training accuracy is 0.6867415730337079.</p>
<p>In iteration 4300, training accuracy is 0.6966292134831461.</p>
<p>In iteration 4400, training accuracy is 0.7020224719101124.</p>
<p>In iteration 4500, training accuracy is 0.706067415730337.</p>
<p>In iteration 4600, training accuracy is 0.7078651685393258.</p>
<p>In iteration 4700, training accuracy is 0.7123595505617978.</p>
<p>In iteration 4800, training accuracy is 0.7155056179775281.</p>
<p>In iteration 4900, training accuracy is 0.717752808988764.</p>
<p>In iteration 5000, training accuracy is 0.717752808988764.</p>
<p>Finished training in 349.1479959487915 seconds.</p>

<p>Expected         so, predicted         so</p>
<p>Expected         is, predicted         it</p>
<p>Expected         it, predicted         it</p>
<p>Expected        not, predicted        not</p>
<p>Expected       with, predicted       with</p>
<p>Expected         me, predicted         my</p>
<p>Expected         as, predicted         as</p>
<p>Expected       with, predicted       with</p>
<p>Expected       that, predicted       that</p>
<p>Expected       muse, predicted       must</p>
<p>Expected     stirrd, predicted     stirrs</p>
<p>Expected         by, predicted         be</p>
<p>Expected          a, predicted          o</p>
<p>Expected    painted, predicted    painted</p>
<p>Expected     beauty, predicted     beauty</p>
<p>Expected         to, predicted         to</p>
<p>Expected        his, predicted        his</p>
<p>Expected      verse, predicted      verse</p>
<p>Expected        who, predicted        who</p>
<p>Expected     heaven, predicted     heaves</p>
<p>Expected     itself, predicted     itself</p>
<p>Expected        for, predicted        for</p>
<p>Expected   ornament, predicted   ornament</p>
<p>Expected       doth, predicted       doth</p>
<p>Expected        use, predicted        use</p>
<p>Expected        and, predicted        and</p>
<p>Expected      every, predicted      every</p>
<p>Expected       fair, predicted       fair</p>
<p>Expected       with, predicted       with</p>
<p>Expected        his, predicted        his</p>
<p>Expected       fair, predicted       fair</p>
<p>Expected       doth, predicted       doth</p>
<p>Expected   rehearse, predicted   rehearst</p>
<p>Expected     making, predicted     making</p>
<p>Expected          a, predicted          o</p>
<p>Expected couplement, predicted couplement</p>
<p>Expected         of, predicted         or</p>
<p>Expected      proud, predicted      prous</p>
<p>Expected    compare, predicted    compare</p>
<p>Expected       with, predicted       with</p>
<p>Expected        sun, predicted        suh</p>
<p>Expected        and, predicted        and</p>
<p>Expected       moon, predicted       moor</p>
<p>Expected       with, predicted       with</p>
<p>Expected      earth, predicted      earts</p>
<p>Expected        and, predicted        and</p>
<p>Expected       seas, predicted       seaf</p>
<p>Expected       rich, predicted       rich</p>
<p>Expected       gems, predicted       geme</p>
<p>Expected       with, predicted       with</p>
<p>Expected     aprils, predicted     aprilf</p>
<p>Expected  firstborn, predicted  firstbors</p>
<p>Expected    flowers, predicted    flowers</p>
<p>Expected        and, predicted        and</p>
<p>Expected        all, predicted        all</p>
<p>Expected     things, predicted     thingt</p>
<p>Expected       rare, predicted       rare</p>
<p>Expected       that, predicted       that</p>
<p>Expected    heavens, predicted    heaveng</p>
<p>Expected        air, predicted        air</p>
<p>Expected         in, predicted         it</p>
<p>Expected       this, predicted       this</p>
<p>Expected       huge, predicted       huge</p>
<p>Expected    rondure, predicted    rondurs</p>
<p>Expected       hems, predicted       heme</p>
<p>Expected          o, predicted          o</p>
<p>Expected        let, predicted        let</p>
<p>Expected         me, predicted         my</p>
<p>Expected       true, predicted       trut</p>
<p>Expected         in, predicted         it</p>
<p>Expected       love, predicted       love</p>
<p>Expected        but, predicted        but</p>
<p>Expected      truly, predicted      truld</p>
<p>Expected      write, predicted      writh</p>
<p>Expected        and, predicted        and</p>
<p>Expected       then, predicted       thee</p>
<p>Expected    believe, predicted    believe</p>
<p>Expected         me, predicted         my</p>
<p>Expected         my, predicted         my</p>
<p>Expected       love, predicted       love</p>
<p>Expected         is, predicted         it</p>
<p>Expected         as, predicted         as</p>
<p>Expected       fair, predicted       fair</p>
<p>Expected         as, predicted         as</p>
<p>Expected        any, predicted        and</p>
<p>Expected    mothers, predicted    mothers</p>
<p>Expected      child, predicted      child</p>
<p>Expected     though, predicted     thougy</p>
<p>Expected        not, predicted        not</p>
<p>Expected         so, predicted         so</p>
<p>Expected     bright, predicted     bright</p>
<p>Expected         as, predicted         as</p>
<p>Expected      those, predicted      those</p>
<p>Expected       gold, predicted       gold</p>
<p>Expected    candles, predicted    candles</p>
<p>Expected       fixd, predicted       fixd</p>
<p>Expected         in, predicted         it</p>
<p>Expected    heavens, predicted    heaveng</p>
<p>Expected        air, predicted        air</p>
<p>Expected        let, predicted        let</p>
<p>Expected       them, predicted       thee</p>
<p>Expected        say, predicted        say</p>
<p>Expected       more, predicted       more</p>
<p>Expected       than, predicted       that</p>
<p>Expected       like, predicted       like</p>
<p>Expected         of, predicted         or</p>
<p>Expected    hearsay, predicted    hearsas</p>
<p>Expected       well, predicted       well</p>
<p>Expected          i, predicted          o</p>
<p>Expected       will, predicted       will</p>
<p>Expected        not, predicted        not</p>
<p>Expected     praise, predicted     praise</p>
<p>Expected       that, predicted       that</p>
<p>Expected    purpose, predicted    purpose</p>
<p>Expected        not, predicted        not</p>
<p>Expected         to, predicted         to</p>
<p>Expected       sell, predicted       self</p>
<p>Expected         my, predicted         my</p>
<p>Expected      glass, predicted      glass</p>
<p>Expected      shall, predicted      shall</p>
<p>Expected        not, predicted        not</p>
<p>Expected   persuade, predicted   persuade</p>
<p>Expected         me, predicted         my</p>
<p>Expected          i, predicted          o</p>
<p>Expected         am, predicted         as</p>
<p>Expected        old, predicted        old</p>
<p>Expected         so, predicted         so</p>
<p>Expected       long, predicted       long</p>
<p>Expected         as, predicted         as</p>
<p>Expected      youth, predicted      youth</p>
<p>Expected        and, predicted        and</p>
<p>Expected       thou, predicted       thou</p>
<p>Expected        are, predicted        art</p>
<p>Expected         of, predicted         or</p>
<p>Expected        one, predicted        one</p>
<p>Expected       date, predicted       date</p>
<p>Expected        but, predicted        but</p>
<p>Expected       when, predicted       whet</p>
<p>Expected         in, predicted         it</p>
<p>Expected       thee, predicted       thee</p>
<p>Expected      times, predicted      times</p>
<p>Expected    furrows, predicted    furrows</p>
<p>Expected          i, predicted          o</p>
<p>Expected     behold, predicted     behold</p>
<p>Expected       then, predicted       thee</p>
<p>Expected       look, predicted       lood</p>
<p>Expected          i, predicted          o</p>
<p>Expected      death, predicted      deats</p>
<p>Expected         my, predicted         my</p>
<p>Expected       days, predicted       daye</p>
<p>Expected     should, predicted     should</p>
<p>Expected    expiate, predicted    expiats</p>
<p>Expected        for, predicted        for</p>
<p>Expected        all, predicted        all</p>
<p>Expected       that, predicted       that</p>
<p>Expected     beauty, predicted     beauty</p>
<p>Expected       that, predicted       that</p>
<p>Expected       doth, predicted       doth</p>
<p>Expected      cover, predicted      coves</p>
<p>Expected       thee, predicted       thee</p>
<p>Expected         is, predicted         it</p>
<p>Expected        but, predicted        but</p>
<p>Expected        the, predicted        the</p>
<p>Expected     seemly, predicted     seemly</p>
<p>Expected    raiment, predicted    raiment</p>
<p>Expected         of, predicted         or</p>
<p>Expected         my, predicted         my</p>
<p>Expected      heart, predicted      heart</p>
<p>Expected      which, predicted      which</p>
<p>Expected         in, predicted         it</p>
<p>Expected        thy, predicted        the</p>
<p>Expected     breast, predicted     breast</p>
<p>Expected       doth, predicted       doth</p>
<p>Expected       live, predicted       live</p>
<p>Expected         as, predicted         as</p>
<p>Expected      thine, predicted      thine</p>
<p>Expected         in, predicted         it</p>
<p>Expected         me, predicted         my</p>
<p>Expected        how, predicted        hom</p>
<p>Expected        can, predicted        cas</p>
<p>Expected          i, predicted          o</p>
<p>Expected       then, predicted       thee</p>
<p>Expected         be, predicted         be</p>
<p>Expected      elder, predicted      elded</p>
<p>Expected       than, predicted       that</p>
<p>Expected       thou, predicted       thou</p>
<p>Expected        art, predicted        art</p>
<p>Expected          o, predicted          o</p>
<p>Expected  therefore, predicted  therefore</p>
<p>Expected       love, predicted       love</p>
<p>Expected         be, predicted         be</p>
<p>Expected         of, predicted         or</p>
<p>Expected    thyself, predicted    thyself</p>
<p>Expected         so, predicted         so</p>
<p>Expected       wary, predicted       ware</p>
<p>Expected         as, predicted         as</p>
<p>Expected          i, predicted          o</p>
<p>Expected        not, predicted        not</p>
<p>Expected        for, predicted        for</p>
<p>Expected     myself, predicted     myself</p>
<p>Expected        but, predicted        but</p>
<p>Expected        for, predicted        for</p>
<p>Expected       thee, predicted       thee</p>
<p>Expected       will, predicted       will</p>
<p>Expected    bearing, predicted    bearing</p>
<p>Expected        thy, predicted        the</p>
<p>Expected      heart, predicted      heart</p>
<p>Expected      which, predicted      which</p>
<p>Expected          i, predicted          o</p>
<p>Expected       will, predicted       will</p>
<p>Expected       keep, predicted       keet</p>
<p>Expected         so, predicted         so</p>
<p>Expected      chary, predicted      chard</p>
<p>Expected         as, predicted         as</p>
<p>Expected     tender, predicted     tender</p>
<p>Expected      nurse, predicted      nurst</p>
<p>Expected        her, predicted        her</p>
<p>Expected       babe, predicted       babe</p>
<p>Expected       from, predicted       from</p>
<p>Expected     faring, predicted     faring</p>
<p>Expected        ill, predicted        ill</p>
<p>Expected    presume, predicted    presume</p>
<p>Expected        not, predicted        not</p>
<p>Expected         on, predicted         or</p>
<p>Expected        thy, predicted        the</p>
<p>Expected      heart, predicted      heart</p>
<p>Expected       when, predicted       whet</p>
<p>Expected       mine, predicted       ming</p>
<p>Expected         is, predicted         it</p>
<p>Expected      slain, predicted      slais</p>
<p>Expected       thou, predicted       thou</p>
<p>Expected     gavest, predicted     gavest</p>
<p>Expected         me, predicted         my</p>
<p>Expected      thine, predicted      thine</p>
<p>Expected        not, predicted        not</p>
<p>Expected         to, predicted         to</p>
<p>Expected       give, predicted       give</p>
<p>Expected       back, predicted       bace</p>
<p>Expected      again, predicted      agais</p>
<p>Expected         as, predicted         as</p>
<p>Expected         an, predicted         as</p>
<p>Expected  unperfect, predicted  unperfect</p>
<p>Expected      actor, predicted      actor</p>
<p>Expected         on, predicted         or</p>
<p>Expected        the, predicted        the</p>
<p>Expected      stage, predicted      stage</p>
<p>Expected        who, predicted        who</p>
<p>Expected       with, predicted       with</p>
<p>Expected        his, predicted        his</p>
<p>Expected       fear, predicted       fear</p>
<p>Expected         is, predicted         it</p>
<p>Expected        put, predicted        puh</p>
<p>Expected    besides, predicted    besided</p>
<p>Expected        his, predicted        his</p>
<p>Expected       part, predicted       part</p>
<p>Expected         or, predicted         or</p>
<p>Expected       some, predicted       some</p>
<p>Expected     fierce, predicted     fierct</p>
<p>Expected      thing, predicted      thine</p>
<p>Expected    replete, predicted    replets</p>
<p>Expected       with, predicted       with</p>
<p>Expected        too, predicted        tor</p>
<p>Expected       much, predicted       much</p>
<p>Expected       rage, predicted       rage</p>
<p>Expected      whose, predicted      whose</p>
<p>Expected  strengths, predicted  strengths</p>
<p>Expected  abundance, predicted  abundance</p>
<p>Expected    weakens, predicted    weakent</p>
<p>Expected        his, predicted        his</p>
<p>Expected        own, predicted        ows</p>
<p>Expected      heart, predicted      heart</p>
<p>Expected         so, predicted         so</p>
<p>Expected          i, predicted          o</p>
<p>Expected        for, predicted        for</p>
<p>Expected       fear, predicted       fear</p>
<p>Expected         of, predicted         or</p>
<p>Expected      trust, predicted      truse</p>
<p>Expected     forget, predicted     forger</p>
<p>Expected         to, predicted         to</p>
<p>Expected        say, predicted        say</p>
<p>Expected        the, predicted        the</p>
<p>Expected    perfect, predicted    perfect</p>
<p>Expected   ceremony, predicted   ceremont</p>
<p>Expected         of, predicted         or</p>
<p>Expected      loves, predicted      loves</p>
<p>Expected       rite, predicted       rith</p>
<p>Expected        and, predicted        and</p>
<p>Expected         in, predicted         it</p>
<p>Expected       mine, predicted       ming</p>
<p>Expected        own, predicted        ows</p>
<p>Expected      loves, predicted      loves</p>
<p>Expected   strength, predicted   strengts</p>
<p>Expected       seem, predicted       seer</p>
<p>Expected         to, predicted         to</p>
<p>Expected      decay, predicted      decay</p>
<p>Expected oercharged, predicted oercharged</p>
<p>Expected       with, predicted       with</p>
<p>Expected     burden, predicted     burdey</p>
<p>Expected         of, predicted         or</p>
<p>Expected       mine, predicted       ming</p>
<p>Expected        own, predicted        ows</p>
<p>Expected      loves, predicted      loves</p>
<p>Expected      might, predicted      might</p>
<p>Expected          o, predicted          o</p>
<p>Expected        let, predicted        let</p>
<p>Expected         my, predicted         my</p>
<p>Expected      books, predicted      books</p>
<p>Expected         be, predicted         be</p>
<p>Expected       then, predicted       thee</p>
<p>Expected        the, predicted        the</p>
<p>Expected  eloquence, predicted  eloquence</p>
<p>Expected        and, predicted        and</p>
<p>Expected       dumb, predicted       dumh</p>
<p>Expected  presagers, predicted  presagers</p>
<p>Expected         of, predicted         or</p>
<p>Expected         my, predicted         my</p>
<p>Expected   speaking, predicted   speaking</p>
<p>Expected     breast, predicted     breast</p>
<p>Expected        who, predicted        who</p>
<p>Expected      plead, predicted      pleat</p>
<p>Expected        for, predicted        for</p>
<p>Expected       love, predicted       love</p>
<p>Expected        and, predicted        and</p>
<p>Expected       look, predicted       lood</p>
<p>Expected        for, predicted        for</p>
<p>Expected recompense, predicted recompenst</p>
<p>Expected       more, predicted       more</p>
<p>Expected       than, predicted       that</p>
<p>Expected       that, predicted       that</p>
<p>Expected     tongue, predicted     tongur</p>
<p>Expected       that, predicted       that</p>
<p>Expected       more, predicted       more</p>
<p>Expected       hath, predicted       hate</p>
<p>Expected       more, predicted       more</p>
<p>Expected   expressd, predicted   expresse</p>
<p>Expected          o, predicted          o</p>
<p>Expected      learn, predicted      leare</p>
<p>Expected         to, predicted         to</p>
<p>Expected       read, predicted       reaf</p>
<p>Expected       what, predicted       what</p>
<p>Expected     silent, predicted     silens</p>
<p>Expected       love, predicted       love</p>
<p>Expected       hath, predicted       hate</p>
<p>Expected       writ, predicted       writ</p>
<p>Expected         to, predicted         to</p>
<p>Expected       hear, predicted       hear</p>
<p>Expected       with, predicted       with</p>
<p>Expected       eyes, predicted       eyes</p>
<p>Expected    belongs, predicted    belonge</p>
<p>Expected         to, predicted         to</p>
<p>Expected      loves, predicted      loves</p>
<p>Expected       fine, predicted       find</p>
<p>Expected        wit, predicted        wit</p>
<p>Expected       mine, predicted       ming</p>
<p>Expected        eye, predicted        eye</p>
<p>Expected       hath, predicted       hate</p>
<p>Expected      playd, predicted      plays</p>
<p>Expected        the, predicted        the</p>
<p>Expected    painter, predicted    painted</p>
<p>Expected        and, predicted        and</p>
<p>Expected       hath, predicted       hate</p>
<p>Expected     stelld, predicted     stells</p>
<p>Expected        thy, predicted        the</p>
<p>Expected    beautys, predicted    beautys</p>
<p>Expected       form, predicted       fore</p>
<p>Expected         in, predicted         it</p>
<p>Expected      table, predicted      table</p>
<p>Expected         of, predicted         or</p>
<p>Expected         my, predicted         my</p>
<p>Expected      heart, predicted      heart</p>
<p>Expected         my, predicted         my</p>
<p>Expected       body, predicted       bods</p>
<p>Expected         is, predicted         it</p>
<p>Expected        the, predicted        the</p>
<p>Expected      frame, predicted      frame</p>
<p>Expected    wherein, predicted    whereit</p>
<p>Expected        tis, predicted        tit</p>
<p>Expected       held, predicted       held</p>
<p>Expected        and, predicted        and</p>
<p>Expected erspective, predicted erspective</p>
<p>Expected         it, predicted         it</p>
<p>Expected         is, predicted         it</p>
<p>Expected        the, predicted        the</p>
<p>Expected   painters, predicted   painters</p>
<p>Expected        art, predicted        art</p>
<p>Expected        for, predicted        for</p>
<p>Expected    through, predicted    througd</p>
<p>Expected        the, predicted        the</p>
<p>Expected    painter, predicted    painted</p>
<p>Expected       must, predicted       must</p>
<p>Expected        you, predicted        you</p>
<p>Expected        see, predicted        see</p>
<p>Expected        his, predicted        his</p>
<p>Expected      skill, predicted      skill</p>
<p>Expected         to, predicted         to</p>
<p>Expected       find, predicted       find</p>
<p>Expected      where, predicted      where</p>
<p>Expected       your, predicted       your</p>
<p>Expected       true, predicted       trut</p>
<p>Expected      image, predicted      image</p>
<p>Expected   pictured, predicted   pictures</p>
<p>Expected       lies, predicted       liee</p>
<p>Expected      which, predicted      which</p>
<p>Expected         in, predicted         it</p>
<p>Expected         my, predicted         my</p>
<p>Expected     bosoms, predicted     bosomy</p>
<p>Expected       shop, predicted       shou</p>
<p>Expected         is, predicted         it</p>
<p>Expected    hanging, predicted    hanging</p>
<p>Expected      still, predicted      still</p>
<p>Expected       that, predicted       that</p>
<p>Expected       hath, predicted       hate</p>
<p>Expected        his, predicted        his</p>
<p>Expected    windows, predicted    windows</p>
<p>Expected     glazed, predicted     glazes</p>
<p>Expected       with, predicted       with</p>
<p>Expected      thine, predicted      thine</p>
<p>Expected       eyes, predicted       eyes</p>
<p>Expected        now, predicted        not</p>
<p>Expected        see, predicted        see</p>
<p>Expected       what, predicted       what</p>
<p>Expected       good, predicted       good</p>
<p>Expected      turns, predicted      turns</p>
<p>Expected       eyes, predicted       eyes</p>
<p>Expected        for, predicted        for</p>
<p>Expected       eyes, predicted       eyes</p>
<p>Expected       have, predicted       have</p>
<p>Expected       done, predicted       dong</p>
<p>Expected       mine, predicted       ming</p>
<p>Expected       eyes, predicted       eyes</p>
<p>Expected       have, predicted       have</p>
<p>Expected      drawn, predicted      drawe</p>
<p>Expected        thy, predicted        the</p>
<p>Expected      shape, predicted      shapy</p>
<p>Expected        and, predicted        and</p>
<p>Expected      thine, predicted      thine</p>
<p>Expected        for, predicted        for</p>
<p>Expected         me, predicted         my</p>
<p>Expected        are, predicted        art</p>
<p>Expected    windows, predicted    windows</p>
<p>Expected         to, predicted         to</p>
<p>Expected         my, predicted         my</p>
<p>Expected     breast, predicted     breast</p>
<p>Expected erethrough, predicted erethrougs</p>
<p>Expected        the, predicted        the</p>
<p>Expected        sun, predicted        suh</p>
<p>Expected   delights, predicted   delights</p>
<p>Expected         to, predicted         to</p>
<p>Expected       peep, predicted       peer</p>
<p>Expected         to, predicted         to</p>
<p>Expected       gaze, predicted       gaze</p>
<p>Expected    therein, predicted    thereit</p>
<p>Expected         on, predicted         or</p>
<p>Expected       thee, predicted       thee</p>
<p>Expected        yet, predicted        yet</p>
<p>Expected       eyes, predicted       eyes</p>
<p>Expected       this, predicted       this</p>
<p>Expected    cunning, predicted    cunning</p>
<p>Expected       want, predicted       wand</p>
<p>Expected         to, predicted         to</p>
<p>Expected      grace, predicted      grace</p>
<p>Expected      their, predicted      their</p>
<p>Expected        art, predicted        art</p>
<p>Expected       they, predicted       thee</p>
<p>Expected       draw, predicted       dras</p>
<p>Expected        but, predicted        but</p>
<p>Expected       what, predicted       what</p>
<p>Expected       they, predicted       thee</p>
<p>Expected        see, predicted        see</p>
<p>Expected       know, predicted       knot</p>
<p>Expected        not, predicted        not</p>
<p>Expected        the, predicted        the</p>
<p>Expected      heart, predicted      heart</p>
<p>Expected        let, predicted        let</p>
<p>Expected      those, predicted      those</p>
<p>Expected        who, predicted        who</p>
<p>Expected        are, predicted        art</p>
<p>Expected         in, predicted         it</p>
<p>Expected     favour, predicted     favous</p>
<p>Expected       with, predicted       with</p>
<p>Expected      their, predicted      their</p>
<p>Expected      stars, predicted      stare</p>
<p>Expected         of, predicted         or</p>
<p>Expected     public, predicted     publis</p>
<p>Expected     honour, predicted     honour</p>
<p>Expected        and, predicted        and</p>
<p>Expected      proud, predicted      prous</p>
<p>Expected     titles, predicted     titles</p>
<p>Expected      boast, predicted      boass</p>
<p>Expected     whilst, predicted     whilst</p>
<p>Expected          i, predicted          o</p>
<p>Expected       whom, predicted       whom</p>
<p>Expected    fortune, predicted    fortune</p>
<p>Expected         of, predicted         or</p>
<p>Expected       such, predicted       such</p>
<p>Expected    triumph, predicted    triumps</p>
<p>Expected       bars, predicted       bare</p>
<p>Expected    unlookd, predicted    unlookd</p>
<p>Expected        for, predicted        for</p>
<p>Expected        joy, predicted        jor</p>
<p>Expected         in, predicted         it</p>
<p>Expected       that, predicted       that</p>
<p>Expected          i, predicted          o</p>
<p>Expected     honour, predicted     honour</p>
<p>Expected       most, predicted       most</p>
<p>Expected      great, predicted      greaf</p>
<p>Expected    princes, predicted    princes</p>
<p>Expected favourites, predicted favourites</p>
<p>Expected      their, predicted      their</p>
<p>Expected       fair, predicted       fair</p>
<p>Expected     leaves, predicted     leaves</p>
<p>Expected     spread, predicted     spreas</p>
<p>Expected        but, predicted        but</p>
<p>Expected         as, predicted         as</p>
<p>Expected        the, predicted        the</p>
<p>Expected   marigold, predicted   marigols</p>
<p>Expected         at, predicted         as</p>
<p>Expected        the, predicted        the</p>
<p>Expected       suns, predicted       sung</p>
<p>Expected        eye, predicted        eye</p>
<p>Expected        and, predicted        and</p>
<p>Expected         in, predicted         it</p>
<p>Expected themselves, predicted themselves</p>
<p>Expected      their, predicted      their</p>
<p>Expected      pride, predicted      pride</p>
<p>Expected       lies, predicted       liee</p>
<p>Expected     buried, predicted     buried</p>
<p>Expected        for, predicted        for</p>
<p>Expected         at, predicted         as</p>
<p>Expected          a, predicted          o</p>
<p>Expected      frown, predicted      frowd</p>
<p>Expected       they, predicted       thee</p>
<p>Expected         in, predicted         it</p>
<p>Expected      their, predicted      their</p>
<p>Expected      glory, predicted      glors</p>
<p>Expected        die, predicted        dit</p>
<p>Expected        the, predicted        the</p>
<p>Expected    painful, predicted    painfum</p>
<p>Expected    warrior, predicted    warriog</p>
<p>Expected   famoused, predicted   famouses</p>
<p>Expected        for, predicted        for</p>
<p>Expected      fight, predicted      fight</p>
<p>Expected      after, predicted      afted</p>
<p>Expected          a, predicted          o</p>
<p>Expected   thousand, predicted   thousand</p>
<p>Expected  victories, predicted  victorier</p>
<p>Expected       once, predicted       once</p>
<p>Expected      foild, predicted      foild</p>
<p>Expected         is, predicted         it</p>
<p>Expected       from, predicted       from</p>
<p>Expected        the, predicted        the</p>
<p>Expected       book, predicted       boor</p>
<p>Expected         of, predicted         or</p>
<p>Expected     honour, predicted     honour</p>
<p>Expected      razed, predicted      razes</p>
<p>Expected      quite, predicted      quite</p>
<p>Expected        and, predicted        and</p>
<p>Expected        all, predicted        all</p>
<p>Expected        the, predicted        the</p>
<p>Expected       rest, predicted       rest</p>
<p>Expected     forgot, predicted     forgot</p>
<p>Expected        for, predicted        for</p>
<p>Expected      which, predicted      which</p>
<p>Expected         he, predicted         he</p>
<p>Expected      toild, predicted      toild</p>
<p>Expected       then, predicted       thee</p>
<p>Expected      happy, predicted      happy</p>
<p>Expected          i, predicted          o</p>
<p>Expected       that, predicted       that</p>
<p>Expected       love, predicted       love</p>
<p>Expected        and, predicted        and</p>
<p>Expected         am, predicted         as</p>
<p>Expected    beloved, predicted    beloved</p>
<p>Expected      where, predicted      where</p>
<p>Expected          i, predicted          o</p>
<p>Expected        may, predicted        may</p>
<p>Expected        not, predicted        not</p>
<p>Expected     remove, predicted     remove</p>
<p>Expected        nor, predicted        not</p>
<p>Expected         be, predicted         be</p>
<p>Expected    removed, predicted    removes</p>
<p>Testing accuracy: 0.6265060240963856</p>
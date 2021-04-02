---
layout: single
title:  "Multinomial Naive Bayes for Sentiment Analysis and Fake Review Detection"
date:   2021-03-30
categories: TextMining
tags: Python MultinomialNB SentimentAnalysis FakeReviewDetection CrossValidation
toc: true
toc_label: "Table of Contents"
toc_icon: "clone"
---


We often see text from the internet automatically classified as positive, negative and in websites like Amazon, they automatically track fake reviews and remove them proactively to prevent bias for their text mining model. Althought they use complex and huge datasets for training process, the goal in this one is to analyse text data and interpret the models, find patterns in determining wrongly predicted text.

The dataset is provided by Prof. Bei Yu at Syracuse University. It is about a fake restaurant review data of true and fake reviews. Some were positive, and some were negative. It is completely fictional. It has columns 'review', 'lie', and 'sentiment' which are straight forward to understand. With this data, the goal is to create a cross validated Multinomial Naive Bayes Model trained on reviews, sentiment for **Sentiment Analysis** and reviews, lie for **Fake Review Detection**

#### Sentiment Analysis
##### First step is to import required libraries


```python
# Importing libaries

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
```


```python
# Viewing sample data
# it is assumed that data is present in the relative path of this file

train_data = pd.read_csv('deception_data_converted_final.tsv', delimiter='\t')
train_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lie</th>
      <th>sentiment</th>
      <th>review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>f</td>
      <td>n</td>
      <td>'Mike\'s Pizza High Point, NY Service was very...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>f</td>
      <td>n</td>
      <td>'i really like this buffet restaurant in Marsh...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>f</td>
      <td>n</td>
      <td>'After I went shopping with some of my friend,...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>f</td>
      <td>n</td>
      <td>'Olive Oil Garden was very disappointing. I ex...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>f</td>
      <td>n</td>
      <td>'The Seven Heaven restaurant was never known f...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Vectorizer options

unigram_bool_vectorizer = CountVectorizer(encoding='latin-1',
                                          binary=True,
                                          min_df=5,
                                          stop_words='english')

unigram_count_vectorizer = CountVectorizer(encoding='latin-1',
                                           binary=False,
                                           min_df=5,
                                           stop_words='english')

gram12_count_vectorizer = CountVectorizer(encoding='latin-1',
                                          ngram_range=(1, 2),
                                          min_df=5,
                                          stop_words='english')

unigram_tfidf_vectorizer = TfidfVectorizer(encoding='latin-1',
                                           use_idf=True,
                                           min_df=5,
                                           stop_words='english')


# Defining model for taking vectorizer options and print the cross validation metrics
def print_CV_MNB_results(train_data, vectorizers_list, train_column,
                         test_column, k_for_CV, labels):
    """This function prints the confusion matrix, accuracy and the classification report of
    cross validation MultinomialNB model"""
    global X
    X = train_data[train_column].values
    global y
    y = train_data[test_column].values

    global nb_clf
    nb_clf = MultinomialNB()

    for vec in vectorizers_list:

        X_train_data_vec = vec.fit_transform(X)
        cv_pred = cross_val_predict(estimator=nb_clf,
                                    X=X_train_data_vec,
                                    y=y,
                                    cv=k_for_CV)
        print(f'=== Results using {vec} as vectorizer ===\n')
        print('=== Confusion Matrix ===')
        print(confusion_matrix(cv_pred, y, labels=labels))
        print('========================')
        print(f'Accuracy: {round(accuracy_score(cv_pred, y), 3)}')
        print('========================')
        print('Classification Report')
        print(classification_report(cv_pred, y, target_names=labels))
        print(
            '====================================================================================='
        )
```


```python
# creating a list of vectorizer types that we want to test
vectorizer_list = [
    unigram_bool_vectorizer,
    unigram_count_vectorizer,
    gram12_count_vectorizer,
    unigram_tfidf_vectorizer
]

# Showing results for each vectorizer and model metrics
print_CV_MNB_results(train_data, vectorizer_list, 'review', 'sentiment', 5,
                     ['n', 'p'])
```

    === Results using CountVectorizer(binary=True, encoding='latin-1', min_df=5, stop_words='english') as vectorizer ===

    === Confusion Matrix ===
    [[32  6]
     [14 40]]
    ========================
    Accuracy: 0.783
    ========================
    Classification Report
                  precision    recall  f1-score   support

               n       0.70      0.84      0.76        38
               p       0.87      0.74      0.80        54

        accuracy                           0.78        92
       macro avg       0.78      0.79      0.78        92
    weighted avg       0.80      0.78      0.78        92

    =====================================================================================
    === Results using CountVectorizer(encoding='latin-1', min_df=5, stop_words='english') as vectorizer ===

    === Confusion Matrix ===
    [[33  5]
     [13 41]]
    ========================
    Accuracy: 0.804
    ========================
    Classification Report
                  precision    recall  f1-score   support

               n       0.72      0.87      0.79        38
               p       0.89      0.76      0.82        54

        accuracy                           0.80        92
       macro avg       0.80      0.81      0.80        92
    weighted avg       0.82      0.80      0.81        92

    =====================================================================================
    === Results using CountVectorizer(encoding='latin-1', min_df=5, ngram_range=(1, 2),
                    stop_words='english') as vectorizer ===

    === Confusion Matrix ===
    [[33  5]
     [13 41]]
    ========================
    Accuracy: 0.804
    ========================
    Classification Report
                  precision    recall  f1-score   support

               n       0.72      0.87      0.79        38
               p       0.89      0.76      0.82        54

        accuracy                           0.80        92
       macro avg       0.80      0.81      0.80        92
    weighted avg       0.82      0.80      0.81        92

    =====================================================================================
    === Results using TfidfVectorizer(encoding='latin-1', min_df=5, stop_words='english') as vectorizer ===

    === Confusion Matrix ===
    [[33  6]
     [13 40]]
    ========================
    Accuracy: 0.793
    ========================
    Classification Report
                  precision    recall  f1-score   support

               n       0.72      0.85      0.78        39
               p       0.87      0.75      0.81        53

        accuracy                           0.79        92
       macro avg       0.79      0.80      0.79        92
    weighted avg       0.81      0.79      0.79        92

    =====================================================================================


The Best Model for sentiment analysis task is the second one. The second and third, both have same accuracy and also other metrics. The difference between the vectorizers is one uses only unigrams and their counts for word representation while the latter one uses both unigrams and bigrams. Computationally speaking, model 2 with only unigram vectors is the best choice because of its less computation time for the task.

##### Model Interpretation (Sentiment Analysis)

Out of the four Vectorizers, the second and third have produced accuracy of 80%. So, if we analyse the results the model produce on the test set. If we understand properly, the Naive Bayes classifier is trained similarly for all the vectorizers. Only difference is the vector representation of the words. So, we need to understand which representation works better.


```python
# Viewing the predictions
# 'n' is for 0 and 'p' is for 1

nb_clf = MultinomialNB()
vec = unigram_count_vectorizer
X_train = train_data['review']
y_train = train_data['sentiment']
X_train_data_vec = vec.fit_transform(X_train)
nb_clf_cv = cross_validate(estimator=nb_clf,
                                    X=X_train_data_vec,
                                    y=y_train,
                                    cv=5, return_estimator=True)
# nb_clf_cv['estimator']
print('Test Scores of Cross Validation')
test_scores = nb_clf_cv['test_score']
print(test_scores, '\n')

print(f'Model {test_scores.argmax()+1} is used for prediction \n')
cv_pred = nb_clf_cv['estimator'][test_scores.argmax()].predict(X_train_data_vec)
print('Predictions')
print(np.array(cv_pred[10:20]))
print('Actual Values')
print(np.array(y_train)[10:20])
```

    Test Scores of Cross Validation
    [0.84210526 0.89473684 0.72222222 0.83333333 0.72222222]

    Model 2 is used for prediction

    Predictions
    ['n' 'p' 'n' 'n' 'n' 'n' 'p' 'n' 'n' 'n']
    Actual Values
    ['n' 'n' 'n' 'n' 'n' 'n' 'n' 'n' 'n' 'n']


Only two out of between 10 predictions are different than the original. This is not bad. So now we will see the wrongly predicted ones and think of why that would have happened.

##### Error Analysis


```python
err_cnt = 0
print('***************** Predicted is \'negative\' but acutal is \'positive\' ********************')
for i in range(0, len(X_train)):
    if(y_train[i]=='p' and cv_pred[i]=='n'):
        print(X_train[i])
        err_cnt = err_cnt+1

print()
print("Type 2 errors:", err_cnt)

print('***************************************************************************************')

print()

print('***************** Predicted is \'positive\' but acutal is \'negative\' ********************')
err_cnt = 0
for i in range(0, len(X_train)):
    if(y_train[i]=='n' and cv_pred[i]=='p'):
        print(X_train[i])
        err_cnt = err_cnt+1
print()
print("Type 1 errors:", err_cnt)
print('***************************************************************************************')
```

    ***************** Predicted is 'negative' but acutal is 'positive' ********************
    'Ruby Tuesday is my favorite America Style Restaurant. The salad is awesome. And I like the baby pork ribs so much . So does the coconut shrimp.'
    ?
    ?
    'Can\'t say too much about it. Just, try it buddy!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!You\'ll regret if you don\'t.'

    Type 2 errors: 4
    ***************************************************************************************

    ***************** Predicted is 'positive' but acutal is 'negative' ********************
    'Mike\'s Pizza High Point, NY Service was very slow and the quality was low. You would think they would know at least how to make good pizza, not. Stick to pre-made dishes like stuffed pasta or a salad. You should consider dining else where.'
    'i really like this buffet restaurant in Marshall street. they have a lot of selection of american, japanese, and chinese dishes. we also got a free drink and free refill. there are also different kinds of dessert. the staff is very friendly. it is also quite cheap compared with the other restaurant in syracuse area. i will definitely coming back here.'
    'I once went to Chipotle at Marshall Street to have my dinner. The experience is horrible!!! When I began to order, I found that there were no steak and chicken! But I want to have the steak burrito! How can a Chipotle have no steak and chicken! What\'s worse, there is no sauce! No any kind of sauce! I can\'t believe my eyes! I will never go to the Chipotle at Marshall Street with my little friends any more!'
    'I went there with two friends at 6pm. Long queue was there. But it didn\'t take us long to wait. The waiter was nice but worked in a hurry. We ordered \'Today\'s Special\', some drinks and two icecreams. I had a steak, a little bit too salty, but acceptable. My friends didn\'t like their lamb chop and cod filet that much. It costed us almost $100. Not worth it. Will not visit there any more.'
    'Pizza Hut Syracuse, NY The only thing worth going here for is the lunch salad bar. The decor is very dated and the pizza is GREESY. Tables and bathroom are dirty. Waitstaff seem to have low expectations of service.'
    'Last week, I went o my favorite Indian \'Thali\' place in Jersey City. I have been there before and have loved the variety of Indian offerings they have in their buffet. However, last time, the number of delicacies were limited and to top it up, they were stale. It felt like they used the same items from the previous day. I was utterly disappointed by the quality of food at one of my favorite Indian restaurant in Jersey City.'
    'This place used to be great. I can\'t believe it\'s current state. Instead of the cool, dimly-lit lounge that I was used to, I was in a cheap, smelly bar. The music has no soul, the bartender is mean. This place no longer exudes a welcoming spirit. The crowd is awkward and old. I want my old hangout back!!'
    'I have been to a Asian restaurant in New York city. The menu is written by Chinese and English. When I choose a famous chinses plate called Gongbao chicken, I was surprised. The taste of it like a Thai flavor, which is cooked by curry. '

    Type 1 errors: 8
    ***************************************************************************************


In the above, we can see the two different types of errors. It can be thought like sometimes when there are new words in the comment, then we can say model will tend to choose wrong category. When more words appear in one category then it is related that people use those in a particular emotion. Now, we need to see which features in the model are the top for making prediction decision

##### Top Features contributing to the classes


```python
# Top features


log_prob = nb_clf_cv['estimator'][test_scores.argmax()].feature_log_prob_
for i in range(len(log_prob)):
    if i == 0:
        print('Coeffients of features sorted in the descending order for negative class')
    else:
        print('Coeffients of features sorted in the descending order for positive class')
    sorted_log_prob_10 = np.argsort(log_prob[i])[-10:]
    print(log_prob[i][sorted_log_prob_10])

    print()
    if i == 0:
        print('====== Top features for negative sentiment category ======')
    else:
        print('====== Top features for positive sentiment category ======')
    for x in sorted_log_prob_10:
        print(list(vec.vocabulary_.items())[x])
    print()
```

    Coeffients of features sorted in the descending order for negative class
    [-4.05415368 -4.05415368 -3.8870996  -3.8870996  -3.67946023 -3.67946023
     -3.45631668 -3.27399512 -2.84113104 -2.84113104]

    ====== Top features for negative sentiment category ======
    ('people', 64)
    ('special', 81)
    ('favorite', 32)
    ('ordered', 61)
    ('wasn', 94)
    ('fresh', 34)
    ('ask', 3)
    ('want', 93)
    ('order', 60)
    ('taste', 84)

    Coeffients of features sorted in the descending order for positive class
    [-3.94931879 -3.94931879 -3.85400861 -3.85400861 -3.76699723 -3.61284655
     -3.3074649  -3.16086143 -2.78616798 -2.75539632]

    ====== Top features for positive sentiment category ======
    ('family', 31)
    ('special', 81)
    ('lunch', 51)
    ('want', 93)
    ('high', 41)
    ('favorite', 32)
    ('place', 66)
    ('pasta', 63)
    ('order', 60)
    ('taste', 84)



The downside of the top features is that both classs have a few common features that contribute for making predictions.

#### Fake Review Detection


```python
# creating a list of vectorizer types that we want to test
vectorizer_list = [
    unigram_bool_vectorizer, unigram_count_vectorizer, gram12_count_vectorizer,
    unigram_tfidf_vectorizer
]

# Showing results for each vectorizer and model metrics

print_CV_MNB_results(train_data, vectorizer_list, 'review', 'lie', 5,
                     ['f', 't'])
```

    === Results using CountVectorizer(binary=True, encoding='latin-1', min_df=5, stop_words='english') as vectorizer ===

    === Confusion Matrix ===
    [[27 19]
     [19 27]]
    ========================
    Accuracy: 0.587
    ========================
    Classification Report
                  precision    recall  f1-score   support

               f       0.59      0.59      0.59        46
               t       0.59      0.59      0.59        46

        accuracy                           0.59        92
       macro avg       0.59      0.59      0.59        92
    weighted avg       0.59      0.59      0.59        92

    =====================================================================================
    === Results using CountVectorizer(encoding='latin-1', min_df=5, stop_words='english') as vectorizer ===

    === Confusion Matrix ===
    [[26 19]
     [20 27]]
    ========================
    Accuracy: 0.576
    ========================
    Classification Report
                  precision    recall  f1-score   support

               f       0.57      0.58      0.57        45
               t       0.59      0.57      0.58        47

        accuracy                           0.58        92
       macro avg       0.58      0.58      0.58        92
    weighted avg       0.58      0.58      0.58        92

    =====================================================================================
    === Results using CountVectorizer(encoding='latin-1', min_df=5, ngram_range=(1, 2),
                    stop_words='english') as vectorizer ===

    === Confusion Matrix ===
    [[27 17]
     [19 29]]
    ========================
    Accuracy: 0.609
    ========================
    Classification Report
                  precision    recall  f1-score   support

               f       0.59      0.61      0.60        44
               t       0.63      0.60      0.62        48

        accuracy                           0.61        92
       macro avg       0.61      0.61      0.61        92
    weighted avg       0.61      0.61      0.61        92

    =====================================================================================
    === Results using TfidfVectorizer(encoding='latin-1', min_df=5, stop_words='english') as vectorizer ===

    === Confusion Matrix ===
    [[26 20]
     [20 26]]
    ========================
    Accuracy: 0.565
    ========================
    Classification Report
                  precision    recall  f1-score   support

               f       0.57      0.57      0.57        46
               t       0.57      0.57      0.57        46

        accuracy                           0.57        92
       macro avg       0.57      0.57      0.57        92
    weighted avg       0.57      0.57      0.57        92

    =====================================================================================


The best model for fake review detection is the third one that uses both unigrams and bigrams. The accuracy reported is almost 61%. We should understand that one model is not suitable for all and also one type of word representation is not suitable for all types of data. Depending on the type of data available the preprocessing of the words changes, and, in this case, bigrams are also important to perform fake review detection. We often see a pattern of words in fake reviews. Hence, maybe it is because bigrams and trigrams may also become useful to predict the review authenticity.

##### Model Interpretation (Fake Review Detection)


```python
# Viewing the predictions
# 'f' is for 0 and 't' is for 1

nb_clf = MultinomialNB()
vec = gram12_count_vectorizer
X_train = train_data['review']
y_train = train_data['lie']
X_train_data_vec = vec.fit_transform(X_train)
nb_clf_cv = cross_validate(estimator=nb_clf,
                                    X=X_train_data_vec,
                                    y=y_train,
                                    cv=5, return_estimator=True)
# nb_clf_cv['estimator']
print('Test Scores of Cross Validation')
test_scores = nb_clf_cv['test_score']
print(test_scores, '\n')

print(f'Model {test_scores.argmax()+1} is used for prediction \n')
cv_pred = nb_clf_cv['estimator'][test_scores.argmax()].predict(X_train_data_vec)
print('Predictions')
print(np.array(cv_pred[10:20]))
print('Actual Values')
print(np.array(y_train)[10:20])
```

    Test Scores of Cross Validation
    [0.57894737 0.73684211 0.55555556 0.61111111 0.55555556]

    Model 2 is used for prediction

    Predictions
    ['f' 'f' 't' 'f' 'f' 't' 'f' 'f' 'f' 'f']
    Actual Values
    ['f' 'f' 'f' 'f' 'f' 'f' 'f' 'f' 'f' 'f']


Only one out of between 10 predictions are different than the original. This is not bad. So now we will see the wrongly predicted ones and think of why that would have happened.

##### Error Analysis


```python
err_cnt = 0
print('************ Predicted is \'True\' but acutal is \'False\' for fake review? question ***********')
for i in range(0, len(X_train)):
    if(y_train[i]=='f' and cv_pred[i]=='t'):
        print(X_train[i])
        err_cnt = err_cnt+1

print()
print("Type 2 errors:", err_cnt)

print('***************************************************************************************')

print()

print('************ Predicted is \'False\' but acutal is \'True\' for fake review? question ***********')
err_cnt = 0
for i in range(0, len(X_train)):
    if(y_train[i]=='t' and cv_pred[i]=='f'):
        print(X_train[i])
        err_cnt = err_cnt+1

print()
print("Type 1 errors:", err_cnt)
print('***************************************************************************************')
```

    ************ Predicted is 'True' but acutal is 'False' for fake review? question ***********
    'Yesterday, I went to a casino-restaurant called \'NoFreeDrinks\'. First, I thought its just a name but later at the blackjack table, when I ordered a attendant to get me a round of Jack and Coke, instead of saying \'Yeah sure\', the attendant said $6. It is preposterous. I have never paid for my drinks at the casino. Clearly, the casino does not understand that the more the people drink, the more they will loose. Someone needs to teach the casino how to do business.'
    'I entered the restaurant and a waitress came by with a blanking looking and threw the menu on the table, said coldly, help yourself. Then she just disappeared, I waited and waited, but no one even notice me until I went directly to the front desk to order the food. Long time later, I finally had the most terrible food in my life and even found an flyer in my plate. I refused to give the tips and the waitress began to get angry and rudely walk away. This is the most terrible experience that I will never forget. '
    'In each of the diner dish there are at least one fly in it. We are waiting for an hour for the dish cooked done. The taste reminds me of a smell that I never want to try any more, and when the food is in the mouth, it could be a nightmare that I really want to wake up. The attitude of waiters is bad enough that I don\'t want to step in to this restaurant again.'
    'In my favorite restaurant Yuenan Restaurant. The noodle with beef is the best noodle in this area. I love it so much and so do my friends.'
    'This restaurant is AMAZING! I felt like life was worth living again when I bit into the carmelized onion tofurky extravaganzaburger. I highly reccommend going here for delicious food, fun atmosphere, and decent prices. Vegetarian cusine=really good.'
    'I went to this ultra-luxurious restaurant in Downtown New York which is known for its exotic and expensive cuisine. I had a glass of champagne along with very expensive Caviar. I had a delicious Chicken Pasta cooked in white sauce. This was followed by mouth melting chocolate brownie and vanilla ice cream. The service standards were superlative and I felt special visiting this restaurant. '
    'This restaurant ROCKS! I mean the food is great and people are great. Everything is great great just great!!! I love it. I like it. '
    'Two days ago, I went to the rooftop restaurant in NYC that served brunch. it was one of the best brunch that I have ever had. The view from the table was serene and I could see both the the Hudson River and the East River with outstanding views of Empire State Building, the Chryslers tower, Freedom tower and the Central park. A great place with great food and a perplexing view'
    'I went into the restaurant, it decorated comfortably with a soft light and nice pictures, the waitress was kind and stand by my side throughout the whole dining time, asking whether I need something more and kept smiling. '
    'The service is good and I just felt like home. Waitresses and waiters always ask me want to I need and how about the taste, in order to bring m ore good menu and service for us to enjoy. And there is also music from the lobby, someones are dancing in the middle.'

    Type 2 errors: 10
    ***************************************************************************************

    ************ Predicted is 'False' but acutal is 'True' for fake review? question ***********
    'Friday is the worse restaurant I have ever gone. Each of the dishes we ordered is quite terrible. We did not finish any of them.'
    'The restaurant environment is bad and I can see flies everywhere. The dishes are not that bad to taste, neither not that good. We discovered two flies in the dishes which I really cannot bear.'
    'This restaurant is quite popular recently. Went there with two of my friends at 6pm, really long queue. We waited for almost 90 minutes to be seated. Seats were narrow. It was too easy to hear clearly what your neighbors were talking about. The waitress was so impatient that we were wondering whether she didn\'t get paid. Food was so so. No idea why it\'s so popular. Never be back again.'
    'We indians are craving Indian food around the university campus. The first time I went to Samrat, i was very excited to taste all the dishes that were served in the buffet. As i started eating I found every was somehow tasting similar. The Bengan Bharta was sweet and gulab jaam were canned and not made freshly. To add to it the food was made with baking soda so that it filled our stomach quickly. '
    'Stronghearts cafe is the BEST! The owners have a great ethic, and the food is TO DIE FOR. I had a pumpkin espresso milkshake, named after Albert Einstein, and it was only $5! #winning The food, though vegan, is amazing because of the fresh ingredients and presentation. Speed of service is great too. They have reading material, wifi, and many tables of different types depending on whether you are with friends or by yourself doing homework. YEAH!'
    ?
    ?
    'Can\'t say too much about it. Just, try it buddy!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!You\'ll regret if you don\'t.'
    'My sister and I ate at this restaurant called Matador. The overall look and ambiance of the restaurant was very appealing. We first ordered strawberry margaritas--which were really good.Then my sister ordered a spinach lasagna with Alfredo sauce and I ordered Pasta ravioli with marinara sauce. My sister and I unanimously agreed they were the best pastas we had ever had. It was a beautiful blend of flavors which complimented each other. I would totally recommend Matador and it was an overall amazing experience.'

    Type 1 errors: 9
    ***************************************************************************************


If you see the pattern in the fake reviews, they are slightly short in the length of the review and
irregular usage of characters, there only the reader knows they are fake reviews because the story or the emotion in the sentences do not connect and can be thought they are quite imaginary. One solution can be to have data, because more text with the similar pattern can improve the model.

##### Top Features contributing to the classes


```python
# Top features

log_prob = nb_clf_cv['estimator'][test_scores.argmax()].feature_log_prob_
for i in range(len(log_prob)):
    if i == 0:
        print('Coeffients of features sorted in the descending order for not a fake review class')
    else:
        print('Coeffients of features sorted in the descending order for fake review class')
    sorted_log_prob_10 = np.argsort(log_prob[i])[-10:]
    print(log_prob[i][sorted_log_prob_10])

    print()
    if i == 0:
        print('====== Top features for not a fake review category ======')
    else:
        print('====== Top features for fake review category ======')
    for x in sorted_log_prob_10:
        print(list(vec.vocabulary_.items())[x])
    print()
```

    Coeffients of features sorted in the descending order for not a fake review class
    [-3.952845   -3.87280229 -3.87280229 -3.87280229 -3.79869432 -3.72970145
     -3.66516293 -3.49331267 -3.00376445 -2.94124409]

    ====== Top features for not a fake review category ======
    ('friends', 38)
    ('worst', 98)
    ('hard', 42)
    ('hour', 45)
    ('sauce', 79)
    ('ask', 4)
    ('special', 83)
    ('salad', 78)
    ('taste', 86)
    ('bad', 7)

    Coeffients of features sorted in the descending order for fake review class
    [-4.09434456 -4.09434456 -3.80666249 -3.80666249 -3.72661978 -3.65251181
     -3.51898042 -3.4583558  -2.82583324 -2.7080502 ]

    ====== Top features for fake review category ======
    ('special', 83)
    ('dish', 28)
    ('hour', 45)
    ('ask', 4)
    ('friends', 38)
    ('salad', 78)
    ('sauce', 79)
    ('dine', 24)
    ('bad', 7)
    ('taste', 86)



In this model too, there is an overlap of true and fake review top classes, and the fake reviews are near real to the true fake review and our model is doing its best in identifying fake reviews inspite of them being fake reviews.


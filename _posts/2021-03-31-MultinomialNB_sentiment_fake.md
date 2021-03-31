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

The dataset is provided by Prof. Bei Yu at Syracuse University. It is about a fake restaurant review data of true and fake reviews. Some were positive, and some were negative. It is completely fictional. It has columns 'review', 'lie', and 'sentiment' which are straight forward to understand. With this data, the goal is to create a cross validated Multinomial Naive Bayes Model trained on reviews and sentiment for **Sentiment Analysis** and reviews, lie for **Fake Review Detection**

## Sentiment Analysis
### First step is to import required libraries


```python
# Importing libaries

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_predict

train_data = pd.read_csv('deception_data_converted_final.tsv',
                         delimiter='\t')
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

unigram_bool_vectorizer = CountVectorizer(
    encoding='latin-1',
    binary=True,
    min_df=5,
    stop_words='english')

unigram_count_vectorizer = CountVectorizer(
    encoding='latin-1',
    binary=False,
    min_df=5,
    stop_words='english')

gram12_count_vectorizer = CountVectorizer(
    encoding='latin-1',
    ngram_range=(1, 2),
    min_df=5,
    stop_words='english')

unigram_tfidf_vectorizer = TfidfVectorizer(
    encoding='latin-1',
    use_idf=True,
    min_df=5,
    stop_words='english')

# Defining model for taking vectorizer options and print the cross validation metrics
def print_CV_MNB_results(train_data, vectorizers_list, train_column, test_column, k_for_CV, labels):
    """This function prints the confusion matrix, accuracy and the classification report of
    cross validation MultinomialNB model"""
    global X
    X = train_data[train_column].values
    global y
    y = train_data[test_column].values

    nb_clf = MultinomialNB()

    for vec in vectorizers_list:

        X_train_data_vec = vec.fit_transform(X)
        cv_pred = cross_val_predict(
            estimator=nb_clf,
            X=X_train_data_vec,
            y=y,
            cv=k_for_CV)
        print(f'=== Results using {vec} as vectorizer ===\n')
        print('=== Confusion Matrix ===')
        print(confusion_matrix(cv_pred, y, labels=labels))
        print('========================')
        print(f'Accuracy: {round(accuracy_score(cv_pred, y), 2)}')
        print('========================')
        print('Classification Report')
        print(classification_report(cv_pred, y, target_names=labels))
        print('=====================================================================================')
```


```python
# creating a list of vectorizer types that we want to test
vectorizer_list = [unigram_bool_vectorizer,
                   unigram_count_vectorizer,
                   gram12_count_vectorizer,
                   unigram_tfidf_vectorizer]

# Showing results for each vectorizer and model metrics
print_CV_MNB_results(train_data,
                     vectorizer_list,
                     'review',
                     'sentiment',
                     5,
                     ['n', 'p'])
```

    === Results using CountVectorizer(binary=True, encoding='latin-1', min_df=5, stop_words='english') as vectorizer ===

    === Confusion Matrix ===
    [[32  6]
     [14 40]]
    ========================
    Accuracy: 0.78
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
    Accuracy: 0.8
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
    Accuracy: 0.8
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
    Accuracy: 0.79
    ========================
    Classification Report
                  precision    recall  f1-score   support

               n       0.72      0.85      0.78        39
               p       0.87      0.75      0.81        53

        accuracy                           0.79        92
       macro avg       0.79      0.80      0.79        92
    weighted avg       0.81      0.79      0.79        92

    =====================================================================================


### Model Interpretation (Sentiment Analysis)


## Fake Review Detection


```python
# creating a list of vectorizer types that we want to test
vectorizer_list = [unigram_bool_vectorizer,
                   unigram_count_vectorizer,
                   gram12_count_vectorizer,
                   unigram_tfidf_vectorizer]

# Showing results for each vectorizer and model metrics

print_CV_MNB_results(train_data,
                     vectorizer_list,
                     'review',
                     'lie',
                     5,
                     ['f', 't'])
```

    === Results using CountVectorizer(binary=True, encoding='latin-1', min_df=5, stop_words='english') as vectorizer ===

    === Confusion Matrix ===
    [[27 19]
     [19 27]]
    ========================
    Accuracy: 0.59
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
    Accuracy: 0.58
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
    Accuracy: 0.61
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
    Accuracy: 0.57
    ========================
    Classification Report
                  precision    recall  f1-score   support

               f       0.57      0.57      0.57        46
               t       0.57      0.57      0.57        46

        accuracy                           0.57        92
       macro avg       0.57      0.57      0.57        92
    weighted avg       0.57      0.57      0.57        92

    =====================================================================================


### Model Interpretation (Fake Review Detection)


```python

```

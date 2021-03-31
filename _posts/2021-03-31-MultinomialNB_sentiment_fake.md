---
layout: single
title:  "Multinomial Naive Bayes for Sentiment and fake review detection"
date:   2021-03-30
categories: TextMining
tags: Python MultinomialNB SentimentAnalysis FakeReviewDetection CrossValidation
toc: true
toc_label: "Table of Contents"
toc_icon: "clone"
---

### Sentiment Detection

```python
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_predict

train_data = pd.read_csv('deception_data_converted_final.tsv', delimiter='\t')

# Vectorizer options

unigram_bool_vectorizer = CountVectorizer(
    encoding='latin-1', binary=True, min_df=5, stop_words='english')

unigram_count_vectorizer = CountVectorizer(
    encoding='latin-1', binary=False, min_df=5, stop_words='english')

gram12_count_vectorizer = CountVectorizer(
    encoding='latin-1', ngram_range=(1, 2), min_df=5, stop_words='english')

unigram_tfidf_vectorizer = TfidfVectorizer(
    encoding='latin-1', use_idf=True, min_df=5, stop_words='english')

def print_CV_MNB_results(train_data, vectorizers_list, train_column, test_column, k_for_CV, labels):

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
vectorizer_list = [unigram_bool_vectorizer,
                   unigram_count_vectorizer,
                   gram12_count_vectorizer,
                   unigram_tfidf_vectorizer]

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


### Fake Review Detection

```python
vectorizer_list = [unigram_bool_vectorizer,
                   unigram_count_vectorizer,
                   gram12_count_vectorizer,
                   unigram_tfidf_vectorizer]

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



```python

```

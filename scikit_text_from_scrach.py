# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 04:22:21 2017

@author: Jyoti Ranjan Panda
"""


import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

X_train = np.array(["new york is a hell of a town",
                    "new york \"was\" originally dutch",
                    "the big apple is great",
                    "new york is also called the big apple",
                    "nyc is nice",
                    "people abbreviate new york city as nyc",
                    "the capital of great britain is london",
                    "london is in the uk",
                    "london is in england",
                    "london is in great britain",
                    "it rains a lot in london",
                    "london hosts the british museum",
                    "new york is great and so is london",
                    "i like london better than new york"])

#X_train_target = np.array(["new york","new york","new york","new york",
#                "new york","new york","london","london",
#                "london","london","london","london",
#                "new york","new york"])

X_train_target = np.array(["0","0","0","0",
                "0","0","1","1",
                "1","1","1","1",
                "0","0"])


X_test = np.array(['nice day in nyc',
                   'welcome to london',
                   'london is rainy',
                   'it is raining in britian',
                   'it is raining in britian and the big apple',
                   'it is raining in britian and nyc',
                   'hello welcome to new york. enjoy it here and london too'])
target_names = ['0', '1']

print(X_train_target.shape)
print(X_train.shape)
classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())])

clf = classifier.fit(X_train, X_train_target)
predicted = classifier.predict(X_test)

for doc, category in zip(X_test, predicted):
    print('{0} => {1}'.format(doc, ', '.join(category)))
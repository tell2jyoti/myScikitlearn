# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 05:44:44 2017

@author: Jyoti Ranjan Panda
"""
import numpy as np
import pandas as pd
data = np.genfromtxt('C:\\Users\\610776\\Downloads\\trainingtext.csv',delimiter=',', dtype=None,skip_header=True)
features = data[:, :3]
targets = data[:, 1]


df=pd.read_csv('C:\\Users\\610776\\Downloads\\trainingtext.csv', sep=',',names=['feature', 'target'])
print(df.feature[0])

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(df.feature)
print(X_train_counts.shape)

from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
print(X_train_tf.shape)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, df.target)

docs_new = ['london is rainy', 'it is raining in new york and the big apple']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)
for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, category))

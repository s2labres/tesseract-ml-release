# -*- coding: utf-8 -*-

"""
feature-reduction.py
~~~~~~~~~~~~~

A module for performing dimensionality reduction.

"""
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE
import numpy as np
# TODO: separate train/test data before dimensionality reduction

def selectKBest(X, y, feature_size=10000):
    clf = LinearSVC()
    clf.fit(X, y)
    coef = clf.coef_

    select_index = np.argpartition(abs(coef[0]), -feature_size)[-feature_size:]

    return select_index


def recursiveFeatureElimination(X, y, feature_size=10000, step=0.1):
    clf = LinearSVC()
    selector = RFE(clf, n_features_to_select=feature_size, step=0.1)
    selector.fit(X, y)

    return selector.support_


def reduce(X, select_index):
    return X[:, select_index]


def devectorize_reduce(X, select_index, vectorizer):
    feature_names = vectorizer.feature_names_
    vectorizer.feature_names_ = np.array(feature_names)[select_index].tolist()

    vocabulary = {}
    for i, name in enumerate(vectorizer.feature_names_):
        vocabulary[name] = i
    vectorizer.vocabulary_ = vocabulary

    reduced_X = reduce(X, select_index)
    reduced_X_dict = vectorizer.inverse_transform(reduced_X)

    return reduced_X_dict

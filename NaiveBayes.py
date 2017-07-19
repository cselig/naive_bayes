# Author: Christian Selig

import numpy as np
import time
from pprint import pprint


# Naive Bayes classifier
# Works with boolean and continuous numerical features. 
# Note: boolean features should be represented as 0 or 1
class NB():

    # params:
    #   classes: list of classes 
    #   smoothing: whether or not the algorithm uses Laplace Smoothing
    #   reporting: report times for training and classification
    #   buckets: number of buckets to split continuous features into
    def __init__(self, classes, smoothing=True, reporting=True, buckets=5):
        self.classes = classes
        self.smoothing = smoothing
        self.reporting = reporting
        self.buckets = buckets


    # discretize so that all features are either 0 or 1. Turn continuous features into multiple 
    # features for each bucket
    # params: 
    #   data: list or array of feature vectors
    def preprocess(self, data):
        # ranges is a 1D array. holds 'd' if feature is discrete, and upper bound for that bucket if 
        # feature is continuous
        ranges = self.create_ranges(data)
        result = self.discretize_features(data, ranges)
        return result


    # take non-discrete feature vectors and return discretized feature vectors
    def discretize_features(self, data, ranges):
        result = []
        for feat_vect in data:
            i = 0
            new_feat_vect = []
            for feat in feat_vect:
                if ranges[i] == 'd':
                    new_feat_vect.append(feat)
                    i += 1
                else: # continuous feature
                    if feat <= ranges[i]: # fencepost
                        new_feat_vect.append(1)
                    else: 
                        new_feat_vect.append(0)
                    i += 1
                    for b in range(1, self.buckets):
                        if feat <= ranges[i] and feat > ranges[i - 1]:
                            new_feat_vect.append(1)
                        else: 
                            new_feat_vect.append(0)
                        i += 1
            result.append(new_feat_vect)
        return result


    # create the ranges of the buckets used to discretize the data later
    def create_ranges(self, data):
        ranges = []
        for i in range(0, len(data[0])):
            discrete = True
            values = [vec[i] for vec in data]
            for value in values:
                if value not in [0, 1]:
                    discrete = False
            if discrete:
                ranges.append('d')
            else:
                mx = max(values)
                mn = min(values)
                step = float(mx - mn) / self.buckets
                for b in range(0, self.buckets):
                    ranges.append(mn + step * (b + 1))
        return ranges


    # params:
    #   data: list of tuples of form (class, f) where f is feature vector
    def train(self, data):
        if self.reporting:
            print('Training in progress...')
            t1 = time.time()

        # discretize all features
        preprocessed = self.preprocess([t[1] for t in data])

        class_arr = np.array([t[0] for t in data])
        data_arr = np.array(preprocessed)
        data_arr = data_arr.astype(np.int)

        m = class_arr.shape[0] # number of observations
        n = data_arr.shape[1] # number of attributes in each feature vector

        # calculate marginal probabilities of each class
        class_totals = {}
        self.class_marg_prob = {}

        for c in self.classes:
            count = (class_arr == c).sum()
            class_totals[c] = count
            self.class_marg_prob[c] =  float(count) / m
        # keys are tuples (attribute #, class), values are counts
        conditional_prob_counts = {}  

        for o in range(0, m): # iterate through observations
            for f in range(0, n): # iterate through attributes
                if data_arr[o, f] == 1:
                    t = (f, class_arr[o])
                    if t not in conditional_prob_counts:
                        conditional_prob_counts[t] = 0
                    conditional_prob_counts[t] += 1

        self.cond_probabilities = {}
        self.compute_cond_probabilities(conditional_prob_counts, class_totals, n)

        if self.reporting:
            print('Training complete in: ' + str(time.time() - t1) + ' seconds')
            print('')


    def compute_cond_probabilities(self, conditional_prob_counts, class_totals, n):
        for t, count in conditional_prob_counts.items():
            if self.smoothing:
                self.cond_probabilities[t] = float(conditional_prob_counts[t] + 1) / (class_totals[t[1]] + 2)
            else:
                self.cond_probabilities[t] = float(conditional_prob_counts[t]) / class_totals[t[1]]

        # fill remaining possible tuples
        for f in range(0, n):
            for c in self.classes:
                if (f, c) not in self.cond_probabilities:
                    if self.smoothing:
                        self.cond_probabilities[(f, c)] = 1.0 / (class_totals[c] + 2)
                    else:
                        self.cond_probabilities[(f, c)] = 0


    # params:
    #   data_list: list of feature vectors (lists)
    # returns
    #   list of predicted classes
    def classify(self, data_list):
        if self.reporting:
            print('Classifying in progress...')
            t1 = time.time()
        data = self.preprocess(data_list)
        result = []
        for d in data:
            result.append(self.classify_single(d))

        if self.reporting:
            print('Classifying complete in: ' + str(time.time() - t1) + ' seconds')
            print('')

        return result


    # helper for classify(), classifies single data point
    def classify_single(self, feature_vector):
        classification_dict = {}

        for c in self.classes:
            bayes_prob = self.class_marg_prob[c]
            for i, a in enumerate(feature_vector):
                if a == 1:
                    bayes_prob *= self.cond_probabilities[(i, c)]
                else: # a == 0
                    bayes_prob *= 1 - self.cond_probabilities[(i, c)]
            classification_dict[c] = bayes_prob

        return max(classification_dict, key=lambda key: classification_dict[key])
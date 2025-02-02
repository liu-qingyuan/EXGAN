# -*- coding: utf-8 -*-
"""A set of utility functions to support outlier detection.
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import numpy as np
from numpy import percentile
import numbers
import math

import torch

import sklearn
from sklearn.metrics import precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.utils import column_or_1d
from sklearn.utils import check_array
from sklearn.utils import check_consistent_length

from sklearn.utils import check_random_state
from sklearn.utils.random import sample_without_replacement

MAX_INT = np.iinfo(np.int32).max
MIN_INT = -1 * MAX_INT


def check_parameter(param, low=MIN_INT, high=MAX_INT, param_name='',
                    include_left=False, include_right=False):
    """Check if an input is within the defined range.

    Parameters
    ----------
    param : int, float
        The input parameter to check.

    low : int, float
        The lower bound of the range.

    high : int, float
        The higher bound of the range.

    param_name : str, optional (default='')
        The name of the parameter.

    include_left : bool, optional (default=False)
        Whether includes the lower bound (lower bound <=).

    include_right : bool, optional (default=False)
        Whether includes the higher bound (<= higher bound).

    Returns
    -------
    within_range : bool or raise errors
        Whether the parameter is within the range of (low, high)

    """

    # param, low and high should all be numerical
    if not isinstance(param, (numbers.Integral, np.integer, np.float)):
        raise TypeError('{param_name} is set to {param} Not numerical'.format(
            param=param, param_name=param_name))

    if not isinstance(low, (numbers.Integral, np.integer, np.float)):
        raise TypeError('low is set to {low}. Not numerical'.format(low=low))

    if not isinstance(high, (numbers.Integral, np.integer, np.float)):
        raise TypeError('high is set to {high}. Not numerical'.format(
            high=high))

    # at least one of the bounds should be specified
    if low is MIN_INT and high is MAX_INT:
        raise ValueError('Neither low nor high bounds is undefined')

    # if wrong bound values are used
    if low > high:
        raise ValueError(
            'Lower bound > Higher bound')

    # value check under different bound conditions
    if (include_left and include_right) and (param < low or param > high):
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of [{low}, {high}].'.format(
                param=param, low=low, high=high, param_name=param_name))

    elif (include_left and not include_right) and (
            param < low or param >= high):
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of [{low}, {high}).'.format(
                param=param, low=low, high=high, param_name=param_name))

    elif (not include_left and include_right) and (
            param <= low or param > high):
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of ({low}, {high}].'.format(
                param=param, low=low, high=high, param_name=param_name))

    elif (not include_left and not include_right) and (
            param <= low or param >= high):
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of ({low}, {high}).'.format(
                param=param, low=low, high=high, param_name=param_name))
    else:
        return True


def check_detector(detector):
    """Checks if fit and decision_function methods exist for given detector

    Parameters
    ----------
    detector : pyod.models
        Detector instance for which the check is performed.

    """

    if not hasattr(detector, 'fit') or not hasattr(detector,
                                                   'decision_function'):
        raise AttributeError("%s is not a detector instance." % (detector))


def standardizer(X, X_t=None, keep_scalar=False):
    """Conduct Z-normalization on data to turn input samples become zero-mean
    and unit variance.

    Parameters
    ----------
    X : numpy array of shape (n_samples, n_features)
        The training samples

    X_t : numpy array of shape (n_samples_new, n_features), optional (default=None)
        The data to be converted

    keep_scalar : bool, optional (default=False)
        The flag to indicate whether to return the scalar

    Returns
    -------
    X_norm : numpy array of shape (n_samples, n_features)
        X after the Z-score normalization

    X_t_norm : numpy array of shape (n_samples, n_features)
        X_t after the Z-score normalization

    scalar : sklearn scalar object
        The scalar used in conversion

    """
    X = check_array(X)
    scaler = StandardScaler().fit(X)

    if X_t is None:
        if keep_scalar:
            return scaler.transform(X), scaler
        else:
            return scaler.transform(X)
    else:
        X_t = check_array(X_t)
        if X.shape[1] != X_t.shape[1]:
            raise ValueError(
                "The number of input data feature should be consistent"
                "X has {0} features and X_t has {1} features.".format(
                    X.shape[1], X_t.shape[1]))
        if keep_scalar:
            return scaler.transform(X), scaler.transform(X_t), scaler
        else:
            return scaler.transform(X), scaler.transform(X_t)

def min_max_normalization(X, X_t=None):
    X = check_array(X)
    scaler = MinMaxScaler().fit(X)
    if X_t is not None:
        X_t = check_array(X_t)
        if X.shape[1] != X_t.shape[1]:
            raise ValueError(
                "The number of input data feature should be consistent"
                "X has {0} features and X_t has {1} features.".format(
                    X.shape[1], X_t.shape[1]))
        return scaler.transform(X), scaler.transform(X_t)
    return scaler.transform(X)

def score_to_label(pred_scores, outliers_fraction=0.1):
    """Turn raw outlier outlier scores to binary labels (0 or 1).

    Parameters
    ----------
    pred_scores : list or numpy array of shape (n_samples,)
        Raw outlier scores. Outliers are assumed have larger values.

    outliers_fraction : float in (0,1)
        Percentage of outliers.

    Returns
    -------
    outlier_labels : numpy array of shape (n_samples,)
        For each observation, tells whether or not
        it should be considered as an outlier according to the
        fitted model. Return the outlier probability, ranging
        in [0,1].
    """
    # check input values
    pred_scores = column_or_1d(pred_scores)
    check_parameter(outliers_fraction, 0, 1)

    threshold = percentile(pred_scores, 100 * (1 - outliers_fraction))
    pred_labels = (pred_scores > threshold).astype('int')
    return pred_labels


def precision_n_scores(y, y_pred, n=None):
    """Utility function to calculate precision @ rank n.

    Parameters
    ----------
    y : list or numpy array of shape (n_samples,)
        The ground truth. Binary (0: inliers, 1: outliers).

    y_pred : list or numpy array of shape (n_samples,)
        The raw outlier scores as returned by a fitted model.

    n : int, optional (default=None)
        The number of outliers. if not defined, infer using ground truth.

    Returns
    -------
    precision_at_rank_n : float
        Precision at rank n score.

    """

    # turn raw prediction decision scores into binary labels
    y_pred = get_label_n(y, y_pred, n)

    # enforce formats of y and labels_
    y = column_or_1d(y)
    y_pred = column_or_1d(y_pred)

    return precision_score(y, y_pred)


def get_label_n(y, y_pred, n=None):
    """Function to turn raw outlier scores into binary labels by assign 1
    to top n outlier scores.

    Parameters
    ----------
    y : list or numpy array of shape (n_samples,)
        The ground truth. Binary (0: inliers, 1: outliers).

    y_pred : list or numpy array of shape (n_samples,)
        The raw outlier scores as returned by a fitted model.

    n : int, optional (default=None)
        The number of outliers. if not defined, infer using ground truth.

    Returns
    -------
    labels : numpy array of shape (n_samples,)
        binary labels 0: normal points and 1: outliers

    Examples
    --------
    >>> from pyod.utils.utility import get_label_n
    >>> y = [0, 1, 1, 0, 0]
    >>> y_pred = [0.1, 0.5, 0.3, 0.2, 0.7]
    >>> get_label_n(y, y_pred)
    array([0, 1, 0, 0, 1])

    """

    # enforce formats of inputs
    y = column_or_1d(y)
    y_pred = column_or_1d(y_pred)

    check_consistent_length(y, y_pred)
    y_len = len(y)  # the length of targets

    # calculate the percentage of outliers
    if n is not None:
        outliers_fraction = n / y_len
    else:
        outliers_fraction = np.count_nonzero(y) / y_len

    threshold = percentile(y_pred, 100 * (1 - outliers_fraction))
    y_pred = (y_pred > threshold).astype('int')

    return y_pred

def gmean_scores(y, y_pred, threshold=0.5):
    """Utility function to calculate precision @ rank n.

    Parameters
    ----------
    y : list or numpy array of shape (n_samples,)
        The ground truth. Binary (0: inliers, 1: outliers).

    y_pred : list or numpy array of shape (n_samples,)
        The raw outlier scores as returned by a fitted model.

    
    Returns
    -------
    Gmean: float
    """
    y_pred = get_label_n(y, y_pred)
    #y_pred = (y_pred > threshold).astype('int')
    ones_all = (y==1).sum()
    ones_correct = ((y==1) & (y_pred==1)).sum()
    zeros_all = (y==0).sum()
    zeros_correct = ((y==0) & (y_pred==0)).sum()
    Gmean = np.sqrt(1.0*ones_correct/ones_all)
    Gmean *= np.sqrt(1.0*zeros_correct/zeros_all)

    return Gmean

def get_gmean(y, y_pred, threshold=0.5):
    """Utility function to calculate precision @ rank n.

    Parameters
    ----------
    y : list or numpy array of shape (n_samples,)
        The ground truth. Binary (0: inliers, 1: outliers).

    y_pred : list or numpy array of shape (n_samples,)
        The raw outlier scores as returned by a fitted model.

    
    Returns
    -------
    Gmean: float
    """
    #y_pred = get_label_n(y, y_pred)
    y_pred = (y_pred >= threshold).astype('int')
    ones_all = (y==1).sum()
    ones_correct = ((y==1) & (y_pred==1)).sum()
    zeros_all = (y==0).sum()
    zeros_correct = ((y==0) & (y_pred==0)).sum()
    Gmean = np.sqrt((1.0*ones_correct/ones_all) * (1.0*zeros_correct/zeros_all))
    #Gmean *= np.sqrt

    return Gmean

def get_f_score(y, y_pred, threshold=0.5):
    """Utility function to calculate precision @ rank n.

    Parameters
    ----------
    y : list or numpy array of shape (n_samples,)
        The ground truth. Binary (0: inliers, 1: outliers).

    y_pred : list or numpy array of shape (n_samples,)
        .

    
    Returns
    -------
    f_score: float
    """
    #y_pred = get_label_n(y, y_pred)
    y_pred = (y_pred >= threshold).astype('int')
    
    f_score = f1_score(y, y_pred)
    #Gmean *= np.sqrt

    return f_score


def AUC_and_Gmean1(y_test, y_scores):
    #y_tensor = torch.from_numpy(y_test)
    index = y_test.view(y_test.size(0), 1).long()
    y_pt = y_scores.gather(1, index).view(-1,).detach().cpu().numpy()

    auc = round(roc_auc_score(y_test.detach().cpu().numpy(), y_pt), ndigits=4)
    gmean = round(get_gmean(y_test.detach().cpu().numpy(), y_pt, 0.5), ndigits=4)
    return auc, gmean

def AUC_and_Gmean(y_test, y_scores):
    auc = round(roc_auc_score(y_test, y_scores), ndigits=4)
    gmean = round(get_gmean(y_test, y_scores, 0.5), ndigits=4)
    return auc, gmean

def geometric_mean_score(y_true, y_pred):
    """Calculate geometric mean score
    
    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_pred : array-like
        Predicted binary labels
        
    Returns
    -------
    float
        Geometric mean score
    """
    # 计算每个类别的准确率
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    p = (y_true == 1).sum()
    n = (y_true == 0).sum()
    
    # 计算敏感度(TPR)和特异度(TNR)
    tpr = tp / p if p > 0 else 0
    tnr = tn / n if n > 0 else 0
    
    # 计算几何平均
    return np.sqrt(tpr * tnr)

def get_measure(y_true, y_pred, threshold=0.5):
    y_pred_labels = (y_pred >= threshold).astype(int)
    
    # 计算总体准确率
    acc = np.mean(y_true == y_pred_labels)
    
    # 计算每个类别的准确率
    acc_0 = np.mean(y_pred_labels[y_true == 0] == 0)  # 健康类准确率
    acc_1 = np.mean(y_pred_labels[y_true == 1] == 1)  # 患病类准确率
    
    # 原有的指标计算
    auc = roc_auc_score(y_true, y_pred)
    fscore = f1_score(y_true, y_pred_labels)
    gmean = geometric_mean_score(y_true, y_pred_labels)
    
    return auc, fscore, gmean, acc, acc_0, acc_1


def get_measure_category(y_true, y_score):
    
    auc = roc_auc_score(y_true, y_score)
    y_pred = np.zeros_like(y_true)
    y_pred[y_score>=0.5] = 1
    #s = get_gmean(y_test, y_scores, 0.5)
    #print(s)
    f1 = f1_score(y_true, y_pred)

    ones_all = (y_true==1).sum()
    ones_correct = ((y_true==1) & (y_pred==1)).sum()
    zeros_all = (y_true==0).sum()
    zeros_correct = ((y_true==0) & (y_pred==0)).sum()
    Gmean = np.sqrt((1.0*ones_correct/ones_all) * (1.0*zeros_correct/zeros_all))

    return auc, f1, Gmean

def get_measure_softmax(y_test, y_scores):
    index = y_test.view(y_test.size(0), 1).long()
    y_pt = y_scores.gather(1, index).view(-1,).detach().cpu().numpy()

    y = y_test.cpu().detach().numpy()
    auc = round(roc_auc_score(y, y_pt), ndigits=4)

    
    _, y_pred = torch.max(y_scores, dim=1)
    y_pred = y_pred.cpu().detach().numpy()

    ones_all = (y==1).sum()
    ones_correct = ((y==1) & (y_pred==1)).sum()
    zeros_all = (y==0).sum()
    zeros_correct = ((y==0) & (y_pred==0)).sum()
    gmean = np.sqrt((1.0*ones_correct/ones_all) * (1.0*zeros_correct/zeros_all))

    f1 = round(f1_score(y, y_pred), ndigits=4)

    return auc, f1, gmean

def get_measure_softmax_v2(y_true, y_predict):
    #index = y_true.view(y_true.size(0), 1).long()
    #y_pt = y_predict.gather(1, index).view(-1,).detach().cpu().numpy()

    y = y_true.cpu().detach().numpy()
    y_scores = y_predict[:, 1].clone().detach().cpu().numpy()
    auc = round(roc_auc_score(y, y_scores), ndigits=4)

    y_pred = np.zeros_like(y_true)
    y_pred[y_scores>=0.5] = 1
    #_, y_pred = torch.max(y_predict, dim=1)
    #y_pred = y_pred.cpu().detach().numpy()

    ones_all = (y==1).sum()
    ones_correct = ((y==1) & (y_pred==1)).sum()
    zeros_all = (y==0).sum()
    zeros_correct = ((y==0) & (y_pred==0)).sum()
    gmean = np.sqrt((1.0*ones_correct/ones_all) * (1.0*zeros_correct/zeros_all))

    f1 = round(f1_score(y, y_pred), ndigits=4)

    return auc, f1, gmean

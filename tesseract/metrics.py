# -*- coding: utf-8 -*-

"""
metrics.py
~~~~~~~~~~

A set of measurement tools to aid users designing time-aware experiments.

"""
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics as skmetrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

from tesseract import utils, temporal


def t_slope(metric):
    """Compute the slope with respect to the given metric.

    Args:
        metric: The metric to operate with respect to.

    Returns:
        float: A measure of the trend in the given time range

    """
    reg = LinearRegression()
    reg.fit(np.arange(1, len(metric)+1).reshape(-1, 1), np.array(metric).reshape(-1, 1))

    return reg.coef_[0][0]


def aut(results, metric=None, idx=None):
    """Compute the AUT with respect to the given metric.

    Note that for results spanning a _single_ time period, AUT = 0 as this is
    not considered a time-aware evaluation.

    Args:
        results: The set of time-aware results to operate over.
        metric: The metric to operate with respect to.
        granularinty
    Returns:
        float: A measure of robustness for the applied model over the time
            spanning the results.

    """
    if isinstance(results, dict) or isinstance(results, pd.DataFrame):
        results = results[metric][:idx]

    if len(results) <= 1:
        return 0

    return np.trapz(results) / (len(results) - 1)


def aut_with_granularity(results, granularity, metric=None):
    """Compute the AUT with respect to the given metric.

    Note that for results spanning a _single_ time period, AUT = 0 as this is
    not considered a time-aware evaluation.

    Args:
        granularity:
        results: The set of time-aware results to operate over.
        metric: The metric to operate with respect to.

    Returns:
        float: A measure of robustness for the applied model over the time
            spanning the results.

    """
    metric = {"f1": skmetrics.f1_score,
              "accuracy": skmetrics.accuracy_score,
              "precision": skmetrics.precision_score,
              "recall": skmetrics.recall_score,
              }[metric]

    if len(results["t_tests"]) <= 1:
        return 0

    y_tests = results["y_tests"]
    y_preds = results["y_preds"]
    t_tests = results["t_tests"]

    results = aut_granularity_split(y_tests, y_preds, t_tests, granularity, metric)

    return np.trapz(results) / (len(results) - 1)


def aut_granularity_split(y_tests, y_preds, t_tests, granularity, metric, test_size=1):
    results = []
    y_tests = np.concatenate(y_tests, axis=None)
    y_preds = np.concatenate(y_preds, axis=None)
    t_tests = np.concatenate(t_tests, axis=None)

    _, idxes = temporal.time_aware_indexes(t_tests, 0, test_size, granularity)
    for idx in idxes:
        y_test = y_tests[idx]
        y_pred = y_preds[idx]
        result = metric(y_test, y_pred)
        results.append(result)

    return results


def error_rate(results, metric='f1'):
    """Return the error rate formulation as it relates to the given metric.

    Args:
        results: The set of time-aware evaluation results to operate over.
        metric: The metric to operate with respect to (f1|precision|recall).

    Returns:
        float: The rate representing error for the given metric.

    """
    return {
        'f1': errors(results) / (np.sum(results['p']) + np.sum(results['n'])),
        'precision': np.sum(results['fn']) / (
                np.sum(results['tp']) + np.sum(results['fn'])),
        'recall': np.sum(results['fp']) / (
                np.sum(results['tn']) + np.sum(results['fp'])),
    }[metric]


def errors(results):
    """Return the total misclassifications in the results."""
    return np.sum(results['fn']) + np.sum(results['fp'])


def plot_alpha_assessment(alpha_assessment_results, outfile=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot((alpha_assessment_results['negative_predictions']['correct'],
                alpha_assessment_results['negative_predictions']['incorrect'],
                alpha_assessment_results['positive_predictions']['correct'],
                alpha_assessment_results['positive_predictions']['incorrect']))
    ax.set_xticklabels(('Neg C', 'Neg IC', 'Pos C', 'Pos IC'))

    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()

    return fig, ax


# def plot_results(results, outfile=None, fields=None, title='Scores over time',
#                  quiet=False):
#     if not quiet:
#         logging.info(results)
#     if outfile:
#         results.to_csv(os.path.splitext(outfile)[0] + '.csv')
#
#     if fields is None:
#         fields = ['f1', 'precision', 'recall',
#                   'f1_n', 'precision_n', 'recall_n']
#
#     colors = ('#F2385A', '#F5A503', '#4AD9D9',
#               '#FF9999', '#FFDD99', '#AAEEEE')
#     ax = results[fields].plot(linestyle='--', marker='o', color=colors)
#
#     plt.title(title)
#     ax.set_xlabel('Testing round')
#     ax.set_ylabel('Score')
#     ax.set_ylim([0, 1])
#     ax.set_yticks(np.arange(0, 1.1, 0.1))
#     ax.set_xticks(results.index)
#     ax.grid('on', which='major', linestyle=':', axis='y')
#     plt.tight_layout()
#
#     if outfile:
#         plt.savefig(outfile)
#     else:
#         plt.show()
#
#     return ax


# def plot_by_time(y, t, granularity='month', type='line', outfile=None):
#     df = pd.DataFrame(y, columns=['positive'], index=t)
#     df['negative'] = [1 ^ x for x in df['positive']]
#
#     try:
#         offset_alias = {
#             'year': '1Y',
#             'quarter': '1Q',
#             'month': '1M',
#             'week': '1W',
#             'day': '1D'
#         }[granularity]
#     except KeyError:
#         # Allow a specific offset alias to be passed in
#         offset_alias = granularity
#
#     df = df.resample(offset_alias).sum()
#
#     colors = ('#cc0000', '#66b3ff')
#     plot_fn = df.plot.bar if type == 'bar' else df.plot
#     ax = plot_fn(color=colors, marker='o', linestyle='--')
#
#     plt.title('Frequency of class membership by {}'.format(granularity))
#     ax.set_xlabel('{}(s)'.format(granularity))
#     ax.set_ylabel('Frequency')
#     ax.grid('on', which='major', linestyle=':', axis='y')
#     plt.tight_layout()
#
#     if outfile:
#         plt.savefig(outfile)
#     else:
#         plt.show()
#
#     return ax


def summarize(y):
    positive = sum(y)
    negative = len(y) - positive
    print('Class counts:')
    print('-' * 20)
    print('negative: {}'.format(negative))
    print('positive: {}'.format(positive))
    print('\nTotal objects:')
    print('-' * 20)
    print('{} ({:.04}% positive)'.format(len(y), positive / len(y) * 100))


def get_train_info(X_train, y_train, t_train, existing=None):
    # Ensure results are a defaultdict(list)

    results = defaultdict(list, existing) if existing else defaultdict(list)

    # Ensure label array is a numpy array

    y_train = np.array(y_train)

    train_pos = np.sum(y_train)

    results['train_pos'].append(train_pos)
    results['train_neg'].append(len(y_train) - train_pos)
    results['train_tot'].append(len(y_train))

    return results


def calculate_metrics(y_true, y_pred, existing=None,
                      raw_scores=None, periods=1):
    periods = len(y_pred) if periods == -1 else periods

    if periods > 1:
        for y_t, y_p in zip(y_true, y_pred):
            existing = calculate_metrics(y_t, y_p, existing, raw_scores)
        return existing

    # Ensure results are a defaultdict(list)

    results = defaultdict(list, existing) if existing else defaultdict(list)

    # Ensure both label vectors are Numpy arrays

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Heuristic to check if input are raw scores

    y_raw = None
    if (raw_scores or
            (raw_scores is None and
             utils.check_for_raw_scores(y_pred))):
        y_raw = y_pred

    # Convert output scores or categorical labels to integer labels

    y_pred = utils.resolve_categorical(y_pred)
    y_true = utils.resolve_categorical(y_true)

    # Ensure labels are encoded as integer labels

    if isinstance(y_pred[0], str):
        if isinstance(y_true[0], str):
            try:
                y_pred = np.array(y_pred, dtype='int32')
                y_true = np.array(y_true, dtype='int32')
            except ValueError:
                enc = LabelEncoder().fit(y_true)
                y_true = enc.transform(y_true)
                y_pred = enc.transform(y_pred)
        else:
            try:
                y_pred = np.array(y_pred, dtype='int32')
            except ValueError:
                y_pred = LabelEncoder().fit_transform(y_pred)

    assert len(set(y_true)) <= 2 and len(set(y_pred)) <= 2

    # Update total positive and negative predictions

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=(0, 1)).ravel()
    p = tp + fn
    n = tn + fp

    results['tp'].append(tp)
    results['fp'].append(fp)
    results['tn'].append(tn)
    results['fn'].append(fn)

    results['p'].append(p)
    results['n'].append(n)
    results['tot'].append(p + n)

    # Update cumulative totals

    results['tp_cumu'].append(np.sum(results['tp']))
    results['fp_cumu'].append(np.sum(results['fp']))
    results['tn_cumu'].append(np.sum(results['tn']))
    results['fn_cumu'].append(np.sum(results['fn']))

    results['p_cumu'].append(np.sum(results['p']))
    results['n_cumu'].append(np.sum(results['n']))
    results['tot_cumu'].append(np.sum(results['tot']))

    # Update true/false positive/negative rates

    if p == 0:
        results['tpr'].append(np.nan)
        results['fnr'].append(np.nan)
    else:
        results['tpr'].append(tp / p)
        results['fnr'].append(fn / p)

    if n == 0:
        results['fpr'].append(np.nan)
        results['tnr'].append(np.nan)
    else:
        results['fpr'].append(fp / n)
        results['tnr'].append(tn / n)

    # Calculate AUC-ROC if raw scores have been supplied

    if y_raw is not None:

        # Some classifiers output with a score/prob for each class, this
        # simply includes only the score/prob of the predicted class as
        # skmetrics.roc_auc_score expects both inputs to be the same shape
        if y_raw.shape != y_true.shape:
            y_scores = np.array([np.max(v) for v in y_raw])
        else:
            y_scores = y_raw

        try:
            results['auc_roc'].append(skmetrics.roc_auc_score(y_true, y_scores))
        except ValueError as e:
            print(e)
            results['auc_roc'].append(np.nan)

    # Calculate precision, recall and F1 wrt positive and negative classes

    results['precision'].append(
        skmetrics.precision_score(y_true, y_pred, pos_label=1))
    results['recall'].append(
        skmetrics.recall_score(y_true, y_pred, pos_label=1))
    results['f1'].append(skmetrics.f1_score(y_true, y_pred, pos_label=1))

    results['precision_n'].append(
        skmetrics.precision_score(y_true, y_pred, pos_label=0))
    results['recall_n'].append(
        skmetrics.recall_score(y_true, y_pred, pos_label=0))
    results['f1_n'].append(skmetrics.f1_score(y_true, y_pred, pos_label=0))

    return results


def print_metrics(results, keys=None, header=True):
    if keys is None:
        keys = [
            ('Actual pos', 'p'),
            ('Actual neg', 'n'),
            ('Total', 'tot'),
            ('hline', 'hline'),
            ('TPR', 'tpr'),
            ('FPR', 'fpr'),
            ('TNR', 'tnr'),
            ('FNR', 'fnr'),
            ('AUC ROC', 'auc_roc'),
            ('hline', 'hline'),
            ('Precision', 'precision'),
            ('Recall', 'recall'),
            ('F1', 'f1'),
            ('hline', 'hline')]
    else:
        if isinstance(keys[0], str):
            keys = [(k.title(), k) for k in keys]

    periods = max(len(v) for v in results.values())

    def print_hline():
        print(('-' * 12) + '+' + ('-' * 7 * periods))

    if header:
        header = '{:12}|  '.format('Test period')
        header += ''.join(['{:^7}'.format(i) for i in range(1, periods + 1)])
        print_hline()
        print(header)
        print_hline()

    for label, key in keys:
        if label == 'hline':
            print_hline()

        elif results[key]:
            row = '{:12}|'.format(label)
            for result in results[key]:
                if isinstance(result, float):
                    row += '{:>7.3f}'.format(result)
                else:
                    row += '{:>7}'.format(result)
            print(row)

        else:
            pass  # Silently skip missing keys

# def cumulative(results, metric):
#     if metric not in ('f1', 'precision', 'recall',
#                       'f1_n', 'precision_n', 'recall_n'):
#         return np.cumsum(results[metric])
#
#     tps = np.cumsum(results['tp'])
#     tns = np.cumsum(results['tn'])
#     fps = np.cumsum(results['fp'])
#     fns = np.cumsum(results['fn'])
#
#     precision = tps / (tps + fps)
#     recall = tps / (tps + fns)
#     f1 = 2 * precision * recall / (precision + recall)
#
#     if metric == 'f1':
#         return f1
#     if metric == 'precision':
#         return precision
#     if metric == 'recall':
#         return recall
#
#     precision_n = tns / (tns + fns)
#     recall_n = tns / (tns + fps)
#     f1_n = 2 * precision_n * recall_n / (precision_n + recall_n)
#
#     if metric == 'f1_n':
#         return f1_n
#     if metric == 'precision_n':
#         return precision_n
#     if metric == 'recall_n':
#         return recall_n

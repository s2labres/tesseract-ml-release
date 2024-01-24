# -*- coding: utf-8 -*-

"""
spatial.py
~~~~~~~~~~

A module for working with the class balance of a dataset. Ensuring the class
distribution of the testing data is similar to what will be encountered in a
real deployment is imperative to sound evaluations -- particularly in the
security domains.

Unlike the testing set, the training set is entirely under the operator's
control and class balance can be manipulated in order to over or underrepresent
the positive class in order to achieve greater recall at the expense of
precision (or vice-versa) during the operational phase.

"""
import copy
import random

import numpy as np

import tesseract.metrics as metrics
import tesseract.utils as utils


def assert_class_distribution(y, positive_rate, variance):
    """Helper function to verify the rate of the positive class across y (C3).

    The testing distribution must reflect the real-world class balance observed
    in real-life, otherwise results can be highly inflated (or deflated) with
    respect to realistic performance. This function will verify that this
    constraint is being respected.

    Args:
        y: An array of output class labels y
        positive_rate: The acceptable rate for the positive class.
        variance: The acceptable deviation (+/-) for the positive rate.

    Returns:
        True if the rate of the positive class is acceptable.

    """
    current = np.sum(y) / len(y)
    diff = np.abs(current - positive_rate)
    return diff <= variance


def search_optimal_train_ratio(clf, X_train, y_train, t_train,
                               proper_train_size, validation_size, granularity,
                               start_tr_rate=None, end_tr_rate=0.6, step=0.05,
                               test_noise=0.00, metric='f1'):
    """Find the optimal training ratio in order to maximise the given metric.

    This function performs a grid search between start_tr_rate and end_tr_rate,
    aiming to maximise the value of the given metric (f1|precision|recall),
    while reporting the error rates accumulated at each stage of the algorithm.

    In order to try and pick a training ratio that will be robust to
    fluctuations in the testing distribution, it's possible to specify a value
    for 'test_noise'. The average-best training ratio across a range of values
    between the tr_rates +/- noise will be reported at each stage of the
    algorithm.

    This function will be performed by taking an 'actual' training set and
    dividing it into a 'proper' training and a 'validation' set. For example,
    12 months of data might be split into 8 months and 4 months. The 4 months
    validation aim to simulate the distribution of objects expected after the
    known 12 months so that the chosen training ratio will still be effective.

    Note that validation size refers to a single testing period, so to use 4
    months in the above example, a value of 1 for validation_size and 'month'
    for granularity will divide the remaining objects after the initial 8
    selected for training into 1 month chunks to use for validation.

    Args:
        clf: The classifier to use during the search.
        X_train: The array of predictors to use.
        y_train: The array of output labels to use.
        t_train: The array of aligned datetimes for X (and therefore y).
        proper_train_size: The size of the set to train with.
        validation_size: The size of a _single_ validation period.
        granularity: The granularity of the testing period (year|month|week|day)
        start_tr_rate: The start train rate (typically the natural distribution).
        end_tr_rate: The end train date to test (typically 0.5).
        step: The learning rate of the grid search.
        test_noise: How much noise in the testing ratio to account for.
        metric: The metric to maximise (f1|precision|recall).

    Returns:
        A dictionary of scores and errors for each tested training ratio.

    """
    import tesseract.temporal as temporal
    # Split again to get training and validation sets for finding K
    splits = temporal.time_aware_train_test_split(
        X_train, y_train, t_train, train_size=proper_train_size,
        test_size=validation_size, granularity=granularity)

    aut_list, error_list, fn_list, fp_list, total_list = [], [], [], [], []

    natural_rate = np.mean([sum(y_val) / len(y_val) for y_val in splits[3]])

    if start_tr_rate is None:
        # Start one step below the natural rate of malware
        start_tr_rate = max(
            (round(float(natural_rate) / step) * step) - step, 0)

    tr_proportions = np.arange(start_tr_rate, end_tr_rate + step, step)

    mid = np.round(natural_rate, 2)
    if test_noise == 0:
        te_proportions = (mid,)
    else:
        te_proportions = np.arange(mid - test_noise, mid + test_noise, 0.01)

    for m in tr_proportions:
        X_train_proper, _, \
        y_train_proper, _, \
        t_train_proper, _ = copy.deepcopy(splits)

        # Downsample training to match percentage of malware n
        train_idxs = downsample_to_rate(y_train_proper, m)

        X_train = X_train_proper[train_idxs]
        y_train = y_train_proper[train_idxs]
        t_train = t_train_proper[train_idxs]

        # Alter ratio of malware in testing periods
        errors, auts, total = [], [], []
        fps, fns = [], []
        for n in te_proportions:

            _, X_validations, \
            _, y_validations, \
            __, t_validations = copy.deepcopy(splits)

            for i, _ in enumerate(y_validations):
                val_idxs = downsample_to_rate(y_validations[i], n)
                X_validations[i] = X_validations[i][val_idxs]
                y_validations[i] = y_validations[i][val_idxs]
                t_validations[i] = t_validations[i][val_idxs]

            # Compute results
            results = temporal.fit_predict(clf, X_train, X_validations,
                                           y_train, y_validations,
                                           t_train, t_validations,
                                           training='stationary',
                                           testing='rolling')

            fps.append(np.sum(results['fp']))
            fns.append(np.sum(results['fn']))
            total.append(np.sum(results['p']) + np.sum(results['n']))
            errors.append(metrics.error_rate(results, metric))
            auts.append(metrics.aut(results, metric))

        # print(m, np.mean(total), np.mean(errors), np.mean(auts))
        error_list.append(np.mean(errors))
        aut_list.append(np.mean(auts))
        fp_list.append(np.mean(fps))
        fn_list.append(np.mean(fns))
        total_list.append(np.mean(total))

    return {
        'errors': error_list,
        'auts': aut_list,
        'phis': tr_proportions,
        'fn': fn_list,
        'fp': fp_list,
        'total': total_list
    }


def find_optimal_train_ratio(clf, X_train, y_train, t_train,
                             proper_train_size, validation_size, granularity,
                             start_tr_rate=None, end_tr_rate=0.6, step=0.05,
                             test_noise=0.00, metric='f1', acceptable_errors=0):
    """Given an acceptable threshold for errors, find the optimal train ratio.

    NOTE: The output of the search function that this wraps has undergone quite
    a few tweaks in terms of input and output, and at least until the full
    release of the library, this implementation should be considered a
    prototype (mileage may vary!).

    Args;
        clf: The classifier to use during the search.
        X_train: The array of predictors to use.
        y_train: The array of output labels to use.
        t_train: The array of aligned datetimes for X (and therefore y).
        proper_train_size: The size of the set to train with.
        validation_size: The size of a _single_ validation period.
        granularity: The granularity of the testing period (year|month|week|day)
        start_tr_rate: The start train rate (typically the natural distribution).
        end_tr_rate: The end train date to test (typically 0.5).
        step: The learning rate of the grid search.
        test_noise: How much noise in the testing ratio to account for.
        metric: The metric to maximise (f1|precision|recall).
        acceptable_errors: The threshold of acceptable errors.

    Returns:
        tuple: The optimal discovered ratio, it's AUT and error rate.

    """
    rates = search_optimal_train_ratio(
        clf, X_train, y_train, t_train, proper_train_size,
        validation_size, granularity, start_tr_rate, end_tr_rate,
        step, test_noise, acceptable_errors)

    phis, auts, errors = rates['phis'], rates['auts'], rates['errors']

    for i in np.argsort(auts)[::-1]:
        if errors[i] <= acceptable_errors:
            return phis[i], auts[i], errors[i]

    print('Warning: No training rate found that allows acceptable error rate')
    return None


def downsample_set(X, y, t, min_pos_rate, max_pos_rate=None,
                   noise_deviation=0.0, fixed_size=False):
    """Enforce a class distribution by downsampling.

    Args:
        X: The array of predictors to use.
        y: The array of output labels to use.
        t: The array of aligned datetimes for X (and therefore y).
        min_pos_rate: The minimum proportion of the positive class acceptable.
        max_pos_rate: The maximum proportion of the positive class acceptable.
        noise_deviation: Addition of noise either side of the given proportions.
        fixed_size: Whether to fix the total size of X to the size of the
            minimum class.

    Returns:
        tuple: A resized X, y and t
    """
    new_idxs = downsample_to_rate(y, min_pos_rate, max_pos_rate,
                                  noise_deviation, fixed_size)
    return X[new_idxs], y[new_idxs], t[new_idxs]


def downsample_to_rate(y, min_pos_rate, max_pos_rate=None,
                       noise_deviation=0.0, fixed_size=False):
    """Enforce a class distribution by downsampling.

    Args:
        y: The array of output labels to use.
        min_pos_rate: The minimum proportion of the positive class acceptable.
        max_pos_rate: The maximum proportion of the positive class acceptable.
        noise_deviation: Addition of noise either side of the given proportions.
        fixed_size: Whether to fix the total size of X to the size of the
            minimum class.

    Returns:
        An array of selected indexes.

    """
    # random.seed(33)
    if max_pos_rate is None:
        max_pos_rate = min_pos_rate

    min_pos_rate = utils.resolve_percentage(min_pos_rate)
    max_pos_rate = utils.resolve_percentage(max_pos_rate)

    if not (0 <= min_pos_rate <= 1 or 0 <= max_pos_rate <= 1):
        raise ValueError(
            'Please supply a proportion in the interval [0, 1]')

    n_pos, n_neg = np.sum(y), np.sum(y == 0)

    # Fix the training set size while downsampling to minority class size
    if fixed_size:
        n_tot = min(n_pos, n_neg)
    else:
        n_tot = n_pos + n_neg

    current_pos_perc = float(n_pos) / float(n_tot)

    if current_pos_perc < min_pos_rate:
        pos_perc = min_pos_rate
    elif current_pos_perc > max_pos_rate:
        pos_perc = max_pos_rate
    else:  # min_pos <= current_pos_perc <= max_pos:
        neg_indexes = np.where(y == 0)[0]
        pos_indexes = np.where(y == 1)[0]
        return np.hstack((neg_indexes, pos_indexes))

    pos_perc += np.random.normal(0, noise_deviation)

    # print("Starting downsampling {:.1f}% malware function: n_gw = {:,} ; n_mw = {:,} ; n_tot = {:,}".format(perc_mw*100, n_gw, n_mw, n_tot))

    can_downsample_pos = True
    can_downsample_neg = True

    # First, try downsampling goodware
    if fixed_size:
        n_neg_to_choose = int((1 - pos_perc) * n_tot)
    else:
        n_neg_to_choose = int(
            (float(1 - pos_perc) / float(pos_perc)) * n_pos)

    if n_neg_to_choose > n_neg:
        n_neg_to_choose = n_neg
        can_downsample_neg = False
        # print("Failed to downsample goodware, since: n_gw_to_pick ({}) > n_gw ({})".format(n_gw_to_pick, n_gw))

    # updating the value n_tot after downsampling the goodware

    if fixed_size:
        n_pos_to_choose = int(pos_perc * n_tot)
    else:
        n_pos_to_choose = int(
            (float(pos_perc) / float(1 - pos_perc)) * n_neg)

    if n_pos_to_choose > n_pos:
        can_downsample_pos = False
        # print("Cannot oversample malware to {:.1f}% of {:,}!".format(perc_mw*100, n_tot))

    # elif n_mw_to_pick < n_pos:
    # print("Downsampled malware to {:.1f}% (n_mw = {:,}, n_mw_to_pick = {:,})".format(perc_mw*100, n_mw, n_mw_to_pick))

    # import IPython; IPython.embed(); exit()

    # print("After downsampling: n_gw = {:,} ; n_mw = {:,} ; n_tot = {:,}".format(n_gw_to_pick, n_mw_to_pick, n_gw_to_pick+n_mw_to_pick))

    neg_indexes = np.where(y == 0)[0]
    pos_indexes = np.where(y == 1)[0]

    neg_idx_subsample, pos_idx_subsample = neg_indexes, pos_indexes

    # Downsample goodware
    if can_downsample_neg:
        neg_idx_subsample = random.sample(list(neg_indexes),
                                          n_neg_to_choose)

    if can_downsample_pos:
        pos_idx_subsample = random.sample(list(pos_indexes),
                                          n_pos_to_choose)

    if not (can_downsample_neg or can_downsample_pos):
        raise Exception("Downsampling failed")

    sampled = np.hstack((np.array(neg_idx_subsample),
                         np.array(pos_idx_subsample)))

    return np.array(sampled, dtype=int)

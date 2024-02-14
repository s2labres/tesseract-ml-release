import logging
import os
import statistics
import numpy as np
from tqdm import tqdm
import pickle as pkl
from sklearn import metrics as mtcs
import multiprocessing as mp
from itertools import repeat


def sort_by_predicted_label(
        scores, predicted_labels, groundtruth_labels, consider='correct'):
    """Sort scores into lists of their respected predicted classes.

    Divide a set of scores into 'predicted positive' and 'predicted
    negative' results. Optionally consider only correct or incorrect
    predictions. `scores`, `predicted_labels`, and `groundtruth_labels`
    should be aligned (one per observation).

    Example:
        >>> s = np.array([0.8, 0.7, 0.6, 0.9])
        >>> y_pred = np.array([1, 1, 0, 0])
        >>> y_true = np.array([1, 0, 1, 0])
        >>> sort_by_predicted_label(s, y_pred, y_true, 'correct')
        (array([0.8]), array([0.9]))
        >>> sort_by_predicted_label(s, y_pred, y_true, 'incorrect')
        (array([0.7]), array([0.6]))
        >>> sort_by_predicted_label(s, y_pred, y_true, 'all')
        (array([0.8, 0.7]), array([0.6, 0.9]))

    Args:
        scores (np.ndarray): Predicted scores to be sorted.
        predicted_labels (np.ndarray): The prediction outcome for each object.
        groundtruth_labels (np.ndarray): The groundtruth label for each object.
        consider (str): ['correct'|'incorrect'|'all']. Whether to consider only
            correct predictions, incorrect predictions, or not to distinguish
            between them.

    Returns:
        (np.ndarray, np.ndarray): Tuple of sorted scores (malware, goodware).

    """

    def predicted(i, k):
        return predicted_labels[i] == k

    def correct(i, k):
        return predicted(i, k) and (groundtruth_labels[i] == k)

    def incorrect(i, k):
        return predicted(i, k) and (groundtruth_labels[i] == (k ^ 1))

    if consider == 'correct':
        select = correct
    elif consider == 'incorrect':
        select = incorrect
    elif consider == 'all':
        select = predicted
    else:
        raise ValueError('Unknown thresholding criteria!')

    scores_mw = [scores[i] for i in range(len(scores)) if select(i, 1)]
    scores_gw = [scores[i] for i in range(len(scores)) if select(i, 0)]

    return np.array(scores_mw), np.array(scores_gw)


def apply_threshold(binary_thresholds, test_scores, y_test):
    """Returns a 'keep mask' describing which elements to include.

    Elements that fall above the threshold (and should be kept) have
    their indexes marked TRUE.

    Elements that fall below the threshold (and should be rejected) have
    their indexes marked FALSE.

    `binary_thresholds` expects a dictionary keyed by 'cred' and/or 'conf',
    with sub-dictionaries containing the thresholds for the mw and gw classes.

    Note that the keys of `binary_thresholds` determine _which_ thresholding
    criteria will be enforced. That is, if only a 'cred' dictionary is supplied
    thresholding will be enforced on cred-only and the same for 'conf'.
    Supplying cred and conf dictionaries will enforce the 'cred+conf'
    thresholding criteria (all thresholds will be applied).

    `test_scores` expects a dictionary in much the same way, with at least the
    same keys as `binary_thresholds` ('cred' and/or 'conf' at the top level).

    Example:
        >>> thresholds = {'cred': {'mw': 0.4, 'gw': 0.6},
        ...               'conf': {'mw': 0.5, 'gw': 0.8}}
        >>> scores = {'cred': [0.4, 0.2, 0.7, 0.8, 0.6],
        ...           'conf': [0.6, 0.8, 0.3, 0.2, 0.4]}
        >>> y = np.array([1, 1, 1, 0, 0])
        >>> apply_threshold(thresholds, scores, y)
        array([ True, False, False, False, False])

    Args:
        binary_thresholds(dict): The threshold to apply.
        test_scores (dict): The test scores to apply the threshold to.
        y_test (np.ndarray): The set of predictions to decide which 'per-class'
            threshold to use. Depending on the stage of conformal evaluation,
            this could be either the predicted or ground truth labels.

    Returns:
        np.ndarray: Boolean mask to use on the elements (1 = kept, 0 = reject).

    """
    # Assert preconditions
    assert (set(binary_thresholds.keys()) in
            [{'cred'}, {'conf'}, {'cred', 'conf'}])

    for key in binary_thresholds.keys():
        assert key in test_scores.keys()
        assert set(binary_thresholds[key].keys()) == {'mw', 'gw'}

    def get_class_threshold(criteria, k):
        return (binary_thresholds[criteria]['mw'] if k == 1
                else binary_thresholds[criteria]['gw'])

    keep_mask = []
    for i, y_prediction in enumerate(y_test):

        cred_threshold, conf_threshold = 0, 0
        current_cred, current_conf = 0, 0

        if 'cred' in binary_thresholds:
            key = 'cred'
            current_cred = test_scores[key][i]
            cred_threshold = get_class_threshold(key, y_prediction)

        if 'conf' in binary_thresholds:
            key = 'conf'
            current_conf = test_scores[key][i]
            conf_threshold = get_class_threshold(key, y_prediction)

        keep_mask.append(
            (current_cred >= cred_threshold) and
            (current_conf >= conf_threshold))

    return np.array(keep_mask, dtype=bool)


def get_performance_with_rejection(y_true, y_pred, keep_mask, full=True):
    """Get test results, rejecting predictions based on a given keep mask.

    Args:
        y_true (np.ndarray): The groundtruth label for each object.
        y_pred (np.ndarray): The set of predictions to decide which 'per-class'
            threshold to use. Depending on the stage of conformal evaluation,
            this could be either the predicted or ground truth labels.
        keep_mask (np.ndarray): A boolean mask describing which elements to
            keep (True) or reject (False).
        full (bool): True if full statistics are required, False otherwise.
            False is computationally less expensive.

    Returns:
        dict: A dictionary of results for baseline, kept, and rejected metrics.

    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    d = {}

    total_neg = len(y_pred) - sum(y_pred)
    total_pos = sum(y_pred)

    kept_total_perc = sum(keep_mask) / len(keep_mask)
    reject_total_perc = sum(~keep_mask) / len(keep_mask)

    kept_neg = len(y_pred[keep_mask]) - sum(y_pred[keep_mask])
    kept_pos = sum(y_pred[keep_mask])

    reject_neg = total_neg - kept_neg
    reject_pos = total_pos - kept_pos

    kept_neg_perc = (kept_neg / total_neg)
    kept_pos_perc = (kept_pos / total_pos)

    reject_neg_perc = 1 - kept_neg_perc
    reject_pos_perc = 1 - kept_pos_perc

    reject_neg_total = reject_neg / len(y_pred)
    reject_pos_total = reject_pos / len(y_pred)

    d.update({'total_neg': total_neg,
              'total_pos': total_pos,
              'kept_total_perc': kept_total_perc,
              'reject_total_perc': reject_total_perc,
              'kept_neg': kept_neg, 'kept_pos': kept_pos,
              'reject_neg': reject_neg, 'reject_pos': reject_pos,
              'kept_neg_perc': kept_neg_perc,
              'kept_pos_perc': kept_pos_perc,
              'reject_neg_perc': reject_neg_perc,
              'reject_pos_perc': reject_pos_perc,
              'reject_neg_total': reject_neg_total,
              'reject_pos_total': reject_pos_total})

    f1_b = mtcs.f1_score(y_true, y_pred)
    f1_k = mtcs.f1_score(y_true[keep_mask],
                         y_pred[keep_mask])
    f1_r = mtcs.f1_score(y_true[~keep_mask],
                         y_pred[~keep_mask])

    d.update({'f1_b': f1_b, 'f1_k': f1_k, 'f1_r': f1_r})

    precision_b = mtcs.precision_score(y_true, y_pred)

    precision_k = mtcs.precision_score(y_true[keep_mask],
                                       y_pred[keep_mask])
    precision_r = mtcs.precision_score(y_true[~keep_mask],
                                       y_pred[~keep_mask])
    d.update({'precision_b': precision_b,
              'precision_k': precision_k,
              'precision_r': precision_r})

    recall_b = mtcs.recall_score(y_true, y_pred)

    recall_k = mtcs.recall_score(y_true[keep_mask],
                                 y_pred[keep_mask])
    recall_r = mtcs.recall_score(y_true[~keep_mask],
                                 y_pred[~keep_mask])
    d.update({'recall_b': recall_b, 'recall_k': recall_k, 'recall_r': recall_r})

    if full:
        cf_baseline = mtcs.confusion_matrix(y_true, y_pred)

        cf_keep = mtcs.confusion_matrix(y_true[keep_mask],
                                        y_pred[keep_mask])
        cf_reject = mtcs.confusion_matrix(y_true[~keep_mask],
                                          y_pred[~keep_mask])
        try:
            tn_b, fp_b, fn_b, tp_b = cf_baseline.ravel()
            tn_k, fp_k, fn_k, tp_k = cf_keep.ravel()
            tn_r, fp_r, fn_r, tp_r = cf_reject.ravel()
        except Exception as e:
            print(f'Transcendent met a problem: {e}')

            return d

        d.update({
            'tn_b': tn_b, 'fp_b': fp_b, 'fn_b': fn_b, 'tp_b': tp_b,
            'tn_k': tn_k, 'fp_k': fp_k, 'fn_k': fn_k, 'tp_k': tp_k,
            'tn_r': tn_r, 'fp_r': fp_r, 'fn_r': fn_r, 'tp_r': tp_r
        })

        d['tpr_b'] = tp_b / (tp_b + fn_b)
        d['tpr_k'] = tp_k / (tp_k + fn_k)
        d['tpr_r'] = tp_r / (tp_r + fn_r)

        d['fpr_b'] = fp_b / (fp_b + tn_b)
        d['fpr_k'] = fp_k / (fp_k + tn_k)
        d['fpr_r'] = fp_r / (fp_r + tn_r)

    return d


def test_with_rejection(
        binary_thresholds, test_scores, groundtruth_labels, predicted_labels, full=True):
    """Get test results, rejecting predictions based on a given threshold.

    `binary_thresholds` expects a dictionary keyed by 'cred' and/or 'conf',
    with sub-dictionaries containing the thresholds for the mw and gw classes.

    Note that the keys of `binary_thresholds` determine _which_ thresholding
    criteria will be enforced. That is, if only a 'cred' dictionary is supplied
    thresholding will be enforced on cred-only and the same for 'conf'.
    Supplying cred and conf dictionaries will enforce the 'cred+conf'
    thresholding criteria (all thresholds will be applied).

    `test_scores` expects a dictionary in much the same way, with at least the
    same keys as `binary_thresholds` ('cred' and/or 'conf' at the top level).

    See Also:
        - `apply_threshold`
        - `get_performance_with_rejection`

    Args:
        binary_thresholds (dict): The threshold to apply.
        test_scores (dict): The test scores to apply the threshold to.
        groundtruth_labels (np.ndarray): The groundtruth label for each object.
        predicted_labels (np.ndarray): The set of predictions to decide which
            'per-class' threshold to use. Depending on the stage of conformal
            evaluation, this could be either the predicted or ground truth
            labels.
        full (boolean): Optimization flag which dictates how much data to return,
            default is True. False gives a lot more performance but removes a lot 
            of metrics. 

    Returns:
        dict: A dictionary of results for baseline, kept, and rejected metrics.

    """
    keep_mask = apply_threshold(
        binary_thresholds=binary_thresholds,
        test_scores=test_scores,
        y_test=predicted_labels)

    results = get_performance_with_rejection(
        y_true=groundtruth_labels,
        y_pred=predicted_labels,
        keep_mask=keep_mask,
        full=full)

    return results


def random_threshold(scores, predicted_labels):
    """Produce random thresholds over the given scores.

    Args:
        scores (dict): The test scores on which to produce a threshold.
        predicted_labels (np.ndarray): The set of predictions to decide which
            'per-class' threshold to use.

    Returns:
        dict: Set of thresholds for malware ('gw') and goodware ('gw') classes.

    """
    scores_mw, scores_gw = sort_by_predicted_label(
        scores, predicted_labels, np.array([]), 'all')
    mw_threshold = np.random.uniform(min(scores_mw), max(scores_mw))
    gw_threshold = np.random.uniform(min(scores_gw), max(scores_gw))
    return {'mw': mw_threshold, 'gw': gw_threshold}


def format_opts(metrics, results):
    """Helper function for formatting the results of a list of metrics."""
    return ('{}: {:.4f} | ' * len(metrics)).format(
        *[item for sublist in
          zip(metrics, [results[k] for k in metrics]) for
          item in sublist])


def find_random_search_thresholds(
        scores, predicted_labels, groundtruth_labels,
        max_metrics='f1_k,kept_total_perc', min_metrics='f1_r',
        ceiling=0.25, max_samples=100, objective_func=None):
    """Perform a random grid search to find the best thresholds on `scores`.

    `scores` expects a dictionary keyed by 'cred' and/or 'conf',
    with sub-dictionaries containing the thresholds for the mw and gw classes.

    Note that the keys of `scores` determine _which_ thresholding criteria will
    be enforced. That is, if only a 'cred' dictionary is supplied, thresholding
    will be enforced on cred-only and the same for 'conf'. Supplying cred and
    conf dictionaries will enforce the 'cred+conf' thresholding criteria (all
    thresholds will be applied).

    `max_metrics` and `min_metrics` describe the metrics that should be
    maximised or minimised if the default objective function is being used
    (a harmonic mean, selected with `objective_func=None`). It expects either
    a list of possible metrics, or a string or comma separated metrics.

    For example, both of the following are acceptable:

        > max_metrics = ['f1_k', 'kept_total_perc']
        > max_metrics = 'f1_k,kept_total_perc'

    `ceiling` describes the constraints of the optimization function. If any of
    the selected metrics exceed the value given then the thresholds chosen are
    discarded. `ceiling` expects a dictionary of metrics and maximum acceptable
    values. Alternatively, arguments can be given in string form as comma-
    separated key:value pairs, e.g., 'key1:value1,key2:value2,key3:value3'.
    Finally, if a float is provided, it's interpreted as being the maximum
    acceptable value for the total number of rejected predictions.

    To summarise, all of the following are equivalent:

        > ceiling = {'total_reject_perc': 0.25}
        > ceiling = 'total_reject_perc:0.25'
        > ceiling = 0.25

    For a list of possible metrics, see the keys in the dict produced by
    `get_performance_with_rejection()`. Note that the default objective
    function assumes that the provided metrics are in the interval [0,1].

    `objective_func` is the objective function to maximise during the random
    search. By default (`objective_func=None`), it will maximise the harmonic
    mean of the given `max_metrics` and 1 - each of the given `min_metrics`.

    A custom objective function can be provided which should expect a result
    dictionary of metrics just like the dictionary produced by
    `get_performance_with_rejection()`.

    See Also:
        - `get_performance_with_rejection`

    Args:
        scores (dict): The test scores on which to perform the random search.
        predicted_labels (np.ndarray): The set of predictions to decide which
            'per-class' threshold to use.
        groundtruth_labels (np.ndarray): The groundtruth label for each object.
        max_metrics: The metrics that should be maximised.
        min_metrics: The metrics that should be minimised.
        ceiling: Can be passed an empty dict if you don't want to enforce any
            constraint in this way.
        max_samples (int): The maximum number of random threshold combinations
            to try before settling for the best performance up to that point.
        objective_func (function): The objective function to maximise.

    Returns:
        dict: Set of thresholds for malware ('gw') and goodware ('gw') classes.

    """

    # Resolve possible formats for `max_metrics` and `min_metrics`.
    def resolve_opt_list(x):
        return x.split(',') if isinstance(x, str) else x

    min_metrics = resolve_opt_list(min_metrics)
    max_metrics = resolve_opt_list(max_metrics)

    # Resolve possible formats of `ceiling`.
    ceiling = {} if ceiling is None else ceiling

    ceiling = ({'total_reject_perc': ceiling}
               if isinstance(ceiling, (int, float)) else ceiling)

    if isinstance(ceiling, str):
        pairs = ceiling.split(',')
        pairs = [x.split(':') for x in pairs]
        ceiling = {k: float(v) for k, v in pairs}

    # Resolve objective function to use during the optimization.
    def harm_mean(d):
        maximise = [d[m] for m in max_metrics]
        maximise.extend([1 - d[m] for m in min_metrics])
        return statistics.harmonic_mean(maximise)

    objective_func = harm_mean if objective_func is None else objective_func

    best_outcome, n_samples = 0, 0
    best_thresholds, best_results = {}, {}

    while True:
        # Choose and package random thresholds
        thresholds = {}
        if 'cred' in scores:
            cred_thresholds = random_threshold(scores['cred'], predicted_labels)
            thresholds['cred'] = cred_thresholds
        if 'conf' in scores:
            conf_thresholds = random_threshold(scores['conf'], predicted_labels)
            thresholds['conf'] = conf_thresholds

        # Test with chosen thresholds
        results = test_with_rejection(
            thresholds, scores, groundtruth_labels, predicted_labels, full=True)

        # Check if any results exceed given constraints (e.g. too many rejects)
        unacceptable = [results[k] > v for k, v in ceiling.items()]
        if any(unacceptable):
            continue

        # 'Score' current thresholds with objective function
        outcome = objective_func(results)

        # If current thresholds are better, save new best outcomes
        if outcome > best_outcome:
            best_outcome = outcome
            best_thresholds = thresholds
            best_results = results

            logging.info('New best: [{:.4f}] @ {} || Max: {}Min: {}'.format(
                outcome, thresholds,
                format_opts(max_metrics, results),
                format_opts(min_metrics, results)))
            # report_results(results)
            logging.warning('{} combinations sampled so far!'.format(n_samples))

        # If the maximum number of thresholds have been sampled, abort search
        if max_samples is not None and n_samples >= max_samples:
            logging.warning(
                'Max samples reached ({}) - search aborted'.format(max_samples))
            logging.info('Settling for: [{}] @ {} || Max: {}Min: {}'.format(
                best_outcome, best_thresholds,
                format_opts(max_metrics, best_results),
                format_opts(min_metrics, best_results)))
            # report_results(results)

            return best_thresholds

        n_samples += 1


def package_cred_conf(cred_values, conf_values, criteria):
    package = {}
    if 'cred' in criteria:
        package['cred'] = cred_values
    if 'conf' in criteria:
        package['conf'] = conf_values

    return package


def compute_single_cred_p_value(
        train_ncms, groundtruth_train, single_test_ncm, single_y_test):
    """Compute a single credibility p-value.

    Credibility p-values describe how 'conformal' a point is with respect to
    the other objects of that class. They're computed as the proportion of
    points with greater NCMs (the number of points _less conforming_ than the
    reference point) over the total number of points.

    Intuitively, a point predicted as malware which is the further away from
    the decision boundary than any other point will have the highest p-value
    out of all other malware points. It will have the smallest NCM (as it is
    the least _non-conforming_) and thus no other points will have a greater
    NCM and it will have a credibility p-value of 1.

    Args:
        train_ncms (np.ndarray): An array of training NCMs to compare the
            reference point against.
        groundtruth_train (np.ndarray): An array of ground truths corresponding
            to `train_ncms`.
        single_test_ncm (float): A single reference point to compute the
            p-value of.
        single_y_test (int): Either the ground truth (calibration) or predicted
            label (testing) of `single_test_ncm`.

    See Also:
        - `compute_p_values_cred_and_conf`
        - `compute_single_conf_p_value`

    Returns:
        float: The p-value for `single_test_ncm` w.r.t. `train_ncms`.

    """
    assert len(set(groundtruth_train)) == 2  # binary classification tasks only

    how_many_are_greater_than_single_test_ncm = 0

    for ncm, groundtruth in zip(train_ncms, groundtruth_train):
        if groundtruth == single_y_test and ncm >= single_test_ncm:
            how_many_are_greater_than_single_test_ncm += 1

    single_cred_p_value = (how_many_are_greater_than_single_test_ncm /
                           sum(1 for y in groundtruth_train if
                               y == single_y_test))
    return single_cred_p_value


def compute_single_conf_p_value(
        train_ncms, groundtruth_train, single_test_ncm, single_y_test):
    """Compute a single confidence p-value.

    The confidence p-value is computed similarly to the credibility p-value,
    except it aims to capture the confidence that the classifier has that the
    point _doesn't_ belong to the opposite class.

    To achieve this we assume that point has the label of the second highest
    scoring class---in binary classification, simply the opposite class---and
    compute the credibility p-value with respect to other points of that class.
    The confidence p-value is (1 - this value).

    Note that in transductive conformal evaluation, the entire classifier
    should be retrained with the reference point given the label of the
    opposite class. Usually, this is computationally prohibitive, and so this
    approximation assumes that the decision boundary undergoes only minimal
    changes when the label of a single point is flipped.

    See Also:
        - `compute_p_values_cred_and_conf`
        - `compute_single_cred_p_value`

    Args:
        train_ncms (np.ndarray): An array of training NCMs to compare the
            reference point against.
        groundtruth_train (np.ndarray): An array of ground truths corresponding
            to `train_ncms`.
        single_test_ncm (float): A single reference point to compute the
            p-value of.
        single_y_test (int): Either the ground truth (calibration) or predicted
            label (testing) of `single_test_ncm`.

    Returns:
        float: The p-value for `single_test_ncm` w.r.t. `train_ncms`.

    """
    assert len(set(groundtruth_train)) == 2  # binary classification tasks only

    # 'Cast' NCMs to NCMs with respect to the opposite class (binary only)
    # train_ncms_opposite_class = -1 * np.array(train_ncms)
    single_y_test_opposite_class = 0 if single_y_test == 1 else 1
    single_test_ncm_opposite_class = -1 * single_test_ncm

    how_many_are_greater_than_single_test_ncm = 0

    for ncm, groundtruth in zip(train_ncms, groundtruth_train):
        if (groundtruth == single_y_test_opposite_class
                and ncm >= single_test_ncm_opposite_class):
            how_many_are_greater_than_single_test_ncm += 1

    single_cred_p_value_opposite_class = (
            how_many_are_greater_than_single_test_ncm /
            sum(1 for y in groundtruth_train if
                y == single_y_test_opposite_class))

    return 1 - single_cred_p_value_opposite_class  # confidence p value


def compute_p_values_cred_and_conf(
        train_ncms, groundtruth_train, test_ncms, y_test):
    """Helper function to compute p-values across an entire array."""
    cred = [compute_single_cred_p_value(train_ncms=train_ncms,
                                        groundtruth_train=groundtruth_train,
                                        single_test_ncm=ncm,
                                        single_y_test=y)
            for ncm, y in tqdm(
            zip(test_ncms, y_test), total=len(y_test), desc='cred pvals', position=0, leave=True)]
    # conf = [compute_single_conf_p_value(train_ncms=train_ncms,
    #                                     groundtruth_train=groundtruth_train,
    #                                     single_test_ncm=ncm,
    #                                     single_y_test=y)
    #         for ncm, y in tqdm(
    #         zip(test_ncms, y_test), total=len(y_test), desc='conf pvals', position=0, leave=True)]

    return {'cred': cred}
    # , 'conf': conf


def get_svm_ncms(decision_function, X_in, y_in):
    """Helper functions to get NCMs across an entire pair of X,y arrays. """
    return [get_single_svm_ncm(decision_function, x, y) for x, y in
            tqdm(zip(X_in, y_in), total=len(y_in), desc='svm ncms', position=0, leave=True)]


def get_single_svm_ncm(decision_function, single_x, single_y):
    """Collect a non-conformity measure from the classifier for `single_x`.

    A note about SVM ncms: In binary classification with a linear SVM, the
    output score is the distance from the hyperplane with respect to the
    positive class. If the score is negative, the prediction is class 0, if
    positive, it's class 1 (in sklearn technically it will be clf.class_[0] and
    clf.class_[1] respectively). To perform thresholding with conformal
    evaluator, we need the distance from the hyperplane with respect to *both*
    classes, so we simply flip the sign to get the 'reflection' for the other
    class.

    Args:
        clf (sklearn.svm.SVC): The classifier to use for the NCMs.
        single_x (np.ndarray): An single feature vector to get the NCM for.
        single_y (int): The ground truth corresponding to feature vector
            `single_x`.

    Returns:
        float: The NCM for the given `single_x`.

    """
    decision = decision_function(single_x)

    # If y (ground truth in calibration, prediction in testing) is malware
    # then flip the sign to ensure the most conforming point is most minimal.
    # decision = -abs(decision)
    # mal;1 -> 0
    if single_y == 1:
        return -decision
    elif single_y == 0:
        return decision
    raise Exception('Unknown class? Only binary decisions supported.')


def cache_data(model, data_path):
    """Cache data (trained model, computed p-values, etc).

    Args:
        model: The data to save.
        data_path: (str) To avoid mix-ups, and to allow safe caching of models
            produced during calibration, it's advised to keep this location
            'fold-specific'.

    See Also:
        - `load_cached_data`

    """

    model_folder_path = os.path.dirname(data_path)

    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)

    logging.info('Saving data to {}...'.format(data_path))
    with open(data_path, 'wb') as f:
        pkl.dump(model, f)
    logging.debug('Done.')


def load_cached_data(data_path):
    """Load cached data (trained model, computed p-values, etc).

    Args:
        data_path: (str) To avoid mix-ups, and to allow safe caching of models
            produced during calibration, it's advised to keep this location
            'fold-specific'.

    See Also:
        - `cache_data`

    Returns:
        The previously cached data.

    """
    logging.info('Loading data from {}...'.format(data_path))
    with open(data_path, 'rb') as f:
        model = pkl.load(f)
    logging.debug('Done.')
    return model


def train_calibration_ice(clf, X_proper_train, X_cal,
                          y_proper_train, y_cal,
                          fold_index):
    """Train calibration set (for a single fold).

    Quite a bit of information is needed here for the later p-value
    computation and probability comparison. The returned dictionary has
    the following structure:

        'cred_p_val_cal_fold'  -->  # Calibration credibility p values
        'conf_p_val_cal_fold'  -->  # Calibration confidence p values
        'ncms_cal_fold'        -->  # Calibration NCMs
        'pred_cal_fold'        -->  # Calibration predictions
        'groundtruth_cal_fold' -->  # Calibration groundtruth
        'probas_cal_fold'      -->  # Calibration probabilities
        'pred_proba_cal_fold'  -->  # Calibration predictions

    Args:
        X_proper_train (np.ndarray): Features for the 'proper training
            set' partition.
        X_cal (np.ndarray): Features for a single calibration set
            partition.
        y_proper_train (np.ndarray): Ground truths for the 'proper
            training set' partition.
        y_cal (np.ndarray): Ground truths for a single calibration set
            partition.
        fold_index: An index to identify the current fold (used for caching).

    Returns:
        dict: Fold results, structure as in the docstring above.

    """
    # Train model with proper training
    clf.fit(X_proper_train, y_proper_train)

    # Get ncms for proper training fold
    logging.debug('Getting training ncms for fold {}...'.format(fold_index))
    groundtruth_proper_train_fold = y_proper_train

    # Get ncms for calibration fold

    logging.debug('Getting calibration ncms for fold {}...'.format(fold_index))
    pred_cal_fold = clf.predict(X_cal)
    groundtruth_cal_fold = y_cal

    # Compute p values for calibration fold

    logging.debug('Computing cal p values for fold {}...'.format(fold_index))
    ncms_cal_fold = get_svm_ncms(clf.decision_function, X_cal, y_cal)
    # data.cache_data(ncms_cal_fold, saved_ncms_name)

    # saved_pvals_name = 'p_vals_{}_cal_fold_{}.p'.format(alg_name, fold_index)
    # saved_pvals_name = os.path.join(saved_data_folder, saved_pvals_name)
    #
    # if os.path.exists(saved_pvals_name):
    #     p_val_cal_fold_dict = data.load_cached_data(saved_pvals_name)
    # else:
    #     # TODO | Doublecheck implications of duplicating the reference
    #     # TODO | point in the 'train_ncms'
    p_val_cal_fold_dict = compute_p_values_cred_and_conf(
        train_ncms=ncms_cal_fold,
        groundtruth_train=groundtruth_cal_fold,
        test_ncms=ncms_cal_fold,
        y_test=groundtruth_cal_fold)
    # data.cache_data(p_val_cal_fold_dict, saved_pvals_name)

    # Compute values for calibration probabilities
    # logging.debug('Computing cal probas for fold {}...'.format(fold_index))
    # probas_cal_fold, pred_proba_cal_fold = get_svm_probs(clf, X_cal)

    return {
        # Calibration credibility p values
        'cred_p_val_cal': p_val_cal_fold_dict['cred'],
        # Calibration confidence p values
        # 'conf_p_val_cal': p_val_cal_fold_dict['conf'],
        'ncms_cal': ncms_cal_fold,  # Calibration NCMs
        'pred_cal': pred_cal_fold,  # Calibration predictions
        'groundtruth_cal': groundtruth_cal_fold,  # Calibration groundtruth
        # 'probas_cal': probas_cal_fold,  # Calibration probabilities
        # 'pred_proba_cal': pred_proba_cal_fold,  # Calibration predictions
    }


def train_calibration_ice_withmodel(
        X_proper_train, X_cal,
        y_proper_train, y_cal, alg_name, fold_index, saved_data_folder, model_name):
    """Train calibration set (for a single fold).

    Quite a bit of information is needed here for the later p-value
    computation and probability comparison. The returned dictionary has
    the following structure:

        'cred_p_val_cal_fold'  -->  # Calibration credibility p values
        'conf_p_val_cal_fold'  -->  # Calibration confidence p values
        'ncms_cal_fold'        -->  # Calibration NCMs
        'pred_cal_fold'        -->  # Calibration predictions
        'groundtruth_cal_fold' -->  # Calibration groundtruth
        'probas_cal_fold'      -->  # Calibration probabilities
        'pred_proba_cal_fold'  -->  # Calibration predictions

    Args:
        X_proper_train (np.ndarray): Features for the 'proper training
            set' partition.
        X_cal (np.ndarray): Features for a single calibration set
            partition.
        y_proper_train (np.ndarray): Ground truths for the 'proper
            training set' partition.
        y_cal (np.ndarray): Ground truths for a single calibration set
            partition.
        fold_index: An index to identify the current fold (used for caching).

    Returns:
        dict: Fold results, structure as in the docstring above.

    """
    # Train model with proper training
    model_name = os.path.join(saved_data_folder, model_name)
    svm = load_cached_data(model_name)

    # Get ncms for calibration fold
    logging.debug('Getting calibration ncms for fold {}...'.format(fold_index))
    pred_cal_fold = svm.predict(X_cal)
    groundtruth_cal_fold = y_cal

    # Compute p values for calibration fold

    logging.debug('Computing cal p values for fold {}...'.format(fold_index))

    ncms_cal_fold = get_svm_ncms(svm, X_cal, y_cal)
    p_val_cal_fold_dict = compute_p_values_cred_and_conf(
        train_ncms=ncms_cal_fold,
        groundtruth_train=groundtruth_cal_fold,
        test_ncms=ncms_cal_fold,
        y_test=groundtruth_cal_fold)

    return {
        # Calibration credibility p values
        'cred_p_val_cal': p_val_cal_fold_dict['cred'],
        # Calibration confidence p values
        # 'conf_p_val_cal': p_val_cal_fold_dict['conf'],
        'ncms_cal': ncms_cal_fold,  # Calibration NCMs
        'pred_cal': pred_cal_fold,  # Calibration predictions
        'groundtruth_cal': groundtruth_cal_fold,  # Calibration groundtruth
        # 'probas_cal': probas_cal_fold,  # Calibration probabilities
        # 'pred_proba_cal': pred_proba_cal_fold,  # Calibration predictions
        'model': svm
    }


def test_with_rejection_keep_masks(
        binary_thresholds, test_scores, groundtruth_labels, predicted_labels, full=True):
    keep_mask = apply_threshold(
        binary_thresholds=binary_thresholds,
        test_scores=test_scores,
        y_test=predicted_labels)

    results = get_performance_with_rejection(
        y_true=groundtruth_labels,
        y_pred=predicted_labels,
        keep_mask=keep_mask,
        full=full)

    return results, keep_mask


def report_results(d, quiet=False):
    """Produce a textual report based on the given results.

    Args:
        d (dict): Results for baseline, kept, and rejected metrics.
        quiet (bool): Whether to also print the results to stdout.

    Returns:
        str: A textual report of the results.

    """
    report_str = ''

    def print_and_extend(report_line):
        nonlocal report_str
        if not quiet:
            print(report_line)
        report_str += report_line + '\n'

    s = '% kept elements: {:.1f}, % rejected elements: {:.1f}'.format(
        d['kept_total_perc'] * 100, d['reject_total_perc'] * 100)
    print_and_extend(s)

    s = '% benign rejected elements: {:.1f}, % malware rejected elements: {:.1f}'.format(
        d['reject_neg_total'] * 100, d['reject_pos_total'] * 100)
    print_and_extend(s)

    s = '% benign kept: {:.1f}, % benign rejected: {:.1f}'.format(
        d['kept_neg_perc'] * 100, d['reject_neg_perc'] * 100)

    print_and_extend(s)

    s = '% malware kept: {:.1f}, % malware rejected: {:.1f}'.format(
        d['kept_pos_perc'] * 100, d['reject_pos_perc'] * 100)

    print_and_extend(s)

    s = ('F1 baseline:  {:>12.2f} | '
         'F1 keep:      {:>12.2f} | '
         'F1 reject:    {:>12.2f}').format(
        d['f1_b'], d['f1_k'], d['f1_r'])

    print_and_extend(s)

    s = ('Pr baseline:  {:>12.2f} | '
         'Pr keep:      {:>12.2f} | '
         'Pr reject:    {:>12.2f}'.format(
        d['precision_b'], d['precision_k'], d['precision_r']))

    print_and_extend(s)

    s = ('Rec baseline: {:>12.2f} | '
         'Rec keep:     {:>12.2f} | '
         'Rec reject:   {:>12.2f}'.format(
        d['recall_b'], d['recall_k'], d['recall_r']))

    print_and_extend(s)

    s = ('TP baseline:  {:>12.2f} | '
         'TP keep:      {:>12.2f} | '
         'TP reject:    {:>12.2f}'.format(d['tp_b'], d['tp_k'], d['tp_r']))
    print_and_extend(s)

    s = ('FP baseline:  {:>12.2f} | '
         'FP keep:      {:>12.2f} | '
         'FP reject:    {:>12.2f}'.format(d['fp_b'], d['fp_k'], d['fp_r']))
    print_and_extend(s)

    s = ('TN baseline:  {:>12.2f} | '
         'TN keep:      {:>12.2f} | '
         'TN reject:    {:>12.2f}'.format(d['tn_b'], d['tn_k'], d['tn_r']))
    print_and_extend(s)

    s = ('FN baseline:  {:>12.2f} | '
         'FN keep:      {:>12.2f} | '
         'FN reject:    {:>12.2f}'.format(d['fn_b'], d['fn_k'], d['fn_r']))
    print_and_extend(s)

    s = ('TPR baseline: {:>12.2f} | '
         'TPR keep:     {:>12.2f} | '
         'TPR reject:   {:>12.2f}'.format(d['tpr_b'], d['tpr_k'], d['tpr_r']))
    print_and_extend(s)

    s = ('FPR baseline: {:>12.2f} | '
         'FPR keep:     {:>12.2f} | '
         'FPR reject:   {:>12.2f}'.format(d['fpr_b'], d['fpr_k'], d['fpr_r']))
    print_and_extend(s)

    return report_str


def find_random_search_thresholds_with_constraints(
        scores, predicted_labels, groundtruth_labels, maximise_vals,
        constraint_vals, max_samples=100, quiet=False, ncpu=-1):
    """Perform a random grid search to find the best thresholds on `scores` in
    parallel.

    This method wraps `find_random_search_thresholds_with_constraints_discrete`
    and parallelizes it. For a full description of this, read the documentation
    of the aformentioned method.

    See Also:
        - `find_random_search_threhsolds_with_constraint_discrete``

    Args:
        scores (dict): The test scores on which to perform the random search.
        predicted_labels (np.ndarray): The set of predictions to decide which
            'per-class' threshold to use.
        groundtruth_labels (np.ndarray): The groundtruth label for each object.
        maximise_vals: The metrics that should be maximised.
        constraint_vals: The metrics that are constrained.
        max_samples (int): The maximum number of random threshold combinations
            to try before settling for the best performance up to that point.
        quiet (bool): If True, logging will be disabled.
        ncpu (int): Number of cpus to use, if negative then we compute it as
            total_cpu + ncpu, if ncpu=1 then we do not parallelize, this is done
            to avoid problems with nested parallelization

    Returns:
        dict: Set of thresholds for malware ('gw') and goodware ('gw') classes.

    """

    ncpu = mp.cpu_count() + ncpu if ncpu < 0 else ncpu

    if ncpu == 1:
        results, thresholds = find_random_search_thresholds_with_constraints_discrete(
            scores, predicted_labels, groundtruth_labels, maximise_vals,
            constraint_vals, max_samples, quiet)

        return thresholds

    samples = [max_samples // ncpu for _ in range(ncpu)]

    with mp.Pool(processes=ncpu) as pool:
        results = pool.starmap(find_random_search_thresholds_with_constraints_discrete,
                               zip(repeat(scores), repeat(predicted_labels), repeat(groundtruth_labels),
                                   repeat(maximise_vals), repeat(constraint_vals), samples, repeat(quiet)))

        results_list = [res[0] for res in results]
        thresholds_list = [res[1] for res in results]

    def resolve_keyvals(s):
        if isinstance(s, str):
            pairs = s.split(',')
            pairs = [x.split(':') for x in pairs]
            return {k: float(v) for k, v in pairs}
        return s

    maximise_vals = resolve_keyvals(maximise_vals)
    constraint_vals = resolve_keyvals(constraint_vals)

    best_maximised = {k: 0 for k in maximise_vals}
    best_constrained = {k: 0 for k in constraint_vals}
    best_thresholds, best_result = {}, {}

    for result, thresholds in zip(results_list, thresholds_list):
        if any([result[k] > best_maximised[k] for k in maximise_vals]):
            best_maximised = {k: result[k] for k in maximise_vals}
            best_constrained = {k: result[k] for k in constraint_vals}
            best_thresholds = thresholds
            best_result = result

            if not quiet:
                logging.info('New best: {} {} @ {} '.format(
                    format_opts(maximise_vals.keys(), result),
                    format_opts(constraint_vals.keys(), result),
                    best_thresholds))
                report_results(best_result)

            continue

        if all([result[k] == best_maximised[k] for k in maximise_vals]):
            if all([result[k] >= best_constrained[k] for k in constraint_vals]):
                best_maximised = {k: result[k] for k in maximise_vals}
                best_constrained = {k: result[k] for k in constraint_vals}
                best_thresholds = thresholds
                best_result = result

                if not quiet:
                    logging.info('New best: {} {} @ {} '.format(
                        format_opts(maximise_vals.keys(), result),
                        format_opts(constraint_vals.keys(), result),
                        best_thresholds))
                    report_results(best_result)

            continue
    print(best_thresholds)
    return best_thresholds


def find_random_search_thresholds_with_constraints_discrete(
        scores, predicted_labels, groundtruth_labels, maximise_vals,
        constraint_vals, max_samples=100, quiet=False, stop_condition=3000):
    """Perform a random grid search to find the best thresholds on `scores`.

    `scores` expects a dictionary keyed by 'cred' and/or 'conf',
    with sub-dictionaries containing the thresholds for the mw and gw classes.

    Note that the keys of `scores` determine _which_ thresholding criteria will
    be enforced. That is, if only a 'cred' dictionary is supplied, thresholding
    will be enforced on cred-only and the same for 'conf'. Supplying cred and
    conf dictionaries will enforce the 'cred+conf' thresholding criteria (all
    thresholds will be applied).

    `maximise_vals` describes the metrics that should be maximised and their
    minimum acceptable values. It expects either a dictionary of metrics, or a
    string or comma separated metrics.

    `constrained_vals` describes the floors for metrics that a threshold must
    pass in order to be acceptable. The algorithm will also try to maximise
    these metrics if possible, although never at the expense of `maximise_vals`.

    Both `maximise_vals` and `constrained_vals` expect a dictionary of metrics
    and maximum acceptable values. Alternatively, arguments can be given in
    string form as comma-separated key:value pairs, for example,
    'key1:value1,key2:value2,key3:value3'.

    Concretely, any of the following are acceptable:

        > maximise_vals = {'f1': 0.95}
        > maximise_vals = 'f1_k:0.95'

        > constrained_vals = {'kept_pos_perc': 0.76, 'kept_neg_perc': 0.76}
        > constrained_vals = kept_pos_perc:0.76,kept_neg_perc:0.76

    For a list of possible metrics, see the keys in the dict produced by
    `get_performance_with_rejection()`. Note that the default objective
    function assumes that the provided metrics are in the interval [0,1].

    See Also:
        - `get_performance_with_rejection`

    Args:
        scores (dict): The test scores on which to perform the random search.
        predicted_labels (np.ndarray): The set of predictions to decide which
            'per-class' threshold to use.
        groundtruth_labels (np.ndarray): The groundtruth label for each object.
        maximise_vals: The metrics that should be maximised.
        constraint_vals: The metrics that are constrained.
        max_samples (int): The maximum number of random threshold combinations
            to try before settling for the best performance up to that point.
        quiet (bool): If True, logging will be disabled.

    Returns:
        dict: Set of thresholds for malware ('gw') and goodware ('gw') classes.

    """

    # as this method is called from multiprocessing, we want to make sure each
    # process has a different seed
    seed = 0
    for l in os.urandom(10): seed += l
    np.random.seed(seed)

    def resolve_keyvals(s):
        if isinstance(s, str):
            pairs = s.split(',')
            pairs = [x.split(':') for x in pairs]
            return {k: float(v) for k, v in pairs}
        return s

    maximise_vals = resolve_keyvals(maximise_vals)
    constraint_vals = resolve_keyvals(constraint_vals)

    best_maximised = {k: 0 for k in maximise_vals}
    best_constrained = {k: 0 for k in constraint_vals}
    best_thresholds, best_result = {}, {}

    logging.info('Searching for threshold on calibration data...')

    stop_counter = 0

    for _ in tqdm(range(max_samples)):
        # Choose and package random thresholds
        thresholds = {}
        if 'cred' in scores:
            cred_thresholds = random_threshold(scores['cred'], predicted_labels)
            thresholds['cred'] = cred_thresholds
        if 'conf' in scores:
            conf_thresholds = random_threshold(scores['conf'], predicted_labels)
            thresholds['conf'] = conf_thresholds

        # Test with chosen thresholds
        result = test_with_rejection(
            thresholds, scores, groundtruth_labels, predicted_labels)

        # Check if any results exceed given constraints (e.g. too many rejects)
        if any([result[k] < constraint_vals[k] for k in constraint_vals]):
            if stop_counter > stop_condition:
                logging.info('Exceeded stop condition, terminating calibration search...')
                break

            stop_counter += 1
            continue

        if any([result[k] < best_maximised[k] for k in maximise_vals]):
            if stop_counter > stop_condition:
                logging.info('Exceeded stop condition, terminating calibration search...')
                break

            stop_counter += 1
            continue

        if any([result[k] > best_maximised[k] for k in maximise_vals]):
            best_maximised = {k: result[k] for k in maximise_vals}
            best_constrained = {k: result[k] for k in constraint_vals}
            best_thresholds = thresholds
            best_result = result

            if not quiet:
                logging.info('New best: {} {} @ {} '.format(
                    format_opts(maximise_vals.keys(), result),
                    format_opts(constraint_vals.keys(), result),
                    best_thresholds))
                report_results(best_result)

            stop_counter = 0
            continue

        if all([result[k] == best_maximised[k] for k in maximise_vals]):
            if all([result[k] >= best_constrained[k] for k in constraint_vals]):
                best_maximised = {k: result[k] for k in maximise_vals}
                best_constrained = {k: result[k] for k in constraint_vals}
                best_thresholds = thresholds
                best_result = result

                if not quiet:
                    logging.info('New best: {} {} @ {} '.format(
                        format_opts(maximise_vals.keys(), result),
                        format_opts(constraint_vals.keys(), result),
                        best_thresholds))
                    report_results(best_result)

            stop_counter = 0
            continue

    if not bool(best_result):
        best_result = result

    return (best_result, best_thresholds)


def get_svm_probs(clf, X_in):
    """Get scores and predictions for comparison with probabilities.

    Note that this function returns the predictions _and_ probabilities given
    by the classifier and that these predictions may different from other
    outputs from the same classifier (such as `predict` or `decision_function`.
    This is due to Platt's scaling (and it's implementation in scikit-learn) in
    which a 5-fold SVM is trained and used to score the observation
    (`predict_proba()` is actually the average of these 5 classifiers).

    The takeaway is to be sure that you're always using probability scores with
    probability predictions and not with the output of other SVC functions.

    Args:
        clf (sklearn.svm.SVC): The classifier to use for the probabilities.
        X_in (np.ndarray): An array of feature vectors to classify.

    Returns:
        (list, list): (Probability scores, probability labels) for `X_in`.

    """
    assert hasattr(clf, 'predict_proba')
    probability_results = clf.predict_proba(X_in)
    probas_cal_fold = [np.max(t) for t in probability_results]
    pred_proba_cal_fold = [np.argmax(t) for t in probability_results]
    return probas_cal_fold, pred_proba_cal_fold

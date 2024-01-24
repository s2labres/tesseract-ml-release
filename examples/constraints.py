from sklearn.svm import SVC

from tesseract import temporal, metrics, mock, spatial, evaluation


# TODO | Note that constraint checks are not currently integrated into the
# TODO | evaluation cycle fit_predict_update, so need to be checked manually

def main():
    # Generate dummy predictors, labels and timestamps from Gaussians
    X, y, t = mock.generate_binary_test_data(10000, '2014', '2016')

    # Partition dataset
    splits = temporal.time_aware_train_test_split(
        X, y, t, train_size=12, test_size=1, granularity='month')

    X_train, X_tests, y_train, y_tests, t_train, t_tests = splits

    for y_test, t_test in zip(y_tests, t_tests):
        temporal.assert_positive_negative_temporal_consistency(y_test, t_test)
        temporal.assert_train_test_temporal_consistency(t_train, t_test)
        spatial.assert_class_distribution(y, 0.5, 0.1)

    # Perform a timeline evaluation
    clf = SVC(kernel='linear', probability=True)
    results = evaluation.fit_predict_update(clf, *splits)

    # View results
    metrics.print_metrics(results)

    # View AUT(F1, 24 months) as a measure of robustness over time
    print(metrics.aut(results, 'f1'))


if __name__ == '__main__':
    main()

from sklearn.svm import LinearSVC

from tesseract import evaluation, temporal, metrics, mock, viz


def main():
    # Generate dummy predictors, labels and timestamps from Gaussians
    X, y, t = mock.generate_binary_test_data(10000, '2014', '2016')

    # Partition dataset
    splits = temporal.time_aware_train_test_split(
        X, y, t, train_size=12, test_size=1, granularity='month')

    # Perform a timeline evaluation
    clf = LinearSVC()
    results = evaluation.fit_predict_update(clf, *splits)

    # View results
    metrics.print_metrics(results)

    # View AUT(F1, 24 months) as a measure of robustness over time
    print(metrics.aut(results, 'f1'))


if __name__ == '__main__':
    main()

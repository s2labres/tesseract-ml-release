from sklearn.svm import SVC

from tesseract import evaluation, temporal, metrics, mock


def main():
    # Generate dummy predictors, labels and timestamps from Gaussians
    X, y, t = mock.generate_binary_test_data(10000, '2014', '2016')

    # Partition dataset
    splits = temporal.time_aware_train_test_split(
        X, y, t, train_size=12, test_size=1, granularity='month')

    X_train, X_tests, y_train, y_tests, t_train, t_tests = splits

    # Perform a timeline evaluation
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)

    y_preds = evaluation.predict(clf, X_tests, nproc=4)
    results = metrics.calculate_metrics(y_tests, y_preds, periods=-1)

    # View results
    metrics.print_metrics(results)

    # View AUT(F1, 24 months) as a measure of robustness over time
    print(metrics.aut(results, 'f1'))


if __name__ == '__main__':
    main()

from sklearn.ensemble import RandomForestClassifier

from tesseract import temporal, mock, evaluation, metrics
from tesseract.rejection import ThresholdRejector


def main():
    X, y, t = mock.generate_binary_test_data(10000, '2000')

    splits = temporal.time_aware_train_test_split(
        X, y, t, train_size=6, test_size=1, granularity='month')

    clf = RandomForestClassifier()

    rejector = ThresholdRejector('<', 0.9)
    results = evaluation.fit_predict_update(clf, *splits, rejectors=[rejector])

    metrics.print_metrics(results)

    print('Number of rejected predictions each period:')
    print(results['rejected'])

    print('Array indices for rejected objects from first test period:')
    print(rejector.rejection_history[0])

    print('Array indices for kept objects from first test period:')
    print(rejector.kept_history[0])


if __name__ == '__main__':
    main()

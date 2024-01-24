from sklearn.svm import LinearSVC

from tesseract import temporal, mock, evaluation, metrics
from tesseract.selection import UncertaintySamplingSelector


def main():
    X, y, t = mock.generate_binary_test_data(10000, '2000')

    splits = temporal.time_aware_train_test_split(
        X, y, t, train_size=6, test_size=1, granularity='month')

    clf = LinearSVC()

    selector = UncertaintySamplingSelector('20%')
    results = evaluation.fit_predict_update(clf, *splits, selectors=[selector])

    metrics.print_metrics(results)

    print('Number of test objects selected each period:')
    print(results['selected'])

    print('Array indices for selected objects from first test period:')
    print(selector.selection_history[0])


if __name__ == '__main__':
    main()

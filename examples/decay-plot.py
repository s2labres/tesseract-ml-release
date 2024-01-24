from sklearn.svm import LinearSVC

from tesseract import evaluation, temporal, metrics, mock, viz
import os

def main():
    os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'

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
    print(metrics.aut_with_granularity(results, 'week', 'f1'))

    plt = viz.plot_decay(results)
    plt.show()


if __name__ == '__main__':
    main()

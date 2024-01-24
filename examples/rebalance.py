from sklearn.ensemble import RandomForestClassifier

from tesseract import temporal, mock, evaluation, metrics
from tesseract.rebalancing import PositiveRateRebalancer


def main():
    X, y, t = mock.generate_binary_test_data(10000, '2000')

    splits = temporal.time_aware_train_test_split(
        X, y, t, train_size=6, test_size=1, granularity='month')

    clf = RandomForestClassifier()

    pr_rebalancer = PositiveRateRebalancer(0.7, max_pos_rate=0.8, schedule='first')
    results = evaluation.fit_predict_update(
        clf, *splits, rebalancers=[pr_rebalancer])

    metrics.print_metrics(results)


if __name__ == '__main__':
    main()

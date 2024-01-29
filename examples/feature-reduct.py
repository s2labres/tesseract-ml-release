import json
import os

from sklearn.svm import LinearSVC

from tesseract import loader, temporal


def main():

    data_dir = 'DATA DIRECTORY GOES HERE'

    # Load features 
    X, y, t, _ = loader.load_features(os.path.join(data_dir, 'raw', 'extended-features', 'extended-features'))

    # Split into training and testing sets
    X_train_full, X_tests_full, y_train, y_tests, t_train, t_tests = \
        temporal.time_aware_train_test_split(X, y, t, train_size=12, test_size=1, granularity='month')

    # SelectKBest feature selection for a classifier
    clf = LinearSVC(dual="auto", max_iter=50000)
    clf.fit(X_train_full, y_train)

    select_index = loader.feature_reduce(clf=clf, dim=10000)

    with open('reduced-Indexes-10000.json', 'w') as fp:
        json.dump(select_index, fp, default=lambda x: int(x))


if __name__ == '__main__':
    main()

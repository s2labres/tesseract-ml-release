import os
import json
import datetime
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
from tesseract import evaluation, temporal, metrics, mock, viz, loader

os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'

## Loading features

def load_dataset(dataset_path):
    print(f'Loading dataset from {dataset_path}')

    with open('{}-X-updated-reduced-10k.json'.format(dataset_path), 'r') as f:
        X = json.load(f)

    print('Loading labels...')
    with open('{}-y-updated.json'.format(dataset_path), 'rt') as f:
        y = json.load(f)

    print('Loading timestamps...')
    with open('{}-meta-updated.json'.format(dataset_path), 'rt') as f:
        T = json.load(f)
    T = [o['dex_date'] for o in T]
    T = np.array([datetime.datetime.strptime(o, '%Y-%m-%dT%H:%M:%S') if "T" in o
             else datetime.datetime.strptime(o, '%Y-%m-%d %H:%M:%S') for o in T])

    # Convert to numpy array and get feature names
    vec =  DictVectorizer()
    X = vec.fit_transform(X).astype("float32")
    y = np.asarray(y)
    feature_names = vec.get_feature_names_out()

    # Get time index of each sample for easy reference
    time_index = {}
    for i in range(len(T)):
        t = T[i]
        if t.year not in time_index:
            time_index[t.year] = {}
        if t.month not in time_index[t.year]:
            time_index[t.year][t.month] = []
        time_index[t.year][t.month].append(i)

    return X, y, time_index, feature_names, T

X, y, time_index, feature_names, T = load_dataset('../extended-features/extended-features')

# Partition dataset
splits = temporal.time_aware_train_test_split(
    X, y, T, train_size=12, test_size=1, granularity='month')

# Perform a timeline evaluation
clf = LinearSVC(C=1)
results = evaluation.fit_predict_update(clf, *splits)


# ################
# View Results
# ################
from pylab import *

pendleblue='#1f8fff'
pendleyellow='#ffa600'

# '#FF9999', '#FFDD99', '#AAEEEE'
plot(results['precision'], marker='o', color=pendleyellow)
plot(results['recall'], marker='o', color='red')
plot(results['f1'], marker='o', color=pendleblue)
legend(['Precision', 'Recall', 'F1'])
xlim([0,23])
xlabel('Testing period (month)')
ylabel('Performance')
grid(axis = 'y')
show()
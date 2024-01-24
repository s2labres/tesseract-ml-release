import json, datetime
import numpy as np
from sklearn.feature_extraction import DictVectorizer

from tesseract import feature_reduction

def load_data(filepath):
    with open(filepath, 'r') as fp:
        data = json.load(fp)
    return data


def transform(X, y, meta):
    vec = DictVectorizer()
    X = vec.fit_transform(X).astype(np.int8)

    y = np.asarray(y).astype(np.int8)

    t = [o['dex_date'] for o in meta]
    t = [datetime.datetime.fromtimestamp(o) if type(o) == int \
                                            else datetime.datetime.strptime(o, '%Y-%m-%d %H:%M:%S') \
                                            for o in t]
    t = np.asarray(t)

    return X, y, t, vec


def main():
    # Load features 
    X = load_data('X.json')
    y = load_data('y.json')
    meta = load_data('meta.json')
    
    # Vectorize features
    X, y, t, vec = transform(X, y, meta)

    # SelectKBest feature selection
    select_index = feature_reduction.selectKBest(X, y, feature_size=10000)

    # OR
    # RecursiveFeatureElimination feature selection
    # select_index = feature_reduction.recursiveFeatureElimination(X, y, feature_size=10000, step=0.1)

    # Write the reduced and devectorized features to a file
    reduced_X_dict = feature_reduction.devectorize_reduce(X, select_index, vec)
    with open('reduced-X-10000.json', 'w') as fp:
        json.dump(reduced_X_dict, fp, default=lambda x: int(x))


if __name__ == '__main__':
    main()

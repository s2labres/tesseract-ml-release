# TESSERACT

As malware evolves over time, the performance of malware detectors tends to degrade. Many solutions in the security literature fail to consider the time information associated with the samples while evaluating their classifier which can induce positive bias in the results. 

This repository contains the source code for a prototype implementation of Tesseract.  

Further details can be found in the paper *TESSERACT: Eliminating Experimental Bias in Malware Classification across Space and Time*. F.  Pendlebury, F. Pierazzi, R. Jordaney, J. Kinder, and L. Cavallaro.  USENIX Sec 2019. Check also `https://s2lab.kcl.ac.uk/projects/tesseract` for up-to-date information on the project, e.g., a talk at USENIX Enigma 2019 at `https://www.usenix.org/conference/enigma2019/presentation/cavallaro`.

If you end up using Tesseract as part of a project or publication, please include a citation of the latest preprint: 

```
@inproceedings{pendlebury2019,
   author = {Feargus Pendlebury, Fabio Pierazzi, Roberto Jordaney, Johannes Kinder, and Lorenzo Cavallaro},
   title = {{TESSERACT: Eliminating Experimental Bias in Malware Classification across Space and Time}},
   booktitle = {28th USENIX Security Symposium},
   year = {2019},
   address = {Santa Clara, CA},
   publisher = {USENIX Association},
   note = {USENIX Sec}
}
```

## Getting Started 

### Installation

Tesseract requires Python 3 (preferably >= 3.5) as well as the statistical learning stack of NumPy, SciPy, and Scikit-learn. 

Create virtual environment (recommended) and install tesseract with script `setup.py`:

```shell
python3 setup.py install 
```

To download the data, run

```shell
make data
```

This should download the feature vectors and store them in
`data/processed`. An example that shows how to reproduce the experiments can be found in
`notebooks/reproduce-tesseract.ipynb`.

### Usage 

Basic usage, dividing a dataset into time-aware sets and performing a time-aware evaluation. 
More complex examples can be found in the `examples/` and `test/` directories. 

```python
from sklearn.svm import LinearSVC
from tesseract import evaluation, temporal, metrics, mock


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

```

## Running the tests 

To run all unittests within the `test/` directory: 

```shell 
python -m unittest 
```

## Current Working State 

Tesseract is still a research prototype and subject to breaking changes, although following a recent redesign we 
expect such changes to be kept to a minimum. Due to this redesign there may also be discrepancies between the current 
implementation and §6 of the Tesseract manuscript---although we are aiming to soon publish a short technical report
that details the new design. We know this can be frustrating and thank you for your patience!

If you encounter a bug or have a feature request, please feel free to contact the maintainer directly 
at `feargus.pendlebury [at] kcl.ac.uk` and cc `lorenzo.cavallaro [at] kcl.ac.uk`.


## Acknowledgements 

This project has been generously sponsored by the UK EP/L022710/1 and EP/P009301/1 EPSRC research grants.

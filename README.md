# HIF
Hybrid Isolation Forest

This is a simple package implementation for the HIF described (among other places) in this [draft paper](hif2017.pdf) for detecting anomalies and outliers from a data point distribution.

## Installation
$ sudo python3  setup.py install
 
It supports python3

## Requirements

No extra requirements are needed.

## Use


### Launching the code
$ python3 -i testHIFDonuts.py

### creating an instance of the donut dataset (normal data) and the anomaly clusters (red, green, cyan)
>>> createDonutData(contamin=.005)

### Creating the HIF
>>> computeHIF(ntrees=256, sample_size=256)

### Evaluating globally the HIF (AUC)
Outputs the best <alpha0> (HIF1) and <alpha1, alpha2> (HIF2) values
>>> plotGlobalAucBis(contamin=True)

### Evaluating globally the 1C-SVM (AUC)
>>> testOneClassSVM(NU=.1, GAMMA=.1)

### Evaluating globally the 2C-SVM (AUC)
>>> testOneClassSVM(C=.1, GAMMA=.1)

### Evaluating cluster by cluster the IF, HIF(1,2), 1C-SVM, 2C-SVM (AUC)
>>> plotDetailedResults(alpha0=.5, alpha1=.5, alpha2=.5)

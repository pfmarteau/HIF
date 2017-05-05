# HIF
Hybrid Isolation Forest

The Hybrid Isolation Forest (HIF) is an extension of the [Isolation Forest (IF) algorithm] (http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html). IF and HIF are designed for detecting anomalies and outliers from a data point distribution. As is, they are alternative methods to the one-class Support Vector Machine.

HIF integrates two extensions dedicated to
* overcome a drawback in the Isolation Forest (IF) algorithm  that limits its use in the scope of anomaly detection 
* provide it with some supervised learning capability from few samples

The HIF is described (among other places) in this [draft paper](hif2017.pdf).  

This is a simple package implementation for the HIF (inspired from this simple Python [implementation of the Isolation Forest algorithm](https://github.com/mgckind/iso_forest)).   



## Installation
(It supports python3, posssibly python2)

$ sudo python3  setup.py install
 

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

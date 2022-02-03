# DSC 180B Capstone Project

An empirical analysis of machine learning fairness using a variety of metrics, models, and datasets.

## Instructions
`python run.py` for full output matrix

`python run.py test` to run on just test data

### Adding your own data
Raw datasets sit in the `rawDatasets/` folder, and can be processed by a script in the `preprocessing/` folder. The output should add a folder to `cleanedDatasets/` containing two elements: a JSON config file (see folder for example format) and a CSV file of the cleaned dataset. Input columns should all be numerical, and output columns should be 0/1 for binary classification.


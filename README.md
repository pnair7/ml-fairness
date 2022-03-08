# Patterns of Fairness in Machine Learning

[Website Link](https://annemxu.github.io/ml-fairness/)

An empirical analysis of machine learning fairness using a variety of metrics, models, and datasets.

## Instructions
`python run.py` for full output matrix

`python run.py test` to run on just test data

### Adding your own data
Raw datasets sit in the `rawDatasets/` folder, and can be processed by a script in the `preprocessing/` folder. The output should add a folder to `cleanedDatasets/` containing two elements: a JSON config file (see folder for example format) and a CSV file of the cleaned dataset. Input columns should all be numerical, and output columns should be 0/1 for binary classification. (If your data is already cleaned to these specifications, it can be placed directly in the `cleanedDatasets` folder, no need to upload the raw dataset or use a preprocessing script.)

#### Example Config File

```
{
    "y_col": "refer",                                    # which column are you predicting?
    "X_cols": [                                          # which columns are the predictor variables?
        "dem_age_band_18-24_tm1",
        "dem_age_band_25-34_tm1",
        "dem_age_band_35-44_tm1",
        ...
        "trig_max-high_tm1",
        "trig_max-normal_tm1",
        "gagne_sum_tm1"
    ],
    "group_cols": [                                      # which column is the protected attribute? is a list for consistency, but just one element
        "race"
    ],
    "prediction_type": "binary",                         # only binary is implemented (not used)
    "dataset_name": "Obermeyer Health Dataset",          # display name for dataset
    "data_path": "rawDatasets/obermeyer_data.csv",       # path to raw dataset (not used)
    "data_script": "preprocessing/obermeyer.py"          # script to preprocess raw data (not used)
}
```

### Adding your own models or metrics
Adding your own models or metrics is a little more involved, but still quite simple. 

Currently, models are contained in `models/sklearn_models.py`. First, you must write a model function that takes in the same parameters as the other models, which can be located in the same file, or in a different file in the `models` directory. Next, in the `utils/utils.py` file, you must import the model, and add the model to the `run_models` function, assigning it a string name for the model to map to the function. Finally, you must add the string name of the model to the list of models at the top of `run.py`.

The process is nearly identical for metrics. Create a metric function that takes in the same parameters as the other metric functions, add the metric to the `apply_metric` function in `utils/utils.py` with its own string name, and add that string name to the list of metrics in `run.py`.

# Feature Robustness in Non-stationary Health Records: Caveats to Deployable Model Performance in Common Clinical Machine Learning Tasks

```
```

# About
This repository contains code to test the generalisability of benchmark models in a clinical setting.

- The environment for this code can be installed from [requirements.txt](requirements.txt).
```
conda create -n mimic_years --file requirements.txt
source activate mimic_years
```

# Usage

1) Prepare MIMIC-III data as described in https://github.com/MLforHealth/MIMIC_Extract

2) If you have permission to access the csv of patients and years of care then place them in the same directory as the flattened data. Those without permission may still reproduce the year-agnostic results. The path to the data can be saved to the DATA_DIR environment variable in [train_job](utils/train_job), [train_job_cpu](utils/train_job_cpu), and [train_no_years_job](utils/train_no_years_job).

3) Ensure that the training scripts are executable. Run them to reproduce the experiments.
```
./train_job
./train_job_cpu
./train_no_years_job
```

Experimental parameters can be fed into AUC_GH.py as arguments. They are outlined below:
* test_size, the test size for training 2001-2002 models, default=0.2
* max_time, the maximum number of hours to use in the flattened data, default=24
* random_seed, a seed for reproducible splits in the data.
* level, The column level of the multindex pandas data of which to group the data, default='itemid', choices=['itemid', 'Level2', 'nlp']
* representation, a transformation to apply to the data before training on the classification task, default='raw', choices=['raw', 'pca', 'umap', 'autoencoder']
* target, the classification task,  default='mort_icu', choices=['mort_icu', 'los_3']
* prefix, a file prefix for debugging and labelling experiments, default=""
* modeltype, select a model to train on the data, default="rf", choices=['rf', 'lr', 'svm', 'rbf-svm', 'knn', 'mlp', 'lstm', 'gru', 'grud']
* train_type, four training paradigms, default="2001-2002", choices=['2001-2002', 'rolling_limited', 'rolling', 'no_years']
* data_dir,full path to the folder containing the data
* gpu, specify which GPUS to train on, default=0
* n_threads, Number of threads to use for CPU model searches, default=maximum available

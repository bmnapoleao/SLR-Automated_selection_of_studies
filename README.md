#  Can Machine Learning Support the Selection of Studies for Systematic Literature Review Updates?
Experiment to analyze the performance of ML algorithms to automate the selection of texts during an SLR update.

## To reproduce this experiment
1) Create new python environment
```
conda create -n automated-slr python=3.8.0
```

2) Activate the new *automated-slr* environment
```
conda activate automated-slr
```

3) Install required libraries 
```
pip install -r requirements.txt
```
4) If this is the first time running the script with this conda env you need to first execute the "setup.py" file to install other packages.
```
python setup.py
```
5) Run the program, informing the number of "features" as an argument from the command line and the .env file to be used. 
Example: 
```
python main.py 500 test_config_env_files/gridsearch_chi2_fs.env 
```
By default, the results will be located at `reports/` dir inside the root dir of this project.
If you want, you can specify the output dir to storage the results file. 
Example:
```
python main.py 500 test_config_env_files/gridsearch_chi2_fs.env ~/automated-slr/results/
```

## RQ1 - _How effective are ML models in selecting studies for SLR updates?_
- To run the same test we used to asnwer this question execute the following:
```
python main.py 1200 test_config_env_files/f1_default/gridsearch_anovaf_fs.env 
```
- A .xlsx file similar to [analysis/rq1_gridsearch_anovaf_fs-k1200.xlsx](analysis/rq1_gridsearch_anovaf_fs-k1200.xlsx) will be generated with the results in the `reports/` directory.
- Run the [analysis/analysis/rq1-rf-threshold-analysis.ipynb](analysis/rq1-rf-threshold-analysis.ipynb) for more details.


## RQ2 - _How much effort can ML models reduce during the study selection activity of SLR updates?_
```
python main.py 1200 test_config_env_files/gridsearch_pearson_fs_macro.env 
```
- A .xlsx file similar to [analysis/rq2-gridsearch-pearson-fs-recall-macro-1200k.xlsx](analysis/rq2-gridsearch-pearson-fs-recall-macro-1200k.xlsx) will be generated with the results in the `reports/` directory.
- Run the [analysis/rq2-svm-threshold-analysis.ipynb](analysis/rq2-svm-threshold-analysis.ipynb) for more details.

## RQ3 - _How does the support of ML in the selection of studies compare to the support of an additional human reviewer?_
- Check the [analysis/rq3-similarity-analysis.ipynb](analysis/rq3-similarity-analysis.ipynb) to see how we performed our similarity analysis using the Euclidean Distance.

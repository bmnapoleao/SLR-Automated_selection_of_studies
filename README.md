#  Automated Selection of Studies for Systematic Literature Reviews in Software Engineering 
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
4) If this is the first time running the script, first execute the "setup.py" file to install other packages.
```
python setup.py
```

# TODO: FIX THIS PART "output-v2" and explain output options... (path/file, path, none)
5) Run the program, informing the number of "features" as an argument from the command line and the .env file to be used. 
Example: 
```
python main.py 500 test_config_env_files/gridsearch_chi2_fs.env 
```
By default, the results will be located at `output/` dir inside the root dir of this project.
If you want, you can specify the output dir to storage the results file. 
Example:
```
python main.py 500 test_config_env_files/gridsearch_chi2_fs.env ~/automated-slr/results/
```

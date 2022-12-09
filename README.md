#  Automated Selection of Studies for Sytematic Literature Reviews in Software Engineering 
Experiment to analyze the performance of ML algorithms to automate the selection of texts during an SLR update.

## To reproduce this experiment
1) Create new python environment
```
conda create -n automated-slr python=3.8.0
```

2) Activate the new *automated-slr* environement
```
conda activate automated-slr
```

3) Install required libraries 
```
pip install -r requirements.txt
```

4) Run the program, informing the number of "features" as an argument from the command line. Exemple: 
```
python main.py 500
```
 

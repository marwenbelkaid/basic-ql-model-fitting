# basic-ql-model-fitting
An example of model fitting for a basic Q-Learning (QL) algorithm based on the scikit-learn Python library.

## General description
This program implements a basic QL model in the basic\_qlestimator.py in the form of an estimator following the BaseEstimator class of the sklearn.base module. To fit this model to the data, the program supports two hyperparameter optimization procedures: Grid Search and Random Search (Bergstra and Bengio, 2012). These are respectively implemented in files modelfitting\_gridsearch.py and modelfitting\_randomsearch.py using the sklearn.model_selection module. 

Two .sh shell scripts are provided to launch the model fitting procedures.

## Workflow
1. Load data from file named data.txt (here two-dimensional)
2. Specify parameter ranges (discrete values for grid search, probability distributions for random search) and number of samples if random search
3. Get parameter sets to be tested depending on the optimization procedure
4. Run the model N times for each parameter set. Executions are parallelized accross parameter sets. For each run
	1. The estimator (QLEstimator, encapsulating the QL model) is instantiated with a given set of parameter values
	2. The fit() function runs the model and collects the results 
	3. A score() function calls the predict() function which computes the estimator's predictions given those results then computes the score. Two score() function are provided, one that computes the score based on the distance to the closest point of the data, one based on the average of all datapoints (or one specific datapoint if average_line is specified)



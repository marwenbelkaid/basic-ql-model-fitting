# basic-ql-model-fitting
An example of model fitting for a basic Q-Learning (QL) algorithm based on the scikit-learn Python library.

## Citation
[![DOI](https://zenodo.org/badge/164864013.svg)](https://zenodo.org/badge/latestdoi/164864013)

    @Misc{basic-ql-model-fitting2019,
      author = {Belkaid, Marwen},
      title  = {Code for Basic Q-Learning Model Fitting}
      doi    = {10.5281/zenodo.2564854},
      year   = {2019}
    }


## General description
This program implements a basic QL model in the basic\_qlestimator.py in the form of an estimator following the structure of BaseEstimator class of the sklearn.base module. To fit this model to the data, the program supports three hyperparameter optimization procedures: Grid Search, Random Search (Bergstra and Bengio, 2012), and Bayesian Optimization (Snoek et al., 2012). The Grid and Random Search methods are respectively implemented in files modelfitting\_gridsearch.py and modelfitting\_randomsearch.py using the sklearn.model_selection module. The Bayesian Optimization method is implemented in file  modelfitting\_bayesoptim.py and is based on the GPyOpt library (The GPyOpt authors, 2016).

Three .sh shell scripts are provided to launch the model fitting procedures.

#### Toy example:
The task that the QL algorithm has to solve here is a 3-armed bandit. For simplicity, 2 targets (or arms) have a 100% reward probability, and the has 0% probability. The measures we want to fit in the data are the success rate (i.e. % of rewards) and the U-turn rate (i.e. % of choosing the same target as at time t-2 when at time t). 

#### References:
* Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. Journal of Machine Learning Research, 13(Feb), 281-305.
* Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical bayesian optimization of machine learning algorithms. In Advances in neural information processing systems (pp. 2951-2959).
* The GPyOpt authors (2016). GPyOpt: A Bayesian Optimization framework in python. http://github.com/SheffieldML/GPyOpt

## Workflow
1. Load data from file named data.txt (here two-dimensional)
2. Specify parameter ranges (discrete values for grid search, probability distributions for random search) and number of samples if random search
3. Get parameter sets to be tested depending on the optimization procedure
4. Run the model N times for each parameter set. Executions are parallelized accross parameter sets. For each run
	1. The estimator (QLEstimator, encapsulating the QL model) is instantiated with a given set of parameter values
	2. The fit() function runs the model and collects the results 
	3. A score() function calls the predict() function which computes the estimator's predictions given those results then computes the score. Two score() function are provided, one that computes the score based on the distance to the closest point of the data, one based on the average of all datapoints (or one specific datapoint if average_line is specified)



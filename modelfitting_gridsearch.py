import numpy as np
from sklearn.model_selection import ParameterGrid
import multiprocessing
import datetime

from basic_qlestimator import QLEstimator




nb_runs		= 20

''' Define job for each worker in the multiproc computation '''
def test_sample(gridpoint):
	print "Testing estimator with sample #", gridpoint['sample_index']
	for k in range(nb_runs):
		gridpoint.update({'run': k })
		model = QLEstimator(**gridpoint)
		model.fit(dataset)
		score_a = model.score_wrt_average(dataset)
		score_c = model.score_wrt_closest(dataset)
		
		
		
''' Get dataset '''
filename = "data.txt"
dataset = np.loadtxt(filename, dtype=np.dtype("float32"), unpack=True, usecols=[1,2]) 
dataset = np.transpose(dataset)
dataset_labels = np.loadtxt(filename, dtype=np.dtype("str"), unpack=True, usecols=[0]) 


	
	
''' Set hyperparameters distributions '''
dist_tau = [1, 2, 5]
dist_alpha = [0.1, 0.5, 1.0]
param_dist = {'tau': dist_tau, 'alpha': dist_alpha}


''' Randomly sample hyperparamters '''
sample_list = list(ParameterGrid(param_dist))
for i, gridpoint in enumerate(sample_list):
	gridpoint.update({'sample_index': i })
	gridpoint.update({'average_line': 0 })	# indicate here line number corresponding to a specific datapoint you want to fit
											# this can be typically be used to avoid computing the data average at every run by manually indicating that average at line 0
											# if -1 is indicated here, then the average is computed in the score() function


''' Run parallel fit '''
nb_proc = multiprocessing.cpu_count() - 1
proc_pool = multiprocessing.Pool(nb_proc)

res = proc_pool.map_async(test_sample, sample_list)
proc_pool.close()
proc_pool.join()

	

	

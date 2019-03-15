import numpy as np
import GPy
import GPyOpt
import multiprocessing
import datetime

from basic_qlestimator import QLEstimator




nb_runs		= 2


''' Do nb_runs times '''
for k in range(nb_runs):

	''' Define job for each worker in the multiproc computation '''
	def test_sample(gridpoint):
		gridpoint_ = {'tau': gridpoint[0][0], 'alpha': gridpoint[0][1]}
		gridpoint_.update({'run': k })
		gridpoint_.update({'average_line': 0 })
		gridpoint_.update({'sample_index': 0 })
		gridpoint_.update({'log_results_internally': 0 })	# using built-in save_report() function instead
		gridpoint_.update({'log_sequences_internally': 0 })	# using built-in save_evaluations() function instead
		model = QLEstimator(**gridpoint_)
		model.fit(dataset)
		score_a = model.score_wrt_average(dataset)
		return (1 - score_a)
			
			
			
	''' Get dataset '''
	filename = "data.txt"
	dataset = np.loadtxt(filename, dtype=np.dtype("float32"), unpack=True, usecols=[1,2]) 
	dataset = np.transpose(dataset)
	dataset_labels = np.loadtxt(filename, dtype=np.dtype("str"), unpack=True, usecols=[0]) 


		
		
	''' Set optimization parameters '''
	max_iter	= 120	# number of allowed aquisitions (after initial exploration)
	max_time	= 60	# maximum exploration horizon in seconds
	eps			= 1e-3	# min diff between consecutive Xs to keep optimizing  

	nb_proc = multiprocessing.cpu_count()	# nb of cores for parallel optim


	''' Set hyperparameters domain (ranges) '''
	hparamdomain = [{'name': 'tau', 'type': 'continuous', 'domain': (1.,5.)},
					{'name': 'alpha', 'type': 'continuous', 'domain': (0.001,1.)}]



	''' Run parallel fits nb_runs times '''
	BOpt = GPyOpt.methods.BayesianOptimization(f=test_sample,           # wrap function to optimize       
												domain=hparamdomain, 	# hyperparams domain
												acquisition_type='LCB',	# Lower Confid Bound
												evaluator_type = 'local_penalization', # for parallel opt
												initial_design_numdata = nb_proc, # initial exploration
												batch_size = nb_proc,
												num_cores = nb_proc)
												
	BOpt.run_optimization(max_iter=max_iter/nb_proc, 
							eps=eps)   

	BOpt.save_evaluations("log/log_run" + "{:02d}".format(k))
	BOpt.save_report("results/results_run" + "{:02d}".format(k))

	

	

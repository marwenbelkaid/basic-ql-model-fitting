from sklearn.base import BaseEstimator
import inspect
import numpy as np
import random
import math

			
''' Define softmax function '''
def softmax(opts, tau):
	norm = 0
	ret = []
	for i in range(0,len(opts)):
		ret.append(math.exp(opts[i]/tau))
		norm += ret[i]
	for i in range(0,len(opts)):
		ret[i] /= norm
	return ret
	
''' Define function for random draw form discrete proba dist '''
def draw_from_discrete_proba(dist):
	d = random.uniform(0, 1) 
	cumul = 0
	for i in range(0, len(dist)):
		if d < dist[i] + cumul:
			return i
		cumul += dist[i]
		


		
''' Define QL algorithm for this problem '''
def rl_3target_vc(alpha, tau, max_steps):
	''' Define states, actions and reward probabilities '''
	IN_A = GOTO_A = 0
	IN_B = GOTO_B = 1
	IN_C = GOTO_C = 2

	states = [IN_A, IN_B, IN_C]
	actions = [GOTO_A, GOTO_B, GOTO_C]
	labels = ["A", 'B', "C"]

	RW_A = 1.
	RW_B = 0.
	RW_C = 1.

	rewards = [RW_A, RW_B, RW_C]

	''' Define output data variables '''
	out_choices = []		# append choices 
	str_choices = ""		# useful for vc
	out_vcrw = []			# append rewards given by vc

	''' Intialize Q table '''
	init_Qval = 1.1
	Q = np.full( (len(states), len(actions)), init_Qval ) # optimistic initialization


	''' Start simulation '''
	s = IN_A
	step = 0
	while step < max_steps:
		''' Act using softmax policy '''
		a = s
		opts = np.copy(Q[s][:])
		opts[s] = -99
		dist = softmax(opts[:], tau)
		while a==s:	# try until GOTO != s, because agent cannot choose the same target in two consecutive trials
			a = draw_from_discrete_proba(dist)
		
		''' Get reward: GOTO_X -> RW_X '''
		draw = random.uniform(0, 1) 
		if draw < rewards[a]:
			r = 1
		else:
			r = 0
			
		''' Update Q table '''
		delta = alpha*( r - Q[s][a] ) # gamma = 0
		Q[s][a] += delta
		
		''' Update state '''
		s = a 	
		
		''' Update loop variable '''
		step += 1
		
		''' Save output data '''
		out_choices.append(a)
		str_choices = str_choices + str(a)
		if r <= 0:
			out_vcrw.append(0)
		else:
			out_vcrw.append(1)
	
	return [out_choices, out_vcrw]
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
class QLEstimator(BaseEstimator):  
	""" 
	This is my toy example of a Q-Learning estimator using softmax policy 
	It has a constructor and it inherits the get_params and set_params methods from BaseEstimator
	It overrides the methods fit(), predict() and score()
	"""
	
	def __init__(self, alpha=0.1, tau=5.0, average_line=-1, sample_index=-1, run=1, log_results_internally=1, log_sequences_internally=1):
		""" 
		This is the constructor 
		
		Parameters
		----------
		alpha, tau : float
			QL parameters, resp. learning rate and softmax temperature
		average_line : int
			if one of the line in the dataset corresponds to the average or has to be fitted, 
			otherwise -1 means the average will be computed online
		sample_index : int
			index of current estimator, the default value -1 should be overriden
		run : int
			index of run (if many executions per paramset)
		log_results_internally, log_sequences_internally : int
			flags indicating whether to save log and result files within estimator
			this is useful when using Grid or Random Search Optimization methods 
			however Bayesian Optimization has built-in log functions
		

		"""
		
		''' Initialize the estimator with contructor arguments '''
		args, _, _, values = inspect.getargvalues(inspect.currentframe())
		values.pop("self")
		for arg, val in values.items():
			setattr(self, arg, val)
		
		''' Add hard-coded parameters '''
		self.max_steps = 10000			# number of trials per run
		self.lensession = 200			# number of trials in one session 
		self.nb_test_sessions = 10		# number of sessions at the end of run of which the average is considered as the final results of the run
		

		''' Print attributes '''
		#for arg, val in values.items():
			#print("{} = {}".format(arg,val))
			
			
	def fit(self, X, y=None):
		""" 
		This is where the QL is run and the results are saved 
		"""
		
		''' Check parameters '''
		self.__check_params(X)
		
		''' Run QL '''
		out_choices, out_vcrw = rl_3target_vc(self.alpha, self.tau, self.max_steps)
		
		''' Set attributes '''
		self.rw_ = out_vcrw
		self.s_ = out_choices
		
		''' Save data '''
		if self.log_sequences_internally == 1:
			datafile = "log/fit_" + str(self.sample_index) + "_run_" + str(self.run)
			data_to_save = np.transpose((out_choices, out_vcrw))
			np.savetxt(datafile, data_to_save, fmt='%d')
			
		
	def predict(self, X, y=None):
		""" 
		This is where the output data is predicted.
		Here, it amounts to evaluate the success rate, u-turn rate, and complexity of the QL agent.
		"""
		try:
			getattr(self, "rw_")
			getattr(self, "s_")
		except AttributeError:
			raise RuntimeError("You must train estimator before predicting data!")
		# use check_is_fitted(self, ['X_', 'y_']) instead ?
			
		''' Get average result over nb_test_sessions last sessions '''
		max_steps = self.max_steps
		lensession = self.lensession
		nb_test_sessions = self.nb_test_sessions
		
		mean_rw_rate = 0
		mean_ut_rate = 0
		
		for i in range(0, nb_test_sessions):
			last_step = max_steps - i*lensession
			first_step = last_step - lensession 
		
			rw_in_session = self.rw_[first_step:last_step]
			s_in_session = self.s_[first_step:last_step]
			
			# rw
			nb_rw = np.bincount(rw_in_session)[1]
			rw_rate = nb_rw * 1. / lensession
			# ut
			nb_ut = 0 
			for k in range(2, lensession):
				if s_in_session[k] == s_in_session[k-2]:
					nb_ut += 1
			ut_rate = nb_ut * 1. / (lensession-2)
			
			mean_rw_rate += rw_rate
			mean_ut_rate += ut_rate
			
		mean_rw_rate /= nb_test_sessions
		mean_ut_rate /= nb_test_sessions
		
		return [mean_rw_rate, mean_ut_rate]	
			
		
		
	def score_wrt_average(self, X, y=None):
		""" 
		This is where the scor eof the estimator wrt the average is computed.
		Here, the score is the distance between prediction and data
		"""	
		score = 0
		
		pred = self.predict(X)
		
		if self.average_line == -1:
			nb_datapoints = len(X)
			data0 = 0.
			data1 = 0.
			for i in range(0, nb_datapoints):
				data0 += X[i][0]
				data1 += X[i][1]
			data0 /= (nb_datapoints * 100.)	# data given as percentage but score measured based on rates		
			data1 /= (nb_datapoints * 100.)	# data given as percentage but score measured based on rates			
		else:
			data0 = X[self.average_line][0] / 100. # data given as percentage but score measured based on rates
			data1 = X[self.average_line][1] / 100. # data given as percentage but score measured based on rates
		
		pred0 = pred[0] 
		pred1 = pred[1] 
		
		score = 1 - ( (abs(pred0 - data0) + abs(pred1 - data1)) / 2. )
		
		if self.log_results_internally == 1:
			logfile = "results/score_a_fit_" + str(self.sample_index) + "_run_" + str(self.run)
			with open(logfile,'a') as outfp:
				data_to_save = '{:4.3f} -1 {:4.3f} {:06.3f} -1 {:4.3f} {:4.3f} {:4.3f} {:4.3f}\n'.format(score, self.alpha, self.tau, data0, data1, pred0, pred1)
				outfp.write(data_to_save)
		
		return score
			
	
	def score_wrt_closest(self, X, y=None):		
		
		closest_score = 0
		closest_index = -1
		
		pred = self.predict(X)
		pred0 = pred[0] 
		pred1 = pred[1] 
		
		nb_datapoints = len(X)
		
		for i in range(0, nb_datapoints):
			if i != self.average_line :
				data0 = X[i][0] / 100.
				data1 = X[i][1] / 100.
				score = 1 - ( (abs(pred0 - data0) + abs(pred1 - data1)) / 2. )
				
				if score > closest_score:
					closest_score = score
					closest_index = i
		
		if self.log_results_internally == 1:
			logfile = "results/score_c_fit_" + str(self.sample_index) + "_run_" + str(self.run)
			with open(logfile,'a') as outfp:
				data_to_save = '{:4.3f} {:d} {:4.3f} {:06.3f} {:d} {:4.3f} {:4.3f}\n'.format(closest_score, closest_index, self.alpha, self.tau, closest_index, pred0, pred1)
				outfp.write(data_to_save)
			
	
	
	
	
	
	
	
	
	
	
	
	
	
	"""
	These are private methods
	"""
	
	def __check_params(self, X):
		""" This is called by fit() to check parameters """
		
		if (self.alpha - 1 > 0.0000001) or (self.alpha - 0 < 0.0000001):
			print "Error: Invalid value for parameter alpha given to {0}. Value must be in [0,1].".format(self)
			exit() 
		if (self.tau - 0 < 0.0000001):
			print "Error: Invalid value for parameter tau given to {0}. Value must be > 0".format(self)
			exit() 
		if (self.sample_index == -1) :
			print "Error: Invalid value for parameter sample_index given to {0}. Default value (-1) should be overriden with a positive value.".format(self)
			exit() 
			
			
			
			

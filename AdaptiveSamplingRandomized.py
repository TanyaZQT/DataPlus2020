import numpy as np
from scipy.interpolate import UnivariateSpline
import random

"""
Returns: 
	data: numpy array (dim = (NUM_SAMPLES, 2*f_max*INTERP_FREQ*length)) containing signals
	freq: numpy array (dim = NUM_SAMPLES, MAX_NUM_COMPONENTS) containing frequencies
"""

def randomize(f_max, length, INTERP_FREQ = 12, NUM_SAMPLES = 1000, MAX_NUM_COMPONENTS = 25):
	
	samples_per_second = 2*f_max # sample at 2*f_max Hz
	pts_per_second = samples_per_second*INTERP_FREQ # total signal points per second

	total_interp_pts = pts_per_second*length # number of interpolation points over the signal length

	# uniformly spaced interval with total_interp_pts points
	signal_interval = np.linspace(0, length, num=total_interp_pts, endpoint='False') 

	# initialize the data
	data = np.zeros((NUM_SAMPLES, len(signal_interval)))

	# array containing the frequencies of the components of all signals
	freq = np.zeros((NUM_SAMPLES, MAX_NUM_COMPONENTS))

	for i in range(NUM_SAMPLES):

		# generate random number of components in our signal
		num_components = np.random.randint(low = 1, high = MAX_NUM_COMPONENTS)

		# random coefficients for components chosen uniformly from [-1, 1]
		coefficients = 2*np.random.uniform(size=num_components)-np.ones(num_components)

		# random frequencies for components chosen uniformly from [0, f_max]
		freq[i, 0:num_components] = np.floor(f_max*np.random.uniform(size=num_components))

		# add constant epsilon to avoid division by zero
		epsilon = 2*num_components*(np.max(np.absolute(coefficients)))

		# initialize signal to begin accumulation
		signal = epsilon*np.ones((1, len(signal_interval)))

		for j in range(num_components):

			# add generated component to signal
			new_component = coefficients[j]*np.sin(2*np.pi*freq[i,j]*signal_interval)

			signal = signal + new_component

		# put signal into data
		data[i, :] = signal

	return data, freq


"""
Returns: 
	signal_1: numpy array (dim = 1, len(data[0]))
	signal_2: numpy array (dim = 1, len(data[0])) (signal_1 != signal_2)

	fmax_1, fmax_2: duple containing the max frequencies for each signal
"""

def choose_two(data, freq):

	# generate two random distinct integers 
	index_1, index_2 = random.sample(range(0, len(data)), 2) 
	
	# get the signals at these indices
	signal_1 = data[index_1, :]
	signal_2 = data[index_2, :]

	# get the max frequencies for each signal
	fmax_1 = np.max(freq[index_1, :])
	fmax_2 = np.max(freq[index_2, :])

	return signal_1, signal_2, fmax_1, fmax_2

"""
Returns: 
	perturb_signal: numpy array (dim = dim(original_locations))

	Perturbs an interval of uniform sampling by up to half of the original spacing
"""

def perturb(original_locations, uniform_spacing):

	# creates len(original_locations) values according to Unif(0, uniform_spacing/2)
	perturb = ((uniform_spacing//2)*np.random.uniform(size=len(original_locations))).astype(int)

	return (original_locations + perturb).astype(int)

"""
Returns: 
	RE: float containing the relative error between Gtr_m and Spl_m
	SE: float containing the squared error between Gtr_m and Spl_m
"""

def forward_pass(Gtr_m, Spl_m, CUTOFF = 15):

	# Trims each signal CUTOFF points from the front and back
	Trim_Gtr_m = Gtr_m[CUTOFF:len(Gtr_m)-CUTOFF]
	Trim_Spl_m = Spl_m[CUTOFF:len(Spl_m)-CUTOFF]

	# Computes the difference signal
	Dq = np.absolute(Trim_Spl_m-Trim_Gtr_m)

	# Computes the relative error signal
	RE_signal = np.divide(Dq, Trim_Gtr_m)

	# Computes the squared error signal
	SE_signal = np.multiply(Dq, Dq)

	# Finds the mean of RE_signal (average relative error)
	RE = np.mean(RE_signal)

	# Finds the total squared error
	SE = np.sum(SE_signal)

	return RE, SE

f_max, length, interp_freq = 20, 5, 10
# randomize signals and get their frequency components
data, freq = randomize(f_max, length, INTERP_FREQ = interp_freq)

# sample two of the signals and compute the fmax for each
sig_1, sig_2, fmax_1, fmax_2 = choose_two(data, freq)


# compute the uniform spacing for each signal (inversely related to max frequency)
spacing_1 = int(np.ceil(interp_freq*f_max/fmax_1))
spacing_2 = int(np.ceil(interp_freq*f_max/fmax_1))

# create uniform sampling intervals for each (numpy arrays)
uniform = np.arange(0, len(data[0]))

interval_1 = uniform[::spacing_1]
interval_2 = uniform[::spacing_2]

# get values at sampled locations
sig_values_1 = sig_1[::spacing_1]
sig_values_2 = sig_2[::spacing_1]

# get interpolated signals
spl_1 = UnivariateSpline(interval_1, sig_values_1)(uniform)
spl_2 = UnivariateSpline(interval_2, sig_values_2)(uniform)

# multiply the interpolated signals and ground truth signals together
Spl_m = np.multiply(spl_1, spl_2)
Gtr_m = np.multiply(sig_1, sig_2)

# compute the energy of the sliced signal
cutoff = 15
signal_energy = np.sum((Gtr_m[cutoff:len(Gtr_m)-cutoff])**2)

# Initial Forward Pass
RE_init, SE_init = forward_pass(Gtr_m, Spl_m)

# establish best values for future passes
best_points = interval_2
best_RE = RE_init
best_EE = SE_init/signal_energy

# iterations to find the best sampling points
NUM_ITER = 1000
for i in range(NUM_ITER):

	# generate new locations through perturbation
	new_sample_loc = perturb(interval_2, spacing_2)

	# get signal values at new sample locations
	new_values = sig_2[new_sample_loc]

	# interpolate for new spline signal
	spl_new = UnivariateSpline(new_sample_loc, new_values)(uniform)

	# multiply signals together
	Spl_m_new = np.multiply(spl_new, spl_1)

	# forward pass with new spline signal
	next_RE, next_SE = forward_pass(Gtr_m, Spl_m_new)

	# Check if either makes an improvement
	if next_RE < best_RE:

		# update values
		best_points = new_sample_loc
		best_RE = next_RE
		best_EE = next_SE/signal_energy

print(RE_init, best_RE)
print(RE_init-best_RE)

print(SE_init/signal_energy, best_EE)
print(SE_init/signal_energy-best_EE)

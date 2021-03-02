"""Random particle position and velocity functions"""

import numpy as np

def uniform_position(x_min, x_max, y_min, y_max):
	"""Random uniform position across domain"""
	return (np.random.random()*(x_max-x_min) + x_min,
			np.random.random()*(y_max-y_min) + y_min)

def uniform_velocity(v_min, v_max):
	"""Random uniform velocity"""
	return (np.random.random()*(v_max-v_min) + v_min,
			np.random.random()*(v_max-v_min) + v_min)

def uniform_velocity_alt(v_max):
	"""Alternate random uniform velocity in speed and angle"""
	th = np.random.random()*2*np.pi
	v = np.random.random()*v_max
	return (v*np.cos(th), v*np.sin(th))

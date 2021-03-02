"""This module contains data io classes to load/save simulation results in HDF5

.. moduleauthor:: C. S. Arridge <c.arridge@lancaster.ac.uk>
.. py:module:: Io

Effectively, the io classes here are slim wrappers around ``h5py``.
"""

# Standard Python modules.
import json
import platform
import datetime
import uuid

# Standard SciPy modules.
import numpy as np
import scipy

# Others.
import h5py
import Modules

class Output:
	"""Main output class

	The class provides the facility to control the frequency at which particular
	data sets are written to file and also the formatting of those datasets.
	The class also automatically writes run meta data (in the top level HDF5
	attributes) to the file.

	Meta data includes:
	* Start time of the run.
	* Platform and processor the run was carried out on.
	* Version information for Python, NumPy, SciPy and the model version.
	* A unique ID code for the run.

	The HDF5 file is structured as follows:
	/log/* - time series quantities, such as step number, divJ, e.g., /log/divJ
	/fields/* - field quantities, such as Ex, e.g., /fields/Ex
	/all_particles/{}/* - all particles further grouped into folders based on the step

	Quantities in /log/ are just simple 1D arrays.  Fields in /fields/ are 3D
	arrays of (index,x,y).  Particles in /all_particles are stored with another
	folder to indicate the step number, e.g., if all particles were written
	on step 11 then there would be a folder /all_particles/00011/, then there
	would be 6 1D datsets for x, y, vx, vy, q, m, e.g., /all_particles/00011/vx.

	The log and field quantities are stored with a particular default length for
	efficiency, so that the storage doesn't need to dynamically expand on every
	write.  When the code tries to store more than this number the storage is
	extended to a larger size.  At the end of the run, a call to
	``Output.close`` will resize all the datasets to remove unused space at the
	end.

	The parameters write_fields, write_all_particles, write_steps are variables
	that store the frequency at which these should be written.  So if
	write_fields is set to 10, every tenth call to ``ready_fields`` will
	return True to indicate that the user should write field data to the file.
	"""

	def __init__(self, nx, ny, log_size=10000, field_size=1000):
		"""Constructor.

		Arguments:
		:param: nx int: Number of grid points in x.
		:param: ny int: Number of grid points in x.
		:param: log_size int: Default number of log entries.
		:param: field_size int: Default number of field entries.
		"""
		# These are the options for how frequently different types of data
		# are written.
		self._write_fields = None
		self._write_all_particles = None
		self._write_tracer_particles = None
		self._write_steps = 1

		# These hold the number of entries that are pre-allocated in the
		# data file; at the end of the run, the datasets will need resizing
		# to remove the empty datasets.
		self._n_log_entries = log_size
		self._n_field_entries = field_size

		# Store the grid size and maximum number of particles.
		self._nx = nx
		self._ny = ny

		# These hold the various log variables (e.g., CFL, divJ) that are
		# written and their datatypes.  The _d_log_vars dictionary maps
		# log variable labels to their datasets in the HDF5 file.
		self._d_log_vars = {}
		self._d_log_dtypes = {}
		self.add_log_quantity('step', np.int32)
		self.add_log_quantity('time', np.float32)
		self.add_log_quantity('cfl', np.float32)
		self.add_log_quantity('divJ', np.float64)
		self.add_log_quantity('divB', np.float64)
		self.add_log_quantity('gauss_law', np.float64)
		self.add_log_quantity('num_particles', np.int32)

		# These hold the various field variables (e.g., Ex, Ux) that are
		# written and their datatypes.  The _d_field_vars dictionary maps
		# field variable labels to their datasets in the HDF5 file.
		self._d_field_vars = {}
		self._d_field_dtypes = {}
		self.add_field_quantity('Ex', np.float64)
		self.add_field_quantity('Ey', np.float64)

		# This is the hande to the actual HDF5 file.
		self._h5file = None

	def open(self, filename='data.hdf5'):
		"""Open the output file and setup the structure

		Arguments:
		:param: filename str: Filename to write data to.
		"""
		self._h5file = h5py.File(filename, 'w')

		# Set meta data.
		self._h5file.attrs['date'] = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
		self._h5file.attrs['platform'] = platform.platform()
		self._h5file.attrs['processor'] = platform.processor()
		self._h5file.attrs['python_version'] = platform.python_version()
		self._h5file.attrs['numpy_version'] = np.__version__
		self._h5file.attrs['scipy_version'] = scipy.__version__
		self._h5file.attrs['hyb_version'] = Modules.__version__
		self._h5file.attrs['nx'] = self._nx
		self._h5file.attrs['ny'] = self._ny
		self._h5file.attrs['id'] = uuid.uuid4().__str__()

		# Setup the groups.
		self._g_fields = self._h5file.create_group('fields')
		self._g_all_particles = self._h5file.create_group('all_particles')
		self._g_tracer_particles = self._h5file.create_group('tracer_particles')
		self._g_log = self._h5file.create_group('log')
		self._g_grid = self._h5file.create_group('grid')

		# Create log variables.
		for k in self._d_log_vars:
			self._d_log_vars[k] = self._g_log.create_dataset(k, (self._n_log_entries,), chunks=(10,), dtype=self._d_log_dtypes[k])

		# Create field variables.
		for k in self._d_field_vars:
			self._d_field_vars[k] = self._g_fields.create_dataset(k, (self._n_field_entries,self._nx+1,self._ny+1), chunks=(1,self._nx+1,self._ny+1), dtype=self._d_field_dtypes[k])

		# Indices
		self._index_log = 0
		self._index_fields = 0

	def close(self):
		"""Finalise the file by resizing all the arrays."""
		# Resize arrays.
		for k in self._d_log_vars:
			self._d_log_vars[k].resize((self._index_log,))
		for k in self._d_field_vars:
			self._d_field_vars[k].resize((self._index_fields,self._nx+1,self._ny+1))

	def store_grid(self, x, y, xmesh, ymesh):
		"""Store the grids used in the simulation run.

		Arguments:
		:param: x np.ndarray: Linear array of x points.
		:param: y np.ndarray: Linear array of y points.
		:param: xmesh np.ndarray: 2D array of x coords from ``numpy.meshgrid``
		:param: ymesh np.ndarray: 2D array of y coords from ``numpy.meshgrid``
		"""
		self._g_grid.create_dataset('x', data=x)
		self._g_grid.create_dataset('y', data=y)
		self._g_grid.create_dataset('xmesh', data=xmesh)
		self._g_grid.create_dataset('ymesh', data=ymesh)

	def store_particle_dataset(self, step, plist):
		"""Write out the entire particle dataset for a given step.

		Arguments:
		:param: step int: Step number (used to name the HDF5 group).
		:param: plist list: List of particle objects.
		"""
		# Create the group and the datasets.
		tmp = self._g_all_particles.create_group('{:06d}'.format(step))
		data_x = tmp.create_dataset('x', (len(plist),), dtype=np.float64)
		data_y = tmp.create_dataset('y', (len(plist),), dtype=np.float64)
		data_vx = tmp.create_dataset('vx', (len(plist),), dtype=np.float64)
		data_vy = tmp.create_dataset('vy', (len(plist),), dtype=np.float64)
		data_q = tmp.create_dataset('q', (len(plist),), dtype=np.float32)
		data_m = tmp.create_dataset('m', (len(plist),), dtype=np.float32)

		# Store all the particle information.
		for i in range(len(plist)):
			data_x[i] = plist[i].x
			data_y[i] = plist[i].y
			data_vx[i] = plist[i].vx
			data_vy[i] = plist[i].vy
			data_q[i] = plist[i].q
			data_m[i] = plist[i].m

	def store_log(self, label, v):
		"""Write out a log value for a particular dataset.

		Arguments:
		:param: label str: Label of the quantity to write to, e.g., 'divJ'
		:param: v: Value to store.
		"""
		self.log[label][self._index_log] = v

	def store_field(self, label, v):
		"""Write out a field value for a particular dataset.

		Arguments:
		:param: label str: Label of the quantity to write to, e.g., 'Ex'
		:param: Ex np.ndarray: Array to store.
		"""
		self._d_field_vars[label][self._index_fields,:,:] = v

	def add_log_quantity(self, label, type):
		"""Add a quantity to log in the file.

		Arguments:
		:param: label str: Label to describe the quantity.
		:param: type np.dtype: NumPy type to store.
		"""
		self._d_log_vars[label] = 'None'
		self._d_log_dtypes[label] = type

	def add_field_quantity(self, label, type):
		"""Add a field quantity to store in the file.

		Arguments:
		:param: label str: Label to describe the quantity.
		:param: type np.dtype: NumPy type to store.
		"""
		self._d_field_vars[label] = 'None'
		self._d_field_dtypes[label] = type

	def set_author(self, v):
		"""Set the author of this run

		Arguments:
		:param: v str: Author.
		"""
		self._h5file.attrs['author'] = v

	def set_description(self, v):
		"""Set the description for this run

		Arguments:
		:param: v str: Description.
		"""
		self._h5file.attrs['desc'] = v

	def ready_steps(self, step):
		"""Check if we need to write a step log entry

		Arguments:
		:param: step int: Step number.
		:returns: bool: True if the step log entry needs writing.
		"""
		if self._write_steps is not None:
			if (step%self._write_steps)==0:
				return True
			else:
				return False
		else:
			return False

	def ready_fields(self, step):
		"""Check if we need to write fields to the output file.

		Arguments:
		:param: step int: Step number.
		:returns: bool: True if the fields need writing.
		"""
		if self._write_fields is not None:
			if (step%self._write_fields)==0:
				return True
			else:
				return False
		else:
			return False

	def ready_all_particles(self, step):
		"""Check if we need to write all the particles to the output file.

		Arguments:
		:param: step int: Step number.
		"""
		if self._write_all_particles is not None:
			if (step%self._write_all_particles)==0:
				return True
			else:
				return False
		else:
			return False

	def ready_tracer_particles(self, step):
		"""Check if we need to write tracer particles to the outut file.

		Arguments:
		:param: step int: Step number.
		:returns: bool: True if the tracer particles need writing.
		"""
		if self._write_tracer_particles is not None:
			if (step%self._write_tracer_particles)==0:
				return True
			else:
				return False
		else:
			return False

	def next_log(self):
		"""Move to the next writable log position and enlarge the storage if necy"""
		self._index_log += 1
		if self._index_log==self._n_log_entries:
			self._enlarge_logs()

	def next_fields(self):
		"""Move to the next writable field position and enlarge the storage if necy"""
		self._index_fields += 1
		if self._index_fields==self._n_field_entries:
			self._enlarge_fields()


	## Enlarge array methods ###################################################
	def _enlarge_logs(self, delta=100):
		"""Make the log entries larger by a certain delta."""
		self._n_log_entries += delta
		for k in self._d_log_vars:
			self._d_log_vars[k].resize((self._n_log_entries,))

	def _enlarge_fields(self, delta=10):
		"""Make the field entries larger by a certain delta."""
		self._n_field_entries += delta
		for k in self._d_field_vars:
			self._d_field_vars[k].resize((self._n_field_entries,self._nx+1,self._ny+1))


	## Get/set property methods ################################################
	@property
	def h5(self):
		return self._h5file

	@property
	def write_fields(self):
		return self._write_fields

	@write_fields.setter
	def write_fields(self, v):
		if v>0:
			self._write_fields = int(v)
		else:
			raise ValueError('Field write frequency must be greater than zero')

	@property
	def write_all_particles(self):
		return self._write_all_particles

	@write_all_particles.setter
	def write_all_particles(self, v):
		if v>0:
			self._write_all_particles = int(v)
		else:
			raise ValueError('All particle write frequency must be greater than zero')

	@property
	def write_tracer_particles(self):
		return self._write_tracer_particles

	@write_tracer_particles.setter
	def write_tracer_particles(self, v):
		if v>0:
			self._write_tracer_particles = int(v)
		else:
			raise ValueError('Tracer particle write frequency must be greater than zero')

	@property
	def write_steps(self):
		return self._write_steps

	@write_steps.setter
	def write_steps(self, v):
		if v>0:
			self._write_steps = int(v)
		else:
			raise ValueError('Step log frequency must be greater than zero')

	@property
	def attrs(self):
		return self._h5file.attrs

	@property
	def log(self):
		return self._d_log_vars

	@property
	def ilog(self):
		return self._index_log

	@property
	def ifields(self):
		return self._index_fields

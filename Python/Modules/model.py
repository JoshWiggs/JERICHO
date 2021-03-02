#Import python native modules
import random as ran
import timeit
import math
import pickle

#Import 3rd party python modules
import numpy as np
import scipy.constants as con
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#Import custom models
from Modules.numerical_differentials import *
from Modules.functions import *
from Modules.classes import Particle_Information as Par
from Modules.classes import Particle_Grid_Information as Grid_Info
import Modules.io
import Modules.rparticle
import Modules

class Hyb2D:
	def __init__(self):
		print('[Hyb2D] Version {} ({})'.format(Modules.__version__, Modules.__date__))
		print('[Hyb2D] '+Modules.__copyright__+'\n')
		print('[Hyb2D] Initialising simulation')
		self.R_P = None
		self.M_P = None
		self.Omega = None
		self.B_eq = None
		self.Periodic = None
		self.Open = None

		#Spatial domain variables in SI units
		self.x_min = None		#Minimum position in x domain
		self.x_max = None		#Maximum position in x domain
		self.y_min = None		#Minimum position in y domain
		self.y_max = None		#Maximum position in y domain
		self.nx = None			#Number of steps taken from x_min to x_max
		self.ny = None			#Number of steps taken from y_min to y_max
		self.dx = None
		self.dy = None
		self.dxdy = None
		self.T_c = None
		self.x = None
		self.y = None
		self.X = None
		self.Y = None

		# Minimum and maximum particle speeds
		self.v_min = None
		self.v_max = None

		#Time Variables
		self.t_min = None #Start time
		self.t_max = None #End time
		self.nt = None #Number of time steps
		self.dt = None
		self.t = None

		#Particle charge and mass lists.
		self.q_c = [] #particle charge
		self.m = [] #particle mass
		self.qm_choice = []

		#Particle numbers and generators.
		self.PPC = None
		self.T_PPC = None
		self.random_particle_position = None
		self.random_particle_velocity = None

		# Random seed.
		self.seed = 921512

		# smoothing
		self.n_smoothing = None

		# Backgroud magnetic and electric fields.
		self.config_Bz_back = None
		self.config_Ex_back = None
		self.config_Ey_back = None

		# Data output variables
		self.data_filename = 'data.hdf5'
		self.data_write_steps = 1
		self.data_write_all_particles = 100
		self.data_write_fields = 10

		self.config_parameters()
		self.sanity_check()
		np.random.seed(self.seed)
		self.setup_grid()
		self.dxdy = self.dx*self.dy
		self.T_PPC = int(self.T_c * self.PPC) #Total number of particles in simulation
		self.initial_conditions()

		print('[Hyb2D] Order of time-step: 10^{}s'.format(orderOfMagnitude(self.dt)))
		print('[Hyb2D] Order of grid spacing: 10^{}m'.format(orderOfMagnitude(self.dx)))
		print('[Hyb2D] Number of particles: {}'.format(self.T_PPC))
		print('[Hyb2D] Number Density: {}m^-2'.format((self.T_PPC/((self.x_max-self.x_min)*(self.y_max-self.y_min)))))

		self.data = Modules.io.Output(self.nx, self.ny)
		self.data.write_steps = self.data_write_steps
		self.data.write_all_particles = self.data_write_all_particles
		self.data.write_fields = self.data_write_fields
		self.data.open(self.data_filename)
		self.data.attrs['R_P'] = self.R_P
		self.data.attrs['M_P'] = self.M_P
		self.data.attrs['Omega'] = self.Omega
		self.data.attrs['x_min'] = self.x_min
		self.data.attrs['x_max'] = self.x_max
		self.data.attrs['y_min'] = self.y_min
		self.data.attrs['y_max'] = self.y_max
		self.data.store_grid(self.x, self.y, self.X, self.Y)

	def cleanup(self):
		self.data.close()

	def setup_grid(self):
		self.dx = (self.x_max - self.x_min) / self.nx #Size of step in x domain
		self.dy = (self.y_max - self.y_min) / self.ny #Size of step in y domain
		self.T_c = self.nx * self.ny #Total number of cells
		self.v_min = (self.dx/2.) #Minimum initial particle speed
		self.v_max = (self.dx) #Maximum initial particle speed

		#Populate dimesnional vectors
		self.x = np.linspace(self.x_min,self.x_max,num=(self.nx+1))
		self.y = np.linspace(self.y_min,self.y_max,num=(self.ny+1))

		#Meshgrid for plotting
		self.X,self.Y = np.meshgrid(self.x,self.y)

		#Electromagentic field variables
		#TODO: Write field classes and add functionality such as internal magnitude calulations
		self.E_x_n = np.zeros(((self.nx + 1),(self.ny + 1))) #Electic field in x domain on current time step
		self.E_y_n = np.zeros(((self.nx + 1),(self.ny + 1))) #Electic field in y domain on current time step
		self.E_x_back = np.zeros(((self.nx + 1),(self.ny + 1))) #Background electric field in the x domain
		self.E_y_back = np.zeros(((self.nx + 1),(self.ny + 1))) #Background electric field in the y domian
		self.B_z_n = np.zeros(((self.nx + 1),(self.ny + 1))) #Magnetic field in z domain on current time step
		self.B_z_back = np.zeros(((self.nx + 1),(self.ny + 1))) #Background magnetic field in z domain
		self.B_z_ind_n_phalf = np.zeros(((self.nx + 1),(self.ny + 1))) #Induced magnetic field in z domain on half time step after current step
		self.B_z_ind_n_mhalf = np.zeros(((self.nx + 1),(self.ny + 1))) #Induced magnetic field in z domain on half time step before current step
		self.B_z_ind_n = np.zeros(((self.nx + 1),(self.ny + 1))) #Induced magnetic field in z domain on current time step
		self.q_grid_n = np.zeros(((self.nx + 1),(self.ny + 1))) #Charge density on current time step
		self.J_x_n = np.zeros(((self.nx + 1),(self.ny + 1))) #Current density in x domain on current time step
		self.J_y_n = np.zeros(((self.nx + 1),(self.ny + 1))) #Current density in y domain on current time step
		self.phi = np.zeros(((self.nx + 1),(self.ny + 1))) #Electrostatic potential

		#Additional field definitions
		self.p_n = np.zeros(((self.nx + 1),(self.ny + 1))) #Pressure field on current time step
		self.p_n_phalf = np.zeros(((self.nx + 1),(self.ny + 1))) #Pressure field on plus a half integer time step
		self.U_x_n_phalf = np.zeros(((self.nx + 1),(self.ny + 1))) #Flow velocity in x domian on plus a half integer time step
		self.U_y_n_phalf = np.zeros(((self.nx + 1),(self.ny + 1))) #Flow velocity in y domian on plus a half integer time step
		self.U_x_n = np.zeros(((self.nx + 1),(self.ny + 1)))
		self.U_y_n = np.zeros(((self.nx + 1),(self.ny + 1)))
		self.E_x_new = np.zeros(((self.nx + 1),(self.ny + 1)))
		self.E_y_new = np.zeros(((self.nx + 1),(self.ny + 1)))
		self.E_x_mix = np.zeros(((self.nx + 1),(self.ny + 1)))
		self.E_y_mix = np.zeros(((self.nx + 1),(self.ny + 1)))
		self.E_x_n_phalf = np.zeros(((self.nx + 1),(self.ny + 1)))
		self.E_y_n_phalf = np.zeros(((self.nx + 1),(self.ny + 1)))

		self.p_n_pone = self.p_n

		self.t = np.linspace(self.t_min,self.t_max,num=self.nt) #Populate dimesnional vectors
		self.dt = (abs(self.t_min) + self.t_max) / self.nt #Size of step on the temporal domain

		self.E_mag_wb = np.empty((len(self.t),self.nx+1,self.ny+1))
		self.d_rho_m_1d_wb = np.empty((len(self.t),self.nx+1,self.ny+1))
		self.q_grid_wb = np.empty((len(self.t),self.nx+1,self.ny+1))
		self.B_mag_wb =np.empty((len(self.t),self.nx+1,self.ny+1))

	def sanity_check(self):
		"""Sanity check the setup for the model"""
		if (not self.Periodic) and (not self.Open):
			raise ValueError('At least one boundary condition must be selected')

		if (self.x_max<self.x_min):
			raise ValueError('xmax must be greater than xmin')

		if (self.y_max<self.y_min):
			raise ValueError('ymax must be greater than ymin')

		if (self.nx<3):
			raise ValueError('Must have at least three grid points in x')

		if (self.ny<3):
			raise ValueError('Must have at least three grid points in x')

		# if (self.T_PPC<1):
		# 	raise ValueError('Must have at least one particle in the simulation')

		if (self.PPC<3):
			print('[Hyb2D] Warning: less than 3 particles per cell')

	def random_particle(self):
		q = np.random.choice(self.q_c, p=self.qm_choice)
		m = np.random.choice(self.m, p=self.qm_choice)
		x, y = self.random_particle_position()
		vx, vy = self.random_particle_velocity()
		return Par._2D(x, y, vx, vy, q, m)

	def initial_conditions(self):
		#Open list for particles properties
		self.p = []

		# Setup initial conditions (particles and fields).
		self.initialise_particles()
		self.initial_field_conditions()

		try:
			av_omegag = np.average(np.abs(self.q_c))*np.average(self.B_z_back)/np.average(self.m)
			print('[Hyb2D] Order of gyro period: 10^{}s'.format(orderOfMagnitude(2*con.pi/av_omegag)))
			print('[Hyb2D] Order of gyro radius: 10^{}m'.format(orderOfMagnitude((np.average(m)*dx)/(np.average(np.abs(q_c))*np.average(B_z_back)))))
		except:
			pass

	def initialise_particles(self):
		for i in range(0,self.T_PPC):
			self.p.append(self.random_particle())

	def initial_field_conditions(self):
		self.g = [] #open list to store grid related information for particles of corresponding index numbers
		tracer_particle = []
		E_mag = []
		for i in range(0,len(self.p)):
			x_i = int(np.floor((self.p[i].x - self.x_min) / self.dx))
			y_i = int(np.floor((self.p[i].y - self.y_min) / self.dy))

			hx = self.p[i].x - self.x[x_i]
			hy = self.p[i].y - self.y[y_i]

			#Calculate contribution of each particle's charge to surrounding grid
			#locations in order to calculate current density on grid
			f_ij = (((self.dx - hx) * (self.dy - hy)) / self.dxdy)
			f_i1j = ((hx * (self.dy - hy)) / self.dxdy)
			f_i1j1 = ((hx * hy) / self.dxdy)
			f_ij1 = (((self.dx - hx)* hy) / self.dxdy)

			self.q_grid_n[x_i][y_i] += ((self.p[i].q)*f_ij)/self.dxdy
			self.q_grid_n[x_i+1][y_i] += ((self.p[i].q)*f_i1j)/self.dxdy
			self.q_grid_n[x_i+1][y_i+1] += ((self.p[i].q)*f_i1j1)/self.dxdy
			self.q_grid_n[x_i][y_i+1] += ((self.p[i].q)*f_ij1)/self.dxdy

			self.U_x_n[x_i][y_i] += (((self.p[i].q*self.p[i].vx))*f_ij)/self.dxdy
			self.U_x_n[x_i+1][y_i] += (((self.p[i].q*self.p[i].vx))*f_i1j)/self.dxdy
			self.U_x_n[x_i+1][y_i+1] += (((self.p[i].q*self.p[i].vx))*f_i1j1)/self.dxdy
			self.U_x_n[x_i][y_i+1] += (((self.p[i].q*self.p[i].vx))*f_ij1)/self.dxdy

			self.U_y_n[x_i][y_i] += (((self.p[i].q*self.p[i].vy))*f_ij)/self.dxdy
			self.U_y_n[x_i+1][y_i] += (((self.p[i].q*self.p[i].vy))*f_i1j)/self.dxdy
			self.U_y_n[x_i+1][y_i+1] += (((self.p[i].q*self.p[i].vy))*f_i1j1)/self.dxdy
			self.U_y_n[x_i][y_i+1] += (((self.p[i].q*self.p[i].vy))*f_ij1)/self.dxdy

			self.g.append(Grid_Info._2D(x_i,y_i,f_ij,f_i1j,f_i1j1,f_ij1))

		self.U_x_n_mhalf = self.U_x_n
		self.U_y_n_mhalf = self.U_y_n

		#Magnetic Field
		self.config_Bz_back(self.B_z_back)
		self.config_Ex_back(self.E_x_back)
		self.config_Ey_back(self.E_y_back)

		self.den_n_mhalf = (np.average(self.m)*(self.q_grid_n/(np.average(self.q_c))))

		#Electric field
		self.phi = Poisson_solver._2D(self.phi,(self.q_grid_n/con.epsilon_0),self.dx,self.dy)

		self.E_x_n = -(diff.x(self.phi,self.dx,Boundary_Type='Periodic'))
		self.E_y_n = -(diff.y(self.phi,self.dy,Boundary_Type='Periodic'))

		self.h = (self.p[0].q*self.dt)/self.p[0].m

		self.par_track = 0


	def advance_one_step(self,n):
		#Clear variables
		U_x_minus = np.zeros(((self.nx + 1),(self.ny + 1)))
		U_y_minus = np.zeros(((self.nx + 1),(self.ny + 1)))

		rm = []
		i = int(0)

		#Step 1: Advance particle velocity using the Lorentz eqn
		for i in range(0,len(self.p)):
			#Interpolate
			B_z = Interpolate._2D(self.B_z_n,self.g,i)
			E_x = Interpolate._2D(self.E_x_n,self.g,i)
			E_y = Interpolate._2D(self.E_y_n,self.g,i)

#			E_mag.append(np.sqrt((E_x**2)+(E_y**2)))

			#Solve
			h = (self.p[i].q*self.dt)/self.p[i].m
			alpha = ((1/2)*B_z) + ((self.p[i].m/self.p[i].q)*self.Omega)
			k = 1/(1+((h**2)*(alpha**2)))

			v_x_n_phalf = k*(((1-((h**2)*(alpha**2)))*self.p[i].vx) + (h*(E_x+(2*self.p[i].vy*alpha))) + ((h**2)*(alpha*E_y)) + (self.dt*(alpha**2)*(self.p[i].x+(h*alpha*self.p[i].y))))
			v_y_n_phalf = k*(((1-((h**2)*(alpha**2)))*self.p[i].vy) + (h*(E_y-(2*self.p[i].vx*alpha))) - ((h**2)*(alpha*E_x)) + (self.dt*(alpha**2)*(self.p[i].y-(h*alpha*self.p[i].x))))

			if v_x_n_phalf > con.c:
				raise Exception('Particle x-velocity = {}m/s \nExceeding speed of light! \n'.format(v_x_n_phalf))
			elif v_y_n_phalf > con.c:
				raise Exception('Particle y-velocity = {}m/s \nExceeding speed of light! \n'.format(v_y_n_phalf))
			else:
				pass

			self.p[i].vx = v_x_n_phalf
			self.p[i].vy = v_y_n_phalf

			#Pre-step 3: Calculate current density at U^-
			U_x_minus[self.g[i].x_i][self.g[i].y_i] += (self.p[i].vx*self.g[i].f_ij)/self.dxdy
			U_x_minus[self.g[i].x_i+1][self.g[i].y_i] += (self.p[i].vx*self.g[i].f_i1j)/self.dxdy
			U_x_minus[self.g[i].x_i+1][self.g[i].y_i+1] += (self.p[i].vx*self.g[i].f_i1j1)/self.dxdy
			U_x_minus[self.g[i].x_i][self.g[i].y_i+1] += (self.p[i].vx*self.g[i].f_ij1)/self.dxdy

			U_y_minus[self.g[i].x_i][self.g[i].y_i] += (self.p[i].vy*self.g[i].f_ij)/self.dxdy
			U_y_minus[self.g[i].x_i+1][self.g[i].y_i] += (self.p[i].vy*self.g[i].f_i1j)/self.dxdy
			U_y_minus[self.g[i].x_i+1][self.g[i].y_i+1] += (self.p[i].vy*self.g[i].f_i1j1)/self.dxdy
			U_y_minus[self.g[i].x_i][self.g[i].y_i+1] += (self.p[i].vy*self.g[i].f_ij1)/self.dxdy

			#Step 2: Advance particle position using the definition of velocity
			self.p[i].x = float(PostionUpdater(self.dt, self.p[i].x, self.p[i].vx))
			self.p[i].y = float(PostionUpdater(self.dt, self.p[i].y, self.p[i].vy))

			#Check position and obtain particle index numbers that require removing
			#from particle list
			#Open / clear variables for use
			x_i = 0
			y_i = 0
			#Calculate grid locations of particle
			x_i = int(np.floor((self.p[i].x - self.x_min) / self.dx))
			y_i = int(np.floor((self.p[i].y - self.y_min) / self.dy))

			if self.Periodic == True:
				#If outside x domain
				if x_i < 0:
					self.p[i].x = self.x_max - (self.dx/10)
				elif x_i > int(self.nx-1):
					self.p[i].x = self.x_min + (self.dx/10)
				else:
					pass
				#If outside y domian
				if y_i < 0:
					self.p[i].y = self.y_max - (self.dy/10)
				elif y_i > int(self.ny-1):
					self.p[i].y = self.y_min + (self.dy/10)
				else:
					pass
			elif self.Open == True:
				#If outside x domain delete
				if x_i < 0 or x_i > int(self.nx-1):
					#DEBUG: Print which particle is removed
					print('Out of range: {}'.format(i))
					rm.append(i)
				#If outside y domian delete
				elif y_i < 0 or y_i > int(self.ny-1):
					#DEBUG: Print which particle is removed
					print('Out of range: {}'.format(i))
					rm.append(i)
				else:
					pass

		#Remove particles out of simulation area using reference list 'rm'
		for iter in sorted(rm, reverse=True):
			del p[iter]

		#Calculate number of particles that require regenerating
		Regenerate_num = self.T_PPC - len(self.p)
		r_p = []

		#If particles require regenerating then activate loop to generate insertion
		#parameters for as many new particles as required
		if Regenerate_num > 0:
			for re in range(0,Regenerate_num):
				self.p.append(self.random_particle())

		#Step 3: Interpolate current density
		self.g = []
		q_grid_n_pone = np.zeros_like(self.q_grid_n)
		U_x_plus = np.zeros(((self.nx + 1),(self.ny + 1)))
		U_y_plus = np.zeros(((self.nx + 1),(self.ny + 1)))
		J_x_plus = np.zeros(((self.nx + 1),(self.ny + 1)))
		J_y_plus = np.zeros(((self.nx + 1),(self.ny + 1)))
		Lambda_n_pone = np.zeros(((self.nx + 1),(self.ny + 1)))
		Gamma_x_n_pone = np.zeros(((self.nx + 1),(self.ny + 1)))
		Gamma_y_n_pone = np.zeros(((self.nx + 1),(self.ny + 1)))
		for i in range(0,len(self.p)):

			x_i = int(np.floor((self.p[i].x - self.x_min)/ self.dx))
			y_i = int(np.floor((self.p[i].y - self.y_min)/ self.dy))

			hx = self.p[i].x - self.x[x_i]
			hy = self.p[i].y - self.y[y_i]

			f_ij = (((self.dx - hx) * (self.dy - hy)) / (self.dx * self.dy))
			f_i1j = ((hx * (self.dy - hy)) / (self.dx * self.dy))
			f_i1j1 = ((hx * hy) / (self.dx * self.dy))
			f_ij1 = (((self.dx - hx)* hy) / (self.dx * self.dy))

			self.g.append(Grid_Info._2D(x_i,y_i,f_ij,f_i1j,f_i1j1,f_ij1))

			#Obtain charge density at n + 1
			q_grid_n_pone[x_i][y_i] += ((self.p[i].q)*f_ij)/(self.dx*self.dy)
			q_grid_n_pone[x_i+1][y_i] += ((self.p[i].q)*f_i1j)/(self.dx*self.dy)
			q_grid_n_pone[x_i+1][y_i+1] += ((self.p[i].q)*f_i1j1)/(self.dx*self.dy)
			q_grid_n_pone[x_i][y_i+1] += ((self.p[i].q)*f_ij1)/(self.dx*self.dy)

			#Obtain mixed time step ion flow velocities and current density
			U_x_plus[x_i][y_i] += (self.p[i].vx*f_ij)/(self.dx*self.dy)
			U_x_plus[x_i+1][y_i] += (self.p[i].vx*f_i1j)/(self.dx*self.dy)
			U_x_plus[x_i+1][y_i+1] += (self.p[i].vx*f_i1j1)/(self.dx*self.dy)
			U_x_plus[x_i][y_i+1] += (self.p[i].vx*f_ij1)/(self.dx*self.dy)

			U_y_plus[x_i][y_i] += (self.p[i].vy*f_ij)/(self.dx*self.dy)
			U_y_plus[x_i+1][y_i] += (self.p[i].vy*f_i1j)/(self.dx*self.dy)
			U_y_plus[x_i+1][y_i+1] += (self.p[i].vy*f_i1j1)/(self.dx*self.dy)
			U_y_plus[x_i][y_i+1] += (self.p[i].vy*f_ij1)/(self.dx*self.dy)

			#Obtain other variables needed for CAM
			Lambda_n_pone[x_i][y_i] = ((self.p[i].q**2)/self.p[i].m)*f_ij
			Lambda_n_pone[x_i+1][y_i] = ((self.p[i].q**2)/self.p[i].m)*f_i1j
			Lambda_n_pone[x_i+1][y_i+1] = ((self.p[i].q**2)/self.p[i].m)*f_i1j1
			Lambda_n_pone[x_i][y_i+1] = ((self.p[i].q**2)/self.p[i].m)*f_ij1

			Gamma_x_n_pone[x_i][y_i] = ((self.p[i].q**2)/self.p[i].m)*self.p[i].vx*f_ij
			Gamma_x_n_pone[x_i+1][y_i] = ((self.p[i].q**2)/self.p[i].m)*self.p[i].vx*f_i1j
			Gamma_x_n_pone[x_i+1][y_i+1] = ((self.p[i].q**2)/self.p[i].m)*self.p[i].vx*f_i1j1
			Gamma_x_n_pone[x_i][y_i+1] = ((self.p[i].q**2)/self.p[i].m)*self.p[i].vx*f_ij1

			Gamma_y_n_pone[x_i][y_i] = ((self.p[i].q**2)/self.p[i].m)*self.p[i].vy*f_ij
			Gamma_y_n_pone[x_i+1][y_i] = ((self.p[i].q**2)/self.p[i].m)*self.p[i].vy*f_i1j
			Gamma_y_n_pone[x_i+1][y_i+1] = ((self.p[i].q**2)/self.p[i].m)*self.p[i].vy*f_i1j1
			Gamma_y_n_pone[x_i][y_i+1] = ((self.p[i].q**2)/self.p[i].m)*self.p[i].vy*f_ij1

		#CAM: Obtain mixed time current densities
		J_x_plus = q_grid_n_pone*U_x_plus
		J_y_plus = q_grid_n_pone*U_y_plus

		#Step 3.5: Obtain flow velocities on the grid points
		#TODO: Needs replacing with proper solution for low denisty areas, nabla^2(E) = 0
		q_grid_n_pone_eq_0 = np.where(q_grid_n_pone==0)
		q_grid_n_pone_neq_0 = np.where(q_grid_n_pone!=0)
		#TODO: Remove need for this
		q_grid_n_pone[q_grid_n_pone==0] = np.average(q_grid_n_pone/100)

		U_x_n_phalf = (1/2)*(U_x_minus + U_x_plus)
		U_y_n_phalf = (1/2)*(U_y_minus + U_y_plus)

		q_grid_n_phalf = (1/2)*(self.q_grid_n + q_grid_n_pone)
		q_grid_n_phalf_neq_0 = np.where(q_grid_n_phalf!=0)
		q_grid_n_phalf_eq_0 = np.where(q_grid_n_phalf==0)

		#Step 4: Advance mangetic field by half a time step by utilising Faraday's eqn
		B_z_ind_n_phalf =  Magnetic_MacCormack(self.dx,self.dy,self.dt,self.B_z_ind_n,self.U_x_n,self.U_y_n)

		B_z_n_phalf = self.B_z_back + B_z_ind_n_phalf

		#Step 5:Advance electric field by half a time step utilising the Electron
		#momentum eqn
		#TODO: Check maths for re-expressing curl(curlB) here
		E_x_new = (U_y_n_phalf*B_z_n_phalf) + ((1/q_grid_n_phalf)*(diff.x(((B_z_n_phalf**2)/2*con.mu_0),self.dx,Boundary_Type='Periodic')))
		E_y_new = -(U_x_n_phalf*B_z_n_phalf) + ((1/q_grid_n_phalf)*(diff.y(((B_z_n_phalf**2)/2*con.mu_0),self.dy,Boundary_Type='Periodic')))

		#TODO: Change pressure calculation to use MHD energy eqn
		E_x_n_phalf = self.E_x_back + (E_x_new - (diff.x(self.p_n_phalf,self.dx,Boundary_Type='Periodic')/q_grid_n_phalf))
		E_y_n_phalf = self.E_y_back + (E_y_new - (diff.y(self.p_n_phalf,self.dy,Boundary_Type='Periodic')/q_grid_n_phalf))

		if len(q_grid_n_phalf_eq_0) == 2 and len(q_grid_n_phalf_eq_0[0]) > 0:
			#Use nabla^2(E)=0 to calculate electric field where charge density tends to
			#zero
			E_x_n_phalf_low_den = Vector_laplacian_solver._2D.x(E_x_n_phalf,E_y_n_phalf,self.dx,self.dy)
			E_y_n_phalf_low_den = Vector_laplacian_solver._2D.y(E_x_n_phalf,E_y_n_phalf,self.dx,self.dy)

			for i in range(0,len(q_grid_n_phalf_eq_0[0])):
				E_x_n_phalf[int(q_grid_n_phalf_eq_0[0][i])][int(q_grid_n_phalf_eq_0[1][i])] = E_x_n_phalf_low_den[int(q_grid_n_phalf_eq_0[0][i])][int(q_grid_n_phalf_eq_0[1][i])]
				E_y_n_phalf[int(q_grid_n_phalf_eq_0[0][i])][int(q_grid_n_phalf_eq_0[1][i])] = E_y_n_phalf_low_den[int(q_grid_n_phalf_eq_0[0][i])][int(q_grid_n_phalf_eq_0[1][i])]
		else:
			pass

		#Step 6: Advance magnetic field an additional half time step in order to get
		#the full integer advance in time required
		self.B_z_ind_n = Magnetic_MacCormack(self.dx,self.dy,self.dt,B_z_ind_n_phalf,U_x_n_phalf,U_y_n_phalf)
		self.B_z_n = self.B_z_back + self.B_z_ind_n

		#CAM step: Advance current density
		E_x_new = (U_y_plus*self.B_z_n) + ((1/q_grid_n_pone)*(diff.x(((self.B_z_n**2)/2*con.mu_0),self.dx,Boundary_Type='Periodic')))
		E_y_new = -(U_x_plus*self.B_z_n) + ((1/q_grid_n_pone)*(diff.y(((self.B_z_n**2)/2*con.mu_0),self.dy,Boundary_Type='Periodic')))

		E_x_mix = self.E_x_back + (E_x_new - (diff.x(self.p_n_pone,self.dx,Boundary_Type='Periodic')/q_grid_n_pone))
		E_y_mix = self.E_y_back + (E_y_new - (diff.y(self.p_n_pone,self.dy,Boundary_Type='Periodic')/q_grid_n_pone))

		if len(q_grid_n_pone_eq_0) == 2 and len(q_grid_n_pone_eq_0[0]) > 0:
			#Use nabla^2(E)=0 to calculate electric field where charge density tends to
			#zero
			E_x_mix_low_den = Vector_laplacian_solver._2D.x(E_x_mix,E_y_mix,self.dx,self.dy)
			E_y_mix_low_den = Vector_laplacian_solver._2D.y(E_x_mix,E_y_mix,self.dx,self.dy)

			for i in range(0,len(q_grid_n_pone_eq_0[0])):
				E_x_mix[int(q_grid_n_pone_eq_0[0][i])][int(q_grid_n_pone_eq_0[1][i])] = E_x_mix_low_den[int(q_grid_n_pone_eq_0[0][i])][int(q_grid_n_pone_eq_0[1][i])]
				E_y_mix[int(q_grid_n_pone_eq_0[0][i])][int(q_grid_n_pone_eq_0[1][i])] = E_y_mix_low_den[int(q_grid_n_pone_eq_0[0][i])][int(q_grid_n_pone_eq_0[1][i])]
		else:
			pass

		self.J_x_n = J_x_plus + (self.dt/2)*((Lambda_n_pone*E_x_mix)+(Gamma_y_n_pone*self.B_z_n))
		self.J_y_n = J_y_plus + (self.dt/2)*((Lambda_n_pone*E_y_mix)-(Gamma_x_n_pone*self.B_z_n))

		self.U_x_n[q_grid_n_pone_neq_0] = self.J_x_n[q_grid_n_pone_neq_0]/q_grid_n_pone[q_grid_n_pone_neq_0]
		self.U_y_n[q_grid_n_pone_neq_0] = self.J_y_n[q_grid_n_pone_neq_0]/q_grid_n_pone[q_grid_n_pone_neq_0]
		self.U_x_n[q_grid_n_pone_eq_0] = 0.
		self.U_y_n[q_grid_n_pone_eq_0] = 0.

		#Step 8: Advance electric field an additional half time step
		#Dropping mu_0 in front of q_grid_n_pone as unclear if needed
		E_x_new = (self.U_y_n*self.B_z_n) + ((1/q_grid_n_pone)*(diff.x(((self.B_z_n**2)/2*con.mu_0),self.dx,Boundary_Type='Periodic')))
		E_y_new = -(self.U_x_n*self.B_z_n) + ((1/q_grid_n_pone)*(diff.y(((self.B_z_n**2)/2*con.mu_0),self.dy,Boundary_Type='Periodic')))

		self.E_x_n = self.E_x_back + (E_x_new - (diff.x(self.p_n_pone,self.dx,Boundary_Type='Periodic')/q_grid_n_pone))
		self.E_y_n = self.E_y_back + (E_y_new - (diff.y(self.p_n_pone,self.dy,Boundary_Type='Periodic')/q_grid_n_pone))

		if len(q_grid_n_pone_eq_0) == 2 and len(q_grid_n_pone_eq_0[0]) > 0:
			#Use nabla^2(E)=0 to calculate electric field where charge density tends to
			#zero
			E_x_n_low_den = Vector_laplacian_solver._2D.x(self.E_x_n,self.E_y_n,self.dx,self.dy)
			E_y_n_low_den = Vector_laplacian_solver._2D.y(self.E_x_n,self.E_y_n,self.dx,self.dy)

			for i in range(0,len(q_grid_n_pone_eq_0[0])):
				self.E_x_n[int(q_grid_n_pone_eq_0[0][i])][int(q_grid_n_pone_eq_0[1][i])] = E_x_n_low_den[int(q_grid_n_pone_eq_0[0][i])][int(q_grid_n_pone_eq_0[1][i])]
				self.E_y_n[int(q_grid_n_pone_eq_0[0][i])][int(q_grid_n_pone_eq_0[1][i])] = E_y_n_low_den[int(q_grid_n_pone_eq_0[0][i])][int(q_grid_n_pone_eq_0[1][i])]
		else:
			pass

		#Clean up & advance any variables required
		self.q_grid_n = q_grid_n_pone
		B_z_ind_n_mhalf = B_z_ind_n_phalf
		U_x_n_mhalf = U_x_n_phalf
		U_y_n_mhalf = U_y_n_phalf
		p_n_mhalf = self.p_n_phalf
		self.p_n = self.p_n_pone

	def run(self):
		end_step = self.nt #nt
		for n in range(1,end_step):
			if n == 1:
				tic = timeit.default_timer()

			if n % 1 == 0:
				p_vx = np.zeros(len(self.p))
				p_vy = np.zeros(len(self.p))
				for i in range(0,len(self.p)):
					p_vx[i] = self.p[i].vx
					p_vy[i] = self.p[i].vy
				print('Simulation Time: {}/{}s'.format(self.t[n],np.max(self.t)))

			self.advance_one_step(n)

			if n == 1:
				toc = timeit.default_timer()
				step_time =(toc - tic)
				run_time = round(step_time*end_step,2)
				print('Estimated run time: {}s'.format(run_time))

			if self.data.ready_steps(n):
				p_vx = np.zeros(len(self.p))
				p_vy = np.zeros(len(self.p))
				for i in range(0,len(self.p)):
					p_vx[i] = self.p[i].vx
					p_vy[i] = self.p[i].vy
				self.data.store_log('step', n)
				self.data.store_log('time', self.t[n])
				self.data.store_log('cfl', (((np.abs(np.average(p_vx))*self.dt)/self.dx)+((np.abs(np.average(p_vy))*self.dt)/self.dy)))
				self.data.store_log('divJ', np.average(diff.x(self.J_x_n,self.dx)+diff.y(self.J_y_n,self.dy)))
				self.data.store_log('divB',  0.0)
				self.data.store_log('gauss_law', np.average((diff.x(self.E_x_n,self.dx)+diff.y(self.E_y_n,self.dy))-(self.q_grid_n/con.epsilon_0)))
				self.data.store_log('num_particles', len(self.p))
				self.data.next_log()

			if self.data.ready_fields(n):
				self.data.store_field('Ex', self.E_x_n)
				self.data.store_field('Ey', self.E_y_n)
				self.data.next_fields()

			if self.data.ready_all_particles(n):
				self.data.store_particle_dataset(n, self.p)

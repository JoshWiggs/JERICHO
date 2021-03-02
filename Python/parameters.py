import scipy.constants as con
from Modules.classes import parameters as parms

### Set Model Options ##########################################################
#Set Boundaries
parms.options.boundaries.periodic = bool(True) #Default
parms.options.boundaries.open = bool(False)

#Set particle distribution method
parms.options.particle_distribution.uniform = bool(True) #Default
parms.options.particle_distribution.Maxwellian = bool(False)

#Set electric field initialisation method
parms.options.electric_initialisation.electrostatic = bool(False)
parms.options.electric_initialisation.electrodynamic = bool(True) #Default

#Select Magnetic field type
parms.options.mag_field_type.dipole = bool(False)
parms.options.mag_field_type.uniform = bool(True)

#Select IO method for saving model outputs
parms.options.IO_method.pickle = bool(True) #Default
parms.options.IO_method.HDF5 = bool(False)

#Enable plotting in main loop
parms.options.plotting.on = bool(False)

#Advanced Coordinate Systems
parms.options.coords.radial = bool(False) #Advanced option for plotting

#Optional Modules
parms.options.modules.constants = bool(True)

### Set Planetary Parameters ###################################################
parms.planetary.R_J = float(71492e3) #Jupiter's radius (m) (NASA fact sheet)
parms.planetary.M_J = float(1898.19e24) #Jupiter's mass (kg) (NASA fact sheet)
parms.planetary.Omega = float(1.7584e-4) #Jupiter's angular velocity (rad/s)
parms.planetary.B_eq = float(1e-10*(417e-6)) #Jupiter's equatorial field strength (T) (CURRENTLY ALTERED)

### Set Model Parameters #######################################################
#Spatial domain variables (SI)
parms.var.x_min = float(50*parms.planetary.R_J)
parms.var.x_max = float(51*parms.planetary.R_J)
parms.var.y_min = float(50*parms.planetary.R_J)
parms.var.y_max = float(51*parms.planetary.R_J)

#Spatial grid Size
parms.var.nx = int(50)
parms.var.ny = int(50)

#Temporal domain variables (SI)
parms.var.t_min = float(0)
parms.var.t_max = float(1)
parms.var.nt = int(1001)

#Plasma Parameters
parms.var.PPC = int(50)
if parms.options.particle_distribution.Maxwellian == True:
    parms.var.T_i = float((500*con.e))

#Species charge & mass
parms.var.q_c = [con.e]
parms.var.m = [con.m_p]

#Other parameters
parms.var.n_smoothing = int(20)

#Validation#####################################################################
#Check set up options
#Boundaries
if parms.options.boundaries.periodic == False and parms.options.boundaries.open == False:
    raise Exception('Please select a boundary condition in options.')
elif parms.options.boundaries.periodic == True and parms.options.boundaries.open == True:
    raise Exception('Please select only one boundary condition.')
#Particle distributions
if parms.options.particle_distribution.uniform == False and parms.options.particle_distribution.Maxwellian == False:
    raise Exception('Please select a particle distribution method in options.')
elif parms.options.particle_distribution.uniform == True and parms.options.particle_distribution.Maxwellian == True:
    raise Exception('Please select only one particle distribution method.')
#Electric field initialisation
if parms.options.electric_initialisation.electrostatic == False and parms.options.electric_initialisation.electrodynamic == False:
    raise Exception('Please select a electric field initialisation method in options.')
elif parms.options.electric_initialisation.electrostatic == True and parms.options.electric_initialisation.electrodynamic == True:
    raise Exception('Please select only one electric field initialisation method.')
#Magnetic field initialisation
if parms.options.mag_field_type.dipole == False and parms.options.mag_field_type.uniform == False:
    raise Exception('Please select a magentic field initialisation method in options.')
elif parms.options.mag_field_type.dipole == True and parms.options.mag_field_type.uniformc == True:
    raise Exception('Please select only one megantic field initialisation method.')

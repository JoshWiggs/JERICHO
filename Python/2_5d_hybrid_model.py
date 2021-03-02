#Import python native modules
import random as ran
import timeit
import math
import pickle

#Import 3rd party python modules
import numpy as np

#Import custom models
from Modules.numerical_differentials import *
from Modules.functions import *
from Modules.classes import Particle_Information as Par
from Modules.classes import Particle_Grid_Information as Grid_Info
import Modules.io

#Open Parameters:
from parameters import *

#Optional modules
if parms.options.modules.constants == True:
    import scipy.constants as con
if parms.options.IO_method.HDF5 == True:
    import Modules.io
if parms.options.plotting.on == True:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm


################################################################################
print('Initialising Simulation...')

#Set Up#########################################################################
dx = (parms.var.x_max - parms.var.x_min) / parms.var.nx #Size of step in x domain
dy = (parms.var.y_max - parms.var.y_min) / parms.var.ny #Size of step in y domain
T_c = parms.var.nx * parms.var.ny #Total number of cells

#Populate dimesnional vectors
x = np.linspace(parms.var.x_min,parms.var.x_max,num=(parms.var.nx+1))
y = np.linspace(parms.var.y_min,parms.var.y_max,num=(parms.var.ny+1))

#Meshgrid for plotting
X,Y = np.meshgrid(x,y)

#Radial calculations
if parms.options.coords.radial == True:
    r_min = np.min(np.sqrt((x**2)+(y**2)))
    r_max = r_min + np.sqrt((dx**2)+(dy**2))
    phi_min = np.min(np.arctan(y/x))
    phi_max = np.max(np.arctan(y/x))

#Electromagentic field variables
E_x_n = np.zeros(((parms.var.nx + 1),(parms.var.ny + 1))) #Electic field in x domain on current time step
E_y_n = np.zeros(((parms.var.nx + 1),(parms.var.ny + 1))) #Electic field in y domain on current time step
E_x_back = np.zeros(((parms.var.nx + 1),(parms.var.ny + 1))) #Background electric field in the x domain
E_y_back = np.zeros(((parms.var.nx + 1),(parms.var.ny + 1))) #Background electric field in the y domian
B_z_n = np.zeros(((parms.var.nx + 1),(parms.var.ny + 1))) #Magnetic field in z domain on current time step
B_z_back = np.zeros(((parms.var.nx + 1),(parms.var.ny + 1))) #Background magnetic field in z domain
B_z_ind_n_phalf = np.zeros(((parms.var.nx + 1),(parms.var.ny + 1))) #Induced magnetic field in z domain on half time step after current step
B_z_ind_n_mhalf = np.zeros(((parms.var.nx + 1),(parms.var.ny + 1))) #Induced magnetic field in z domain on half time step before current step
B_z_ind_n = np.zeros(((parms.var.nx + 1),(parms.var.ny + 1))) #Induced magnetic field in z domain on current time step
q_grid_n = np.zeros(((parms.var.nx + 1),(parms.var.ny + 1))) #Charge density on current time step
J_x_n = np.zeros(((parms.var.nx + 1),(parms.var.ny + 1))) #Current density in x domain on current time step
J_y_n = np.zeros(((parms.var.nx + 1),(parms.var.ny + 1))) #Current density in y domain on current time step
phi = np.zeros(((parms.var.nx + 1),(parms.var.ny + 1))) #Electrostatic potential

#Additional field definitions
U_x_n_phalf = np.zeros(((parms.var.nx + 1),(parms.var.ny + 1))) #Flow velocity in x domian on plus a half integer time step
U_y_n_phalf = np.zeros(((parms.var.nx + 1),(parms.var.ny + 1))) #Flow velocity in y domian on plus a half integer time step
U_x_n = np.zeros(((parms.var.nx + 1),(parms.var.ny + 1))) #Flow velocity in x domian
U_y_n = np.zeros(((parms.var.nx + 1),(parms.var.ny + 1))) #Flow velocity in y domian
E_x_new = np.zeros(((parms.var.nx + 1),(parms.var.ny + 1))) #Induced electric field in x domain
E_y_new = np.zeros(((parms.var.nx + 1),(parms.var.ny + 1))) #Induced electric field in y domain
E_x_mix = np.zeros(((parms.var.nx + 1),(parms.var.ny + 1))) #Electric field in x domain for CAM solver
E_y_mix = np.zeros(((parms.var.nx + 1),(parms.var.ny + 1))) #Electric field in y domain for CAM solver
E_x_n_phalf = np.zeros(((parms.var.nx + 1),(parms.var.ny + 1))) #Electic field in x domain on current time step on half integer time step
E_y_n_phalf = np.zeros(((parms.var.nx + 1),(parms.var.ny + 1))) #Electic field in y domain on current time step on half integer time step

#Time Variables
t = np.linspace(parms.var.t_min,parms.var.t_max,num=parms.var.nt)
dt = (parms.var.t_max - parms.var.t_min) / parms.var.nt

print('Order of time step: 10^{}s \nOrder of grid spacing: 10^{}m \n'.format(orderOfMagnitude(dt),orderOfMagnitude(dx)))

T_PPC = int((T_c * parms.var.PPC)) #Total number of particles in simulation

print('Number of Particles: {} \nNumber Density: {}m^-2 \n'.format(T_PPC,(T_PPC/((parms.var.x_max-parms.var.x_min)*(parms.var.y_max-parms.var.y_min)))))

#Field Parameters
if parms.options.mag_field_type.dipole == True:
    B_z_0 = (parms.planetary.B_eq)/(np.sqrt((X/parms.planetary.R_J)**2 + (Y/parms.planetary.R_J)**2)**3)
elif parms.options.mag_field_type.uniform == True:
    B_z_0 = np.empty_like(B_z_n)
    B_z_0.fill(parms.planetary.B_eq)
E_x_0 = 0.
E_y_0 = 0.

#Create Particles###############################################################
#Open list for particles properties
p = []

#Ion velocity distribution variables (uniform)
v_min = -(dx) #Minimum initial particle speed
v_max = (dx) #Maximum initial particle speed
#Ion thermal velocity (Maxwellian)
if parms.options.particle_distribution.Maxwellian == True:
    v_th_maxwell = []
    for i in range(0,len(parms.var.m)):
        v_th_maxwell.append(np.sqrt((2*parms.var.T_i)/(parms.var.m[i])))
    v_th_maxwell = np.array(v_th_maxwell)
    v_th = np.sqrt((v_th_maxwell**2).sum())
elif parms.options.particle_distribution.uniform == True:
    v_th = np.sqrt((v_min**2)+(v_max**2))

for i in range(0,T_PPC):
    #TODO: Generalise
    j = ran.randint(0,int(len(parms.var.q_c)-1)) #Random integer for mass and charge selection
    s = ran.choice((-1,1))
    n = ran.choice((-1,1))
    Par_x = ran.uniform(parms.var.x_min,parms.var.x_max) #Particle x coordinate
    Par_y = ran.uniform(parms.var.y_min,parms.var.y_max) #Particle y coordinate
    if parms.options.particle_distribution.uniform == True:
        Par_vx = (ran.uniform(v_min,v_max)) #Particle velocity on the x direction
        Par_vy = (ran.uniform(v_min,v_max)) #Particle velocity on the y direction
    elif parms.options.particle_distribution.Maxwellian == True:
        Par_vx = s*(v_th_maxwell[j]/np.sqrt(2)) #Particle velocity on the x direction
        Par_vy = n*(v_th_maxwell[j]/np.sqrt(2)) #Particle velocity on the y direction
    else:
        raise Exception("Please select particle initialisation method")
    Par_q = parms.var.q_c[j] #Particle charge
    Par_m = parms.var.m[j] #Particle mass

    p.append(Par._2D(Par_x,Par_y,Par_vx,Par_vy,Par_q,Par_m)) #add particle to particle list

#IC's###########################################################################
#at n = 0
#Interpolate
g = [] #open list to store grid related information for particles of corresponding index numbers
tracer_particle = [] #open list to store particle parameters at each time step

#Declare loop variables outside of object for faster access inside main loop
x_min = parms.var.x_min
x_max = parms.var.x_max
y_min = parms.var.y_min
y_max = parms.var.y_max
nx = parms.var.nx
ny = parms.var.ny
Omega = parms.planetary.Omega

#Loop through particle list to interpolate particle parameters using Cloud-in-Cell
#method
for i in range(0,len(p)):

    #Obtain index variables
    x_i = int(np.floor((p[i].x - x_min) / dx)) #index of grid cell particle occupies in x domain
    y_i = int(np.floor((p[i].y - y_min) / dy)) #index of grid cell particle occupies in y domain

    hx = p[i].x - x[x_i] #Distance from x index
    hy = p[i].y - y[y_i] #Distance from y index

    #Calculate contribution of each particle's charge to surrounding grid
    #locations in order to calculate current density on grid
    f_ij = (((dx - hx) * (dy - hy)) / (dx * dy))
    f_i1j = ((hx * (dy - hy)) / (dx * dy))
    f_i1j1 = ((hx * hy) / (dx * dy))
    f_ij1 = (((dx - hx)* hy) / (dx* dy))

    #DEBUG: Interpolation check
    if i == 0:
        if (f_ij+f_i1j+f_i1j1+f_ij1) >= 0.99:
            pass
        else:
            raise Exception('Interpolation debug test failed, please check parameters')

    #Intepolate charge density
    q_grid_n[x_i][y_i] += ((p[i].q)*f_ij)/(dx*dy)
    q_grid_n[x_i+1][y_i] += ((p[i].q)*f_i1j)/(dx*dy)
    q_grid_n[x_i+1][y_i+1] += ((p[i].q)*f_i1j1)/(dx*dy)
    q_grid_n[x_i][y_i+1] += ((p[i].q)*f_ij1)/(dx*dy)

    #Intepolate ion flow velocities in x domain
    U_x_n[x_i][y_i] += (((p[i].q*p[i].vx))*f_ij)/(dx*dy)
    U_x_n[x_i+1][y_i] += (((p[i].q*p[i].vx))*f_i1j)/(dx*dy)
    U_x_n[x_i+1][y_i+1] += (((p[i].q*p[i].vx))*f_i1j1)/(dx*dy)
    U_x_n[x_i][y_i+1] += (((p[i].q*p[i].vx))*f_ij1)/(dx*dy)

    #Intepolate ion flow velocities in y domain
    U_y_n[x_i][y_i] += (((p[i].q*p[i].vy))*f_ij)/(dx*dy)
    U_y_n[x_i+1][y_i] += (((p[i].q*p[i].vy))*f_i1j)/(dx*dy)
    U_y_n[x_i+1][y_i+1] += (((p[i].q*p[i].vy))*f_i1j1)/(dx*dy)
    U_y_n[x_i][y_i+1] += (((p[i].q*p[i].vy))*f_ij1)/(dx*dy)

    #Store interpolation parameters
    g.append(Grid_Info._2D(x_i,y_i,f_ij,f_i1j,f_i1j1,f_ij1))

#TODO: Back interpolate for estimation
U_x_n_mhalf = U_x_n.copy() #Ion flow velocity in x domain at previous half-integer time step
U_y_n_mhalf = U_y_n.copy() #Ion flow velocity in y domain at previous half-integer time step

#Populate background fields
B_z_back = (B_z_0)
E_x_back.fill(E_x_0)
E_y_back.fill(E_y_0)

#Electric field initialisation
if parms.options.electric_initialisation.electrostatic == True:
    #Obtain electric field from static distribution
    phi = Poisson_solver._2D(phi,(q_grid_n/con.epsilon_0),dx,dy) #Electrostatic potential
    E_x_n = -(diff.x(phi,dx,Boundary_Type='Periodic')) #Electric field in x domain from electrostatic potential
    E_y_n = -(diff.y(phi,dy,Boundary_Type='Periodic')) #Electric field in y domain from electrostatic potential
elif parms.options.electric_initialisation.electrodynamic == True:
    E_x_n = -(U_y_n*B_z_n) - ((1/q_grid_n)*(diff.x(((B_z_back**2)/2*con.mu_0),dx,Boundary_Type='Periodic')))
    E_y_n = (U_x_n*B_z_n) - ((1/q_grid_n)*(diff.y(((B_z_back**2)/2*con.mu_0),dy,Boundary_Type='Periodic')))
else:
    raise Exception("Please select electric field initialisation method")

#Calculate electron flow velocities
U_e_x = U_x_n - (diff.y(B_z_n,dy)/(con.mu_0*q_grid_n)) #Electron flow velocity in x domain
U_e_y = U_y_n + (diff.x(B_z_n,dx)/(con.mu_0*q_grid_n)) #Electron flow velocity in y domain
U_e_mag = np.sqrt((U_e_x**2)+(U_e_y**2)) #Magnitude of electron flow velocities
U_i_mag = np.sqrt((U_x_n**2)+(U_y_n**2))

#Initial pressure solver
#TODO: Add back when stable

#Parameter for particle pusher
h = (p[0].q*dt)/p[0].m

#Select particle for tracer
#par_track = 0

try:
    print('Order of gyro period: 10^{}s \nOrder of gyro radii: 10^{}m \n'.format(orderOfMagnitude(2*con.pi*(np.average(parms.var.m)/(np.average(np.abs(parms.var.q_c))*np.average(B_z_back)))),orderOfMagnitude((np.average(parms.var.m)*dx)/(np.average(np.abs(parms.var.q_c))*np.average(B_z_back)))))
except:
    pass

#Set up IO methods for saving model outputs ####################################
if parms.options.IO_method.pickle == True:
    #Open lists for pickling
    E_mag_wb = []
    q_grid_wb = []
    B_mag_wb =[]
elif parms.options.IO_method.HDF5 == True:
    #Open output file
    dataio = Modules.io.Output(nx, ny)
    dataio.write_steps = 1
    dataio.write_all_particles = 5
    dataio.write_fields = 10
    dataio.open('results.hdf5')
    dataio.attrs['R_P'] = parms.planetary.R_J
    dataio.attrs['M_P'] = parms.planetary.M_J
    dataio.attrs['Omega'] = parms.planetary.Omega
    dataio.attrs['B_eq'] = parms.planetary.B_eq
    dataio.attrs['x_min'] = parms.var.x_min
    dataio.attrs['x_max'] = parms.var.x_max
    dataio.attrs['y_min'] = parms.var.y_min
    dataio.attrs['y_max'] = parms.var.y_max
    dataio.store_grid(x, y, X, Y)
else:
    pass

#Main Loop######################################################################
#DEBUG: Manually select end step in the temporal domain
end_step = 100 #parms.var.nt #Select time step to finish simulation

#Start main simualtion run
for n in range(1,end_step):

    #Record system time at first time step to calculate time taken per step
    if n == 1:
        tic = timeit.default_timer()

    if n % 10 == 0 or n == 1:
        print('Time Step: {} Time: {}s\ndivB: {} divJ: {} \nGuass Law: {} \nNumber of Particles: {} \nCFL: {} \n'.format(n,round(t[n],2),(0),np.average(diff.x(J_x_n,dx)+diff.y(J_y_n,dy)),np.average((diff.x(E_x_n,dx)+diff.y(E_y_n,dy))-(q_grid_n/con.epsilon_0)),len(p),(np.average(np.abs(U_x_n))*dt/dx)+(np.average(np.abs(U_y_n))*dt/dy)))

    #try:
        #tracer_particle.append(Par._2D(p[par_track].x,p[par_track].y,p[par_track].vx,p[par_track].vy,p[par_track].q,p[par_track].m))
    #except:
        #pass

    #Overzealous clear variables
    U_x_minus = np.zeros(((nx + 1),(ny + 1))) #Ion flow velocity in x domain at mixed time step (v^n+1/2,x^n)
    U_y_minus = np.zeros(((nx + 1),(ny + 1))) #Ion flow velocity in y domain at mixed time step (v^n+1/2,x^n)
    rm = [] #List of particles to remove from domain (open boundaries)
    i = int(0) #Reset iter

    #Step 1: Advance particle velocities and positions
    #Loop through particle list
    for i in range(0,len(p)):
        #Interpolate
        B_z = Interpolate._2D(B_z_n,g,i)
        E_x = Interpolate._2D(E_x_n,g,i)
        E_y = Interpolate._2D(E_y_n,g,i)

        #Solve equation of motion
        h = (p[i].q*dt)/p[i].m
        alpha = ((1/2)*B_z) + ((p[i].m/p[i].q)*Omega)
        k = 1/(1+((h**2)*(alpha**2)))

        v_x_n_phalf = k*(((1-((h**2)*(alpha**2)))*p[i].vx) + (h*(E_x+(2*p[i].vy*alpha))) + ((h**2)*(alpha*E_y)) + (dt*(alpha**2)*(p[i].x+(h*alpha*p[i].y))))
        v_y_n_phalf = k*(((1-((h**2)*(alpha**2)))*p[i].vy) + (h*(E_y-(2*p[i].vx*alpha))) - ((h**2)*(alpha*E_x)) + (dt*(alpha**2)*(p[i].y-(h*alpha*p[i].x))))

        #Check particles are not exceeding speed of light
        if v_x_n_phalf > con.c:
            raise Exception('Particle x-velocity = {}m/s \nExceeding speed of light! \n'.format(v_x_n_phalf))
        elif v_y_n_phalf > con.c:
            raise Exception('Particle y-velocity = {}m/s \nExceeding speed of light! \n'.format(v_y_n_phalf))
        else:
            pass

        #Update particle velocity
        p[i].vx = v_x_n_phalf
        p[i].vy = v_y_n_phalf

        #WAVES TEST#############################################################
        """
        func = ((40)*np.cos((20*t[n])))*(np.exp(-1*p[i].x))
        if n > 199:
            if func > 0:
                p[i].vx += func
            else:
                pass
        """
        ########################################################################

        #Pre-step 3: Calculate ion flow velocities at U^-
        U_x_minus[g[i].x_i][g[i].y_i] += (p[i].vx*g[i].f_ij)/(dx*dy)
        U_x_minus[g[i].x_i+1][g[i].y_i] += (p[i].vx*g[i].f_i1j)/(dx*dy)
        U_x_minus[g[i].x_i+1][g[i].y_i+1] += (p[i].vx*g[i].f_i1j1)/(dx*dy)
        U_x_minus[g[i].x_i][g[i].y_i+1] += (p[i].vx*g[i].f_ij1)/(dx*dy)

        U_y_minus[g[i].x_i][g[i].y_i] += (p[i].vy*g[i].f_ij)/(dx*dy)
        U_y_minus[g[i].x_i+1][g[i].y_i] += (p[i].vy*g[i].f_i1j)/(dx*dy)
        U_y_minus[g[i].x_i+1][g[i].y_i+1] += (p[i].vy*g[i].f_i1j1)/(dx*dy)
        U_y_minus[g[i].x_i][g[i].y_i+1] += (p[i].vy*g[i].f_ij1)/(dx*dy)

        #Step 2: Advance particle position using the definition of velocity
        p[i].x = float(PostionUpdater(dt,p[i].x,p[i].vx))
        p[i].y = float(PostionUpdater(dt,p[i].y,p[i].vy))

        #Check position and obtain particle index numbers that require removing
        #from particle list
        #Open / clear variables for use
        x_i = 0
        y_i = 0
        #Calculate grid locations of particle
        x_i = int(np.floor((p[i].x - x_min) / dx))
        y_i = int(np.floor((p[i].y - y_min) / dy))

        #Move particles from one edge of model domain to other conserving particle
        #properties (periodic boundaries)
        if parms.options.boundaries.periodic == True:
            #If outside x domain
            if x_i < 0:
                p[i].x = x_max - (dx/10)
            elif x_i > int(nx-1):
                p[i].x = x_min + (dx/10)
            else:
                pass
            #If outside y domian
            if y_i < 0:
                p[i].y = y_max - (dy/10)
            elif y_i > int(ny-1):
                p[i].y = y_min + (dy/10)
            else:
                pass

        #If particles move outside of model domain remove from particle list
        #(open boundaries)
        elif parms.options.boundaries.open == True:
            #If outside x domain delete
            if x_i < 0 or x_i > int(nx-1):
                #DEBUG: Print which particle is removed
                print('Out of range: {}'.format(i))
                rm.append(i)
            #If outside y domian delete
            elif y_i < 0 or y_i > int(ny-1):
                #DEBUG: Print which particle is removed
                print('Out of range: {}'.format(i))
                rm.append(i)
            else:
                pass
            #Remove particles out of simulation area using reference list 'rm'
            for iter in sorted(rm, reverse=True):
                del p[iter]
            #Calculate number of particles that require regenerating
            Regenerate_num = T_PPC - len(p)
            r_p = []
            #If particles require regenerating then activate loop to generate insertion
            #parameters for as many new particles as required
            if Regenerate_num > 0:
                for re in range(0,Regenerate_num):
                    #This works in the same way as the loop in the pre-process
                    i = ran.randint(0,int(len(q_c)-1))
                    r_par_r = ran.uniform(r_min,r_max)
                    r_par_phi = ran.uniform(phi_min,phi_max)
                    r_par_x = r_par_r*np.cos(r_par_phi)
                    r_par_y = r_par_r*np.sin(r_par_phi)
                    r_par_vx = (ran.uniform(v_min,v_max))
                    r_par_vy = (ran.uniform(v_min,v_max))
                    r_par_q = q_c[i]
                    r_par_m = m[i]

                    p.append(Par._2D(r_par_x,r_par_y,r_par_vx,r_par_vy,r_par_q,r_par_m))

    #Step 3: Interpolate current density
    g = [] #Clear list for storing grid properties for corresponding particles
    q_grid_n_pone = np.zeros_like(q_grid_n) #Clean array for storing charge density at advanced time step
    U_x_plus = np.zeros(((nx + 1),(ny + 1))) #Ion flow velocity in x domain at mixed time step (v^n+1/2,x^n+1)
    U_y_plus = np.zeros(((nx + 1),(ny + 1))) #Ion flow velocity in y domain at mixed time step (v^n+1/2,x^n+1)
    J_x_plus = np.zeros(((nx + 1),(ny + 1))) #Charge density in x domain at mixed time step (\rho_c^n+1,v^n+1/2,x^n+1)
    J_y_plus = np.zeros(((nx + 1),(ny + 1))) #Charge density in y domain at mixed time step (\rho_c^n+1,v^n+1/2,x^n+1)
    Lambda_n_pone = np.zeros(((nx + 1),(ny + 1)))
    Gamma_x_n_pone = np.zeros(((nx + 1),(ny + 1)))
    Gamma_y_n_pone = np.zeros(((nx + 1),(ny + 1)))

    #Loop through advanced particles to interpolate bulk parameters
    for i in range(0,len(p)):

        #Calculate grid location of particle
        x_i = int(np.floor((p[i].x - x_min)/ dx))
        y_i = int(np.floor((p[i].y - y_min)/ dy))

        #Calculate distance from grid index vertex
        hx = p[i].x - x[x_i]
        hy = p[i].y - y[y_i]

        #Calculate Cloud-in-cell weighting
        f_ij = (((dx - hx) * (dy - hy)) / (dx * dy))
        f_i1j = ((hx * (dy - hy)) / (dx * dy))
        f_i1j1 = ((hx * hy) / (dx * dy))
        f_ij1 = (((dx - hx)* hy) / (dx * dy))

        #Store particle grid properties
        g.append(Grid_Info._2D(x_i,y_i,f_ij,f_i1j,f_i1j1,f_ij1))

        #Obtain charge density at n + 1
        q_grid_n_pone[x_i][y_i] += ((p[i].q)*f_ij)/(dx*dy)
        q_grid_n_pone[x_i+1][y_i] += ((p[i].q)*f_i1j)/(dx*dy)
        q_grid_n_pone[x_i+1][y_i+1] += ((p[i].q)*f_i1j1)/(dx*dy)
        q_grid_n_pone[x_i][y_i+1] += ((p[i].q)*f_ij1)/(dx*dy)

        #Obtain mixed time step ion flow velocities
        U_x_plus[x_i][y_i] += (p[i].vx*f_ij)/(dx*dy)
        U_x_plus[x_i+1][y_i] += (p[i].vx*f_i1j)/(dx*dy)
        U_x_plus[x_i+1][y_i+1] += (p[i].vx*f_i1j1)/(dx*dy)
        U_x_plus[x_i][y_i+1] += (p[i].vx*f_ij1)/(dx*dy)

        U_y_plus[x_i][y_i] += (p[i].vy*f_ij)/(dx*dy)
        U_y_plus[x_i+1][y_i] += (p[i].vy*f_i1j)/(dx*dy)
        U_y_plus[x_i+1][y_i+1] += (p[i].vy*f_i1j1)/(dx*dy)
        U_y_plus[x_i][y_i+1] += (p[i].vy*f_ij1)/(dx*dy)

        #Obtain other variables needed for CAM
        Lambda_n_pone[x_i][y_i] += ((p[i].q**2)/p[i].m)*f_ij
        Lambda_n_pone[x_i+1][y_i] += ((p[i].q**2)/p[i].m)*f_i1j
        Lambda_n_pone[x_i+1][y_i+1] += ((p[i].q**2)/p[i].m)*f_i1j1
        Lambda_n_pone[x_i][y_i+1] += ((p[i].q**2)/p[i].m)*f_ij1

        Gamma_x_n_pone[x_i][y_i] += ((p[i].q**2)/p[i].m)*p[i].vx*f_ij
        Gamma_x_n_pone[x_i+1][y_i] += ((p[i].q**2)/p[i].m)*p[i].vx*f_i1j
        Gamma_x_n_pone[x_i+1][y_i+1] += ((p[i].q**2)/p[i].m)*p[i].vx*f_i1j1
        Gamma_x_n_pone[x_i][y_i+1] += ((p[i].q**2)/p[i].m)*p[i].vx*f_ij1

        Gamma_y_n_pone[x_i][y_i] += ((p[i].q**2)/p[i].m)*p[i].vy*f_ij
        Gamma_y_n_pone[x_i+1][y_i] += ((p[i].q**2)/p[i].m)*p[i].vy*f_i1j
        Gamma_y_n_pone[x_i+1][y_i+1] += ((p[i].q**2)/p[i].m)*p[i].vy*f_i1j1
        Gamma_y_n_pone[x_i][y_i+1] += ((p[i].q**2)/p[i].m)*p[i].vy*f_ij1

    #CAM: Obtain mixed time current densities
    J_x_plus = q_grid_n_pone*U_x_plus
    J_y_plus = q_grid_n_pone*U_y_plus

    #Step 3.5: Obtain flow velocities on the grid points
    #TODO: Needs replacing with proper solution for low denisty areas, nabla^2(E) = 0
    q_grid_n_eq_0 = np.where(q_grid_n==0)
    q_grid_n_neq_0 = np.where(q_grid_n!=0)
    q_grid_n_pone_eq_0 = np.where(q_grid_n_pone==0)
    q_grid_n_pone_neq_0 = np.where(q_grid_n_pone!=0)
    #TODO: Remove need for this
    q_grid_n_pone[q_grid_n_pone==0] = np.average(q_grid_n_pone/100)

    #Obtain ion flow velocity on the half-integer time step by averaging mixed
    #time flows
    U_x_n_phalf = (1/2)*(U_x_minus + U_x_plus)
    U_y_n_phalf = (1/2)*(U_y_minus + U_y_plus)

    #Obtain charge densities on the half-integer time step by averaging current
    #and advanced variables
    q_grid_n_phalf = (1/2)*(q_grid_n + q_grid_n_pone)
    q_grid_n_phalf_neq_0 = np.where(q_grid_n_phalf!=0)
    q_grid_n_phalf_eq_0 = np.where(q_grid_n_phalf==0)

    #Step 4: Advance mangetic field by half a time step by utilising Faraday's eqn
    B_z_ind_n_phalf =  Magnetic_MacCormack(dx,dy,dt,B_z_ind_n,U_x_n,U_y_n)

    #Obtain full magnetic field at half integer time step by combining induced
    #field with backgroud
    B_z_n_phalf = B_z_back + B_z_ind_n_phalf

    #Step 5:Advance electric field by half a time step utilising the Electron
    #momentum eqn
    E_x_new = -(U_y_n_phalf*B_z_n_phalf) - ((1/q_grid_n_phalf)*(diff.x(((B_z_n_phalf**2)/2*con.mu_0),dx,Boundary_Type='Periodic')))
    E_y_new = (U_x_n_phalf*B_z_n_phalf) - ((1/q_grid_n_phalf)*(diff.y(((B_z_n_phalf**2)/2*con.mu_0),dy,Boundary_Type='Periodic')))

    #Advance pressure field to the half integer time step
    #Removed in current version for stability

    #Obtain full electric field at half integer time step by combining induced
    #field with backgroud
    E_x_n_phalf = E_x_back + (E_x_new)
    E_y_n_phalf = E_y_back + (E_y_new)

    if len(q_grid_n_phalf_eq_0) == 2 and len(q_grid_n_phalf_eq_0[0]) > 0:
        #Use nabla^2(E)=0 to calculate electric field where charge density tends to
        #zero
        E_x_n_phalf_low_den = Vector_laplacian_solver._2D.x(E_x_n_phalf,E_y_n_phalf,dx,dy)
        E_y_n_phalf_low_den = Vector_laplacian_solver._2D.y(E_x_n_phalf,E_y_n_phalf,dx,dy)

        for i in range(0,len(q_grid_n_phalf_eq_0[0])):
            E_x_n_phalf[int(q_grid_n_phalf_eq_0[0][i])][int(q_grid_n_phalf_eq_0[1][i])] = E_x_n_phalf_low_den[int(q_grid_n_phalf_eq_0[0][i])][int(q_grid_n_phalf_eq_0[1][i])]
            E_y_n_phalf[int(q_grid_n_phalf_eq_0[0][i])][int(q_grid_n_phalf_eq_0[1][i])] = E_y_n_phalf_low_den[int(q_grid_n_phalf_eq_0[0][i])][int(q_grid_n_phalf_eq_0[1][i])]
    else:
        pass

    #Step 6: Advance magnetic field an additional half time step in order to get
    #the full integer advance in time required
    B_z_ind_n = Magnetic_MacCormack(dx,dy,dt,B_z_ind_n_phalf,U_x_n_phalf,U_y_n_phalf)

    #Obtain magnetic field on advanced time step by combining induced field with
    #backgroud
    B_z_n = B_z_back + B_z_ind_n

    #CAM step: Calculate mixed time step eletric field
    E_x_new = -(U_y_plus*B_z_n) - ((1/q_grid_n_pone)*(diff.x(((B_z_n**2)/2*con.mu_0),dx,Boundary_Type='Periodic')))
    E_y_new = (U_x_plus*B_z_n) - ((1/q_grid_n_pone)*(diff.y(((B_z_n**2)/2*con.mu_0),dy,Boundary_Type='Periodic')))

    #Obtain full electric field at mixed time step by combining induced field with
    #backgroud
    E_x_mix = E_x_back + (E_x_new)
    E_y_mix = E_y_back + (E_y_new)

    if len(q_grid_n_pone_eq_0) == 2 and len(q_grid_n_pone_eq_0[0]) > 0:
        #Use nabla^2(E)=0 to calculate electric field where charge density tends to
        #zero
        E_x_mix_low_den = Vector_laplacian_solver._2D.x(E_x_mix,E_y_mix,dx,dy)
        E_y_mix_low_den = Vector_laplacian_solver._2D.y(E_x_mix,E_y_mix,dx,dy)

        for i in range(0,len(q_grid_n_pone_eq_0[0])):
            E_x_mix[int(q_grid_n_pone_eq_0[0][i])][int(q_grid_n_pone_eq_0[1][i])] = E_x_mix_low_den[int(q_grid_n_pone_eq_0[0][i])][int(q_grid_n_pone_eq_0[1][i])]
            E_y_mix[int(q_grid_n_pone_eq_0[0][i])][int(q_grid_n_pone_eq_0[1][i])] = E_y_mix_low_den[int(q_grid_n_pone_eq_0[0][i])][int(q_grid_n_pone_eq_0[1][i])]
    else:
        pass

    #Use mixed time electric field to advance current densities
    J_x_n = J_x_plus + (dt/2)*((Lambda_n_pone*E_x_mix)+(Gamma_y_n_pone*B_z_n))
    J_y_n = J_y_plus + (dt/2)*((Lambda_n_pone*E_y_mix)-(Gamma_x_n_pone*B_z_n))

    #Calculate ion flow velocites on advanced time step
    U_x_n[q_grid_n_pone_neq_0] = J_x_n[q_grid_n_pone_neq_0]/q_grid_n_pone[q_grid_n_pone_neq_0]
    U_y_n[q_grid_n_pone_neq_0] = J_y_n[q_grid_n_pone_neq_0]/q_grid_n_pone[q_grid_n_pone_neq_0]
    U_x_n[q_grid_n_pone_eq_0] = 0.
    U_y_n[q_grid_n_pone_eq_0] = 0.

    #Step 8: Advance electric field an additional half time step
    #TODO: Dropping mu_0 in front of q_grid_n_pone as unclear if needed CHECK
    E_x_new = -(U_y_n*B_z_n) - ((1/q_grid_n_pone)*(diff.x(((B_z_n**2)/2*con.mu_0),dx,Boundary_Type='Periodic')))
    E_y_new = (U_x_n*B_z_n) - ((1/q_grid_n_pone)*(diff.y(((B_z_n**2)/2*con.mu_0),dy,Boundary_Type='Periodic')))

    #Obtain full electric field at advanced time step by combining induced field with
    #backgroud
    E_x_n = E_x_back + (E_x_new)
    E_y_n = E_y_back + (E_y_new)

    if len(q_grid_n_pone_eq_0) == 2 and len(q_grid_n_pone_eq_0[0]) > 0:
        #Use nabla^2(E)=0 to calculate electric field where charge density tends to
        #zero
        E_x_n_low_den = Vector_laplacian_solver._2D.x(E_x_n,E_y_n,dx,dy)
        E_y_n_low_den = Vector_laplacian_solver._2D.y(E_x_n,E_y_n,dx,dy)

        for i in range(0,len(q_grid_n_pone_eq_0[0])):
            E_x_n[int(q_grid_n_pone_eq_0[0][i])][int(q_grid_n_pone_eq_0[1][i])] = E_x_n_low_den[int(q_grid_n_pone_eq_0[0][i])][int(q_grid_n_pone_eq_0[1][i])]
            E_y_n[int(q_grid_n_pone_eq_0[0][i])][int(q_grid_n_pone_eq_0[1][i])] = E_y_n_low_den[int(q_grid_n_pone_eq_0[0][i])][int(q_grid_n_pone_eq_0[1][i])]
    else:
        pass

    #Advance pressure field to advanced time step
    #Removed for stability

    #Clean up & advance any variables required
    q_grid_n = q_grid_n_pone.copy()
    B_z_ind_n_mhalf = B_z_ind_n_phalf.copy()
    U_x_n_mhalf = U_x_n_phalf.copy()
    U_y_n_mhalf = U_y_n_phalf.copy()

    if parms.options.IO_method.pickle == True:
        if n ==1 or n % 100 == 0:
            E_mag_wb.append(np.sqrt((E_x_n**2)+(E_y_n**2)))
            q_grid_wb.append(q_grid_n)
            B_mag_wb.append(B_z_n)

    if n == 1:
        toc = timeit.default_timer()
        step_time =(toc - tic)
        run_time = round(step_time*end_step,2)
        print('Estimated run time: {}s'.format(run_time))

    #Plotting ##################################################################

    if parms.options.plotting.on == True:
        if n % 10 == 0:
            E_mag_n = np.sqrt((E_x_n**2)+(E_y_n**2))
            plt.clf()
            plt.subplot(1,2,1)
            plt.contourf((Y),(X),E_mag_n)
            plt.colorbar(label='$|E|$')
            plt.xlabel('x (m)')
            plt.ylabel('y (m)')
            plt.subplot(1,2,2)
            plt.contourf((Y),(X),B_z_n)
            plt.colorbar(label='$|B|$')
            plt.xlabel('x (m)')
            plt.ylabel('y (m)')
            plt.draw()
            plt.pause(0.0001)

    #IO ########################################################################
    if parms.options.IO_method.HDF5 ==True:
        if dataio.ready_steps(n):
            p_vx = np.zeros(len(p))
            p_vy = np.zeros(len(p))
            for i in range(0,len(p)):
                p_vx[i] = p[i].vx
                p_vy[i] = p[i].vy
            dataio.store_log('step', n)
            dataio.store_log('time', t[n])
            dataio.store_log('cfl', (((np.abs(np.average(p_vx))*dt)/dx)+((np.abs(np.average(p_vy))*dt)/dy)))
            dataio.store_log('divJ', np.average(diff.x(J_x_n,dx)+diff.y(J_y_n,dy)))
            dataio.store_log('divB',  0.0)
            dataio.store_log('gauss_law', np.average((diff.x(E_x_n,dx)+diff.y(E_y_n,dy))-(q_grid_n/con.epsilon_0)))
            dataio.store_log('num_particles', len(p))
            dataio.next_log()

        if dataio.ready_fields(n):
            dataio.store_field('Ex', E_x_n)
            dataio.store_field('Ey', E_y_n)
            dataio.next_fields()
            # write E_x_n, E_y_n, q_grid_n, B_z_ind, U_x_n, p_n

        if dataio.ready_all_particles(n):
            dataio.store_particle_dataset(n, p)

        #if n % 10 == 0:
            #oFile = open('E_field.p', 'wb')
            #pickle.dump(E_mag_n, oFile)

if parms.options.IO_method.HDF5 ==True:
    dataio.close()
elif parms.options.IO_method.pickle == True:
    pickle.dump(t,open('t_vect.p','wb'))
    pickle.dump(E_mag_wb, open('E_mag.p','wb'))
    pickle.dump(q_grid_wb, open('q_grid.p','wb'))
    pickle.dump(B_mag_wb, open('B_mag.p','wb'))
else:
    pass

print('Simulation complete.')

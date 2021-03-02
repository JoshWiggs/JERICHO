"""
Module containing custom classes used in the model to optimize memory usage and maximum computational speed. These custom classes make use of the dataclass functionality from the standard dataclasses python library in order to pre-allocate class instance attributes via __slots__. The expected input datatype for each of these attributes is also predefined, however not prescriptive as python still allows for dynamic datatyping.

 .. moduleauthor:: J. A. Wiggs <j.wiggs@lancaster.ac.uk>

Dependencies
------------

* Dataclasses: From the python standard library, this module provides a decorator and functions for automatically adding generated special methods such as __init__() and __repr__() to user-defined classes. It was originally described in PEP 557.
"""
from dataclasses import dataclass
@dataclass
class Particle_Information:
    """
    This class contains all the subclasses used to contain information required to propagate a simulated particle through the module in both two and three dimensions.
    """
    @dataclass
    class _2D:
        """
        Information required to be stored for each particle in the module in order to move it through a two-dimensional simulated space.

        **Arguments**

        x_position: float
         |  Position of the particle in the x-domain
         |  Attribute: x

        y_position: float
         |  Position of the particle in the y-domain
         |  Attribute: y

        x_velocity: float
         |  Speed of particle motion in the x-domain
         |  Attribute: vx

        y_velocity: float
         |  Speed of particle motion in the y-domain
         |  Attribute: vy

        charge: float
         |  Charge of particle
         |  Attribute: q

        mass: float
         |  Mass of particle
         |  Attribute: m
        """
        __slots__ = ['x','y','vx','vy','q','m']
        def __init__(self,x_position: float,y_position: float,x_velocity: float,y_velocity: float,charge: float,mass: float):
          self.x = x_position
          self.y = y_position
          self.vx = x_velocity
          self.vy = y_velocity
          self.q = charge
          self.m = mass

    @dataclass
    class _3D:
        """
        Information required to be stored for each particle in the module in order to move it through a three-dimensional simulated space.

        **Arguments**

        x_position: float
         |  Position of the particle in the x-domain
         |  Attribute: x

        y_position: float
         |  Position of the particle in the y-domain
         |  Attribute: y

        z_position: float
         |  Position of the particle in the z-domain
         |  Attribute: z

        x_velocity: float
         |  Speed of particle motion in the x-domain
         |  Attribute: vx

        y_velocity: float
         |  Speed of particle motion in the y-domain
         |  Attribute: vy

        z_velocity: float
         |  Speed of particle motion in the z-domain
         |  Attribute: vz

        charge: float
         |  Charge of particle
         |  Attribute: q

        mass: float
         |  Mass of particle
         |  Attribute: m
        """
        __slots__ = ['x','y','z','vx','vy','vz','q','m']
        def __init__(self,x_position,y_position,z_position,x_velocity,y_velocity,z_velocity,charge,mass):
            self.x = x_position
            self.y = y_position
            self.z = z_position
            self.vx = x_velocity
            self.vy = y_velocity
            self.vz = z_velocity
            self.q = charge
            self.m = mass

@dataclass
class Particle_Grid_Information:
    """
    This class contains all the subclasses used to store information correlating particles to the electromagnetic grid within the simulation for both two and three dimensions. The parameters used for this correlation are the set of grid points lowest in numerical value that describe the grid cell a particle occupies (commonly referred to as the 'bottom left' of the cell). Also, the weighting factors, obtained via first order interpolation, used to collect particle properties on the grid points.
    """
    @dataclass
    class _2D:
        """
        Information required to be stored for each particle for interactions with the EM grids in two-dimensions.

        **Arguments**

        x_grid_position: int
         |  Lowest grid index of the cell a particle occupies in the x-domain
         |  Attribute: x_i

        y_grid_position: int
         |  Lowest grid index of the cell a particle occupies in the y-domain
         |  Attribute: y_i

        weight_ij: float
         |  Value of proportion of particle parameter given to the bottom left grid point
         |  Attribute: f_ij

         weight_i1j: float
          |  Value of proportion of particle parameter given to the bottom right grid point
          |  Attribute: f_i1j

         weight_i1j1: float
          |  Value of proportion of particle parameter given to the top right grid point
          |  Attribute: f_i1j1

         weight_ij1: float
          |  Value of proportion of particle parameter given to the top left grid point
          |  Attribute: f_i1j
        """
        __slot__ = ['x_i','y_i','f_ij','f_i1j','f_i1j1','f_ij1']
        def __init__(self,x_grid_position: int,y_grid_position: int,weight_ij: float,weight_i1j: float,weight_i1j1: float,weight_ij1: float):
            self.x_i = x_grid_position
            self.y_i = y_grid_position
            self.f_ij = weight_ij
            self.f_i1j = weight_i1j
            self.f_i1j1 = weight_i1j1
            self.f_ij1 = weight_ij1
    @dataclass
    class _3D:
        """
        Information required to be stored for each particle for interactions with the EM grids in three-dimensions.

        **Arguments**

        x_grid_position: int
         |  Lowest grid index of the cell a particle occupies in the x-domain
         |  Attribute: x_i

        y_grid_position: int
         |  Lowest grid index of the cell a particle occupies in the y-domain
         |  Attribute: y_i

        z_grid_position: int
         |  Lowest grid index of the cell a particle occupies in the z-domain
         |  Attribute: z_i

        weight_ijk: float
         |  Value of proportion of particle parameter given to the front bottom left grid point
         |  Attribute: f_ijk

         weight_i1jk: float
          |  Value of proportion of particle parameter given to the front bottom right grid point
          |  Attribute: f_i1jk

         weight_i1j1k: float
          |  Value of proportion of particle parameter given to the front top right grid point
          |  Attribute: f_i1j1k

         weight_ij1k: float
          |  Value of proportion of particle parameter given to the front top left grid point
          |  Attribute: f_i1jk

        weight_ijk1: float
         |  Value of proportion of particle parameter given to the rear bottom left grid point
         |  Attribute: f_ijk1

         weight_i1jk1: float
          |  Value of proportion of particle parameter given to the rear bottom right grid point
          |  Attribute: f_i1jk1

         weight_i1j1k1: float
          |  Value of proportion of particle parameter given to the rear top right grid point
          |  Attribute: f_i1j1k1

         weight_ij1k1: float
          |  Value of proportion of particle parameter given to the rear top left grid point
          |  Attribute: f_i1jk1
        """
        __slot__ = ['x_i','y_i','z_i','f_ijk','f_i1jk','f_i1j1k','f_ij1k','f_ijk1','f_i1jk1','f_i1j1k1','f_ij1k1']
        def __init__(self,x_grid_position: int,y_grid_position: int,z_grid_position: int,weight_ijk: float,weight_i1jk: float,weight_i1j1k: float,weight_ij1k: float,weight_ijk1: float,weight_i1jk1: float,weight_i1j1k1: float,weight_ij1k1: float):
            self.x_i = x_grid_position
            self.y_i = y_grid_position
            self.z_i = z_grid_position
            self.f_ijk = weight_ijk
            self.f_i1jk = weight_i1jk
            self.f_i1j1k = weight_i1j1k
            self.f_ij1k = weight_ij1k
            self.f_ijk1 = weight_ijk1
            self.f_i1jk1 = weight_i1jk1
            self.f_i1j1k1 = weight_i1j1k1
            self.f_ij1k1 = weight_ij1k1

@dataclass
class parameters:
    """
    Object for holding the variables used to initialise JERICHO-PY and select the features that are needed for a simulation run. These options and values are assigned in a separate 'parameters' file.
    """
    @dataclass
    class options:
        """
        Set of classes that form a number of switches which turn off and on features in the code.
        """
        class boundaries:
            """
            Choose between either periodic or open boundaries

            **Arguments**

            periodic: Bool

            open: Bool
            """
            __slots__ = ['periodic','open']
            periodic: bool
            open: bool
        class coords:
            """
            Turn on set of calculations which transform the cartesian co-ordinates into polar cylindrical, these can then be accessed for plotting

            **Arguments**

            radial: Bool
            """
            __slots__ = ['radial']
            radial: bool
        class particle_distribution:
            """
            Choose initialisation technique for assigning particle speeds, uniform randomly distributes between 2 pre-defined values whereas Maxwellian injects all particles at the same temperature

            **Arguments**

            uniform: Bool

            Maxwellian: Bool
            """
            __slots__ = ['uniform','Maxwellian']
            uniform: bool
            Maxwellian: bool
        class electric_initialisation:
            """
            Choose if the initial electric field configuration is calculated using either the electrostatic approximation or assuming an eletrodynamic set-up

            **Arguments**

            electrostatic: Bool

            electrodynamic: Bool
            """
            __slots__ = ['electrostatic','electrodynamic']
            electrostatic: bool
            electrodynamic: bool
        class IO_method:
            """
            Switches to activate the in-built IO methods

            **Arguments**

            pickle: Bool

            HDF5: Bool
            """
            __slots__ = ['pickle','HDF5']
            pickle: bool
            HDF5: bool
        class plotting:
            """
            Switch to turn in-line plotting on and off, this will activate a plotting procedure that visualises the model data as the model runs

            **Arguments**

            on: Bool
            """
            __slots__ = ['on']
            on: bool
        class modules:
            """
            Switch to turn optional modules on and off. Constants refers to the scipy.constants module.

            **Arguments**

            constants: Bool
            """
            __slots__ = ['constants']
            constants: bool
        class mag_field_type:
            """
            Choose between magnetic field initialisation techniques, dipole will calculate a dipoler field for the modelled region whereas uniform will set the field to a single background value

            **Arguments**

            dipole: Bool

            uniform: Bool
            """
            __slots__ = ['dipole','uniform']
            dipole: bool
            uniform: bool
    @dataclass
    class var:
        """
        General variable controlling the geometry of the modelled domain and the plasma simulated inside it. The units assumed are indicated in brackets after the variable description.

        x_min: float
            Location of the start of the x-axis (m)
        x_max: float
            Location of the end of the x-axis (m)
        y_min: float
            Location of the start of the y-axis (m)
        y_max: float
            Location of the end of the y-axis (m)
        nx: int
            Number of vertices used to discretise the EM fields in the x-direction
        ny: int
            Number of vertices used to discretise the EM fields in the y-direction
        t_min: float
            Start time of the simulation (s)
        t_end: float
            End time of the simulation (s)
        nt: int
            Number of steps taken in the temporal domain to progress from the start to the end time
        PPC: int
            Particles per cell, the ratio of particles to the number of grid cell constructed in the spatial domain
        T_i: float
            (Optional) If using the Maxwellian initialisation option, this is the temperature at which the particles are initialised (k)
        q_c: list
            List of charges for different ion species forming the simulated plasma (C)
        m: list
            List of masses for different ion species forming the simulated plasma (kg)
        n_smoothing: int
            Some optional functions use recursive self-smoothing, this controls the number of loops these functions use to perform this smoothing
        """
        __slots__ = ['x_min','x_max','y_min','y_max','nx','ny','t_min','t_max','nt','q_c','m','n_smoothing','PPC']
        #Spatial domain variables (SI)
        x_min: float
        x_max: float
        y_min: float
        y_max: float
        #Spatial grid Size
        nx: int
        ny: int
        #Temporal domain variables (SI)
        t_min: float
        t_max: float
        nt: int
        #Plasma parameters
        PPC: int
        T_i = float
        #Particle charge & mass lists
        q_c: list
        m: list
        #Other parameters
        n_smoothing: int
    @dataclass
    class planetary:
        """
        Variables that contain the properties of the planet you wish to simulate, notation assumes Jupiter since that is the target of the model. The units assumed are indicated in brackets after the variable description.

        R_J: float
            Planetary radius (m)
        M_J: float
            Planetary mass (kg)
        Omega: float
            Planetary rotational velocity, in the case of where rotational velocity differ depending on atmospheric height, it can be assumed to be ionospheric rotational velocity (rad/s)
        B_eq: float
            Equatorial magnetic field strength (T)
        """
        __slots__ = ['R_J','M_J','Omega','B_eq']
        R_J: float
        M_J: float
        Omega: float
        B_eq: float

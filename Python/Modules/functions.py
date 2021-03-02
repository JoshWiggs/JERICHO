"""
Module containing set of simple 'helper' functions and more complex 'advanced' functions utilised in the main model to perform various operations.

 .. moduleauthor:: J. A. Wiggs <j.wiggs@lancaster.ac.uk>

Dependencies
------------

* Math: From the python stand library, this module provides access to the mathematical functions defined by the C standard.
* Numpy_: From the python package index, a fundamental package for scientific computing with Python.
* Matplotlib_: From the python package index, a comprehensive library for creating static, animated, and interactive visualizations in Python.
* numerical_differentials: Set of numerical differentiators built bespoke for this project.

.. _Numpy: https://numpy.org/
.. _Matplotlib: https://matplotlib.org/
"""

import math
import numpy as np
from Modules.numerical_differentials import *
try:
    import matplotlib.pyplot as plt
except:
    pass

"""
Helper functions
----------------
"""

def Set2DBoundaries(array,value):
    """
    Set elements along the 4 edges of a 2-dimesional array to same value.

    **Arguments**

    array
        2-dimensional array which will have its edge elements altered by the function

    value
        The value, usually numerical (int or float), that the edge elements will be set to

    **Dependencies**

    np
        Numpy imported with np as alias
    """
    dimension_number = int(len(np.shape(array)))
    if dimension_number != 2:
        raise Exception('Set2DBoundaries only accepts 2-dimensional arrays as inputs')
    else:
        array[:,0] = value
        array[:,-1] = value
        array[0,:] = value
        array[-1,:] = value
def Set3DBoundaries(array,value):
    """
    Set elements along the 6 faces of a 3-dimesional array to same value.

    **Arguments**

    array
        3-dimensional array which will have its edge elements altered by
        the function

    value
        The value, usually numerical (int or float), that the edge
        elements will be set to

    **Dependencies**

    np
        Numpy imported with np as alias
    """
    dimension_number = int(len(np.shape(array)))
    if dimension_number != 3:
        raise Exception('Set3DBoundaries only accepts 3-dimensional arrays as inputs')
    else:
        array[:,:,0] = value
        array[:,:,-1] = value
        array[:,0,:] = value
        array[:,-1,:] = value
        array[0,:,:] = value
        array[-1,:,:] = value
def PostionUpdater(dt,x_n,v_n_phalf):
    """
    Returns the updated position value of a particle by taking the current position of the particle and calculating the distance to move it utilising the size of the step taken in time and the particle's velocity.

    **Arguments**

    dt
        Size of time step
    x_n
        Particle's current position
    v_n_phalf
        Velocity value used to calculate distance moved
    """
    x_n_pone = x_n + (dt*v_n_phalf)
    return x_n_pone
def contour2D(X,Y,array):
    """
    Time saving plotting function that displays a filled 2-dimensional contour plot of a given array. No need for error messages to be included as matplotlib takes care of that already.

    **Arguments**

    X,Y
        2-dimesional meshgrid's of the 2 co-ordinate vectors
    array
        2-dimensional array that will be used to create the contour plot

    **Dependencies**

    plt
        matplotlib.pyplot imported with the alias plt
    """
    plt.contourf(Y,X,array)
    plt.colorbar()
    plt.show()
def Integrate3Dto2D(array):
    """
    Converts a 3-dimensional array to a 2-dimensional array by integrating in the z-domian. Highly useful for when plotting 3-dimensional fields.

    **Arguments**

    array
        3-dimensional that will be returned as a 2-dimensional array

    **Dependencies**

    np
        Numpy imported with np as alias
    """
    dimension_number = int(len(np.shape(array)))
    if dimension_number != 3:
        raise Exception('Integrate3Dto2D only accepts 3-dimensional arrays as inputs')
    else:
        array_2d = np.zeros(((len(array[0])),(len(array[0][0]))))
        for i in range(0,len(array)):
            array_2d += array[i,:,:]
        return array_2d
def Integrate2Dto1D(array):
    """
    Converts a 2-dimensional array to a 1-dimensional list by integrating in the y-domian. Highly useful for when plotting 2-dimensional fields.

    **Arguments**

    array
        2-dimensional that will be returned as a 1-dimensional array

    **Dependencies**

    np
        Numpy imported with np as alias
    """
    dimension_number = int(len(np.shape(array)))
    if dimension_number != 2:
        raise Exception('Integrate3Dto2D only accepts 3-dimensional arrays as inputs')
    else:
        array_1d = np.zeros(len(array[0]))
        for i in range(0,len(array)):
            array_1d += array[:,i]
        return array_1d
def contour3D(x,y,array):
    """
    Time saving plotting function that displays a filled 2-dimensional contour plot of a given array. Accepts a 3-dimensional array and utilises the Integrate3Dto2D function that integrates the given array in the z-domain to reduce it to a 2-dimensional array. No need for error messages to be included as matplotlib takes care of that already.

    **Arguments**

    x,y
        Dimesional vectors of the 2 co-ordinate vectors

    array
        3-dimensional array that will be used to create the contour plot

    **Dependencies**

    plt
        matplotlib.pyplot imported with the alias plt
    """
    array_2d = Integrate3Dto2D(array)
    plt.contourf(x,y,array_2d)
    plt.colorbar()
    plt.show()
def orderOfMagnitude(number):
    """
    Return the mathematical order of magnitude of a given value

    **Arguments**

    number
        The variable containing the value whose order of magnitude will
        be returned

    **Dependencies**

    math
        Python's native math module is imported
    """
    return math.floor(math.log(number, 10))
class plot_tracer:
    """
    Time saving plotting functions that traces the path a particle, on a pre-chosen index in the particle list, through the simulated domain. The list of tracer particle positions is taken from global memory for use. No need for error messages to be included as matplotlib takes care of that
    """
    def _2D(tracer_x,tracer_y):
        """
        2-dimensional implementation

        **Arguments**

        tracer_x
            List of tracer particle positions in the x-domain

        tracer_y
            List of tracer particle positions in the y-domain

        **Dependencies**

        plt
            matplotlib.pyplot imported with the alias plt
        """
        plt.plot(tracer_x,tracer_y)
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.show()
    def _3D(tracer_x,tracer_y,tracer_z):
        """
        3-dimensional implementation

        **Arguments**

        tracer_x
            List of tracer particle positions in the x-domain

        tracer_y
            List of tracer particle positions in the y-domain

        tracer_z
            List of tracer particle positions in the z-domain

        **Dependencies**

        plt
            matplotlib.pyplot imported with the alias plt
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(tracer_x,tracer_y,tracer_z)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$z$')
        plt.show()
    def radial(tracer_phi,tracer_r):
        """
        Handles plotting for co-ordinates that have been transformed into polar-cylindrical

        **Arguments**

        tracer_phi
            List of tracer particle positions in the phi-domain (azimuthal) with co-ordinates transformed into polar-cylindrical

        tracer_r
            List of tracer particle positions in the r-domain (radial) with co-ordinates transformed into polar-cylindrical

        **Dependencies**

        plt
            matplotlib.pyplot imported with the alias plt
        """
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='polar')
        ax.plot(tracer_phi,tracer_r)
        ax.set_thetamin(0)
        ax.set_thetamax(90)
        ax.set_rmin(np.min(tracer_r))
        ax.set_rmax(np.max(tracer_r))
        plt.show()

"""
Advanced functions
------------------
"""
class Poisson_solver:
    """
    Provides an iterative numerical solution to the well-known Poisson equation.
    """
    def _2D(output,source,delta_1,delta_2):
        """
        2-dimensional implementation

        **Arguments**

        output
            Pre-defined array that will contain the solution to the equation

        source
            Array containing the field that will be operated on

        delta_1
            Size of step taken in position co-ordinate corresponding to the row direction of the array

        delta_2
            Size of step taken in position co-ordinate corresponding to the column direction of the array
        """
        n_smoothing = 20
        d1_2 = delta_1**2
        d2_2 = delta_2**2
        for loop_number in range(0,n_smoothing):
            output[1:-1,1:-1] = ((d1_2*d2_2*(source[1:-1,1:-1])) + (d2_2*(output[1:-1,2:]+output[1:-1,:-2])) + (d1_2*(output[2:,1:-1]+output[:-2,1:-1])))/(2*(d1_2+d2_2))
        return output
class Vector_laplacian_solver:
    """
    Provides a numeircal solution to the vector laplacian for vector fields.
    """
    class _2D:
        """
        Solution in 2-dimensions (x,y), with the x co-ordinate corresponding to the row direction of the array and the y to the column direction of the array.
        """
        def x(vector_dimension_1,vector_dimension_2,delta_1,delta_2):
            """
            Solver for the x-dimension

            **Arguments**

            vector_dimension_1
                Component of vector field in position co-ordinate corresponding to the row direction of the array

            vector_dimension_2
                Component of vector field in position co-ordinate corresponding to the column direction of the array

            delta_1
                Size of step taken in position co-ordinate corresponding to the row direction of the array

            delta_2
                Size of step taken in position co-ordinate corresponding to the column direction of the array
            """
            output = np.zeros_like(vector_dimension_1)
            output[1:-1:,1:-1] = ((vector_dimension_1[1:-1,2:]+vector_dimension_1[1:-1,:-2])/2)
            + ((delta_1*(vector_dimension_2[2:,2:]-vector_dimension_2[:-2,2:]-vector_dimension_2[2:,:-2]+vector_dimension_2[:-2,:-2]))/(8*delta_2))
            return output
        def y(vector_dimension_1,vector_dimension_2,delta_1,delta_2):
            """
            Solver for the y-dimension

            **Arguments**

            vector_dimension_1
                Component of vector field in position co-ordinate corresponding to the row direction of the array

            vector_dimension_2
                Component of vector field in position co-ordinate corresponding to the column direction of the array

            delta_1
                Size of step taken in position co-ordinate corresponding to the row direction of the array

            delta_2
                Size of step taken in position co-ordinate corresponding to the column direction of the array
            """
            output = np.zeros_like(vector_dimension_1)
            output[1:-1:,1:-1] = ((vector_dimension_2[2:,1:-1]+vector_dimension_2[:-2,1:-1])/2)
            + ((delta_2*(vector_dimension_1[2:,2:]-vector_dimension_1[:-2,2:]-vector_dimension_1[2:,:-2]+vector_dimension_1[:-2,:-2]))/(8*delta_1))
            return output
class Lax_time_step:
    """
    Performs the well known Lax method on input field to improve numerical stability in 2-dimensions.
    """
    def _2D(array):
        """
        2-dimensional implementation

        **Arguments**

        array
            Matrix containing the field being progressed in the temporal domain
        """
        output = array.copy()
        output[1:-1,1:-1] = (array[1:-1,2:]+array[1:-1,:-2]+array[2:,1:-1]+array[:-2,1:-1])/4
        return output
class Interpolate:
    """
    Interpolates the field strength as felt by particles at their current position in the spatial domain from the surrounding PIC grid in both 2 & 3-dimensions.
    """
    def _2D(array,g,i):
        """
        2-dimensional implementation

        **Arguments**

        array
            matrix containing the field that requires interpolating to the particle
        g
            grid_info custom class list containing the grid information for all particle in the simulated domain
        i
            index number of particle which is under-going interpolation
        """
        value = (array[g[i].x_i,g[i].y_i]*g[i].f_ij)+ \
        (array[g[i].x_i+1,g[i].y_i]*g[i].f_i1j)+ \
        (array[g[i].x_i+1,g[i].y_i+1]*g[i].f_i1j1)+ \
        (array[g[i].x_i,g[i].y_i+1]*g[i].f_ij1)
        return value
    def _3D(array,g,i):
        """
        3-dimensional implementation

        **Arguments**

        array
            Matrix containing the field that requires interpolating to the particle
        g
            Grid_info custom class list containing the grid information for all particle in the simulated domain
        i
            Index number of particle which is under-going interpolation
        """
        value = (array[g[i].x_i,g[i].y_i,g[i].z_i]*g[i].f_ijk)+ \
        (array[g[i].x_i+1,g[i].y_i,g[i].z_i]*g[i].f_i1jk)+ \
        (array[g[i].x_i+1,g[i].y_i+1,g[i].z_i]*g[i].f_i1j1k)+ \
        (array[g[i].x_i,g[i].y_i+1,g[i].z_i]*g[i].f_ij1k)+ \
        (array[g[i].x_i,g[i].y_i,g[i].z_i+1]*g[i].f_ijk1)+ \
        (array[g[i].x_i+1,g[i].y_i,g[i].z_i+1]*g[i].f_i1jk1)+ \
        (array[g[i].x_i+1,g[i].y_i+1,g[i].z_i+1]*g[i].f_i1j1k1)+ \
        (array[g[i].x_i,g[i].y_i+1,g[i].z_i+1]*g[i].f_ij1k1)
        return value

def Magnetic_MacCormack(dx,dy,dt,magnetic_field,ion_flow_velocity_x,ion_flow_velocity_y):
    """
    Numerical magnetic field advancer utilising the MacCormack predictor-corrector scheme in conjunction with Lax temporal smoothing to obtain field on next half-integer time step.

    **Arguments**

    dx
        Step size of spatial discretisation in the x domain

    dy
        Step size of spatial discretisation in the y domain

    dt
        Step size of temporal discretisation

    magnetic_field
        Magnetic field on time-step that requires advancing

    ion_flow_velocity_x
        Ion flow velocities in x domain of model co-ordinates on time-step that requires advancing

    ion_flow_velocity_y
        Ion flow velocities in y domain of model co-ordinates on time-step that requires advancing
    """
    mag_predictor = Lax_time_step._2D(magnetic_field) + (dt/2)*magnetic_field*(diff.euler_forward.x(ion_flow_velocity_x,dx)+diff.euler_forward.y(ion_flow_velocity_y,dy))
    mag_output = (1/2)*(Lax_time_step._2D(magnetic_field) + mag_predictor + (dt/2)*mag_predictor*(diff.euler_backward.x(ion_flow_velocity_x,dx)+diff.euler_backward.y(ion_flow_velocity_y,dy)))
    return mag_output
